"""LLM-driven agent reasoning loop.

The LLM is the decision engine. C3 is the constraint layer.
Each turn: LLM reads COS → decides action → C3 validates → C3 stores → COS updates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from c3ae.llm.backends import ChatBackend
from c3ae.llm.venice_chat import Message
from c3ae.memory_spine.spine import MemorySpine
from c3ae.types import SearchResult


# ── Structured action types the LLM can emit ────────────────────────────

@dataclass
class AgentAction:
    action: str  # "search", "write_knowledge", "add_evidence", "think", "done"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTurn:
    step: int
    reasoning: str
    action: AgentAction
    context: list[SearchResult] = field(default_factory=list)
    llm_response: str = ""
    cos_snapshot: str = ""


@dataclass
class AgentResult:
    session_id: str
    task: str
    turns: list[AgentTurn] = field(default_factory=list)
    final_answer: str = ""
    llm_stats: dict[str, Any] = field(default_factory=dict)
    knowledge_written: list[str] = field(default_factory=list)
    evidence_created: list[str] = field(default_factory=list)
    governance_blocks: list[str] = field(default_factory=list)


SYSTEM_PROMPT = """You are a reasoning agent with access to a persistent memory system.
You work through tasks step by step. At each step, you MUST respond with a JSON object containing:

{
  "reasoning": "Your thinking about what to do next",
  "action": "search" | "write_knowledge" | "add_evidence" | "think" | "done",
  "params": { ... action-specific parameters ... }
}

Actions:
- "search": Search memory for relevant information.
  params: {"query": "search terms"}
- "add_evidence": Record a verified claim with sources.
  params: {"claim": "...", "sources": ["..."], "confidence": 0.0-1.0, "reasoning": "..."}
- "write_knowledge": Write a verified conclusion to the knowledge bank.
  params: {"title": "...", "content": "...", "tags": ["..."], "evidence_ids": ["..."]}
- "think": Continue reasoning without taking action.
  params: {"thought": "..."}
- "done": Task is complete.
  params: {"answer": "final answer"}

Rules:
- Before writing knowledge, you MUST first add evidence with add_evidence.
- Use evidence_ids from your add_evidence actions in write_knowledge.
- Search memory before making claims.
- Stay focused on the original task. Do not drift.
- Be specific and factual. Do not hallucinate.

Respond ONLY with the JSON object. No other text."""


class AgentLoop:
    """LLM-driven reasoning loop constrained by C3 memory."""

    def __init__(
        self,
        spine: MemorySpine,
        chat: ChatBackend,
        max_turns: int = 20,
        identity: str | None = None,
    ) -> None:
        self.spine = spine
        self.chat = chat
        self.max_turns = max_turns
        self.identity = identity

    async def run(self, task: str, metadata: dict[str, Any] | None = None) -> AgentResult:
        """Execute the full agent loop."""
        session_id = uuid4().hex
        self.spine.start_session(session_id, metadata)
        self.spine.cos.create(
            session_id, f"Task started: {task}",
            open_questions=[task],
        )

        result = AgentResult(session_id=session_id, task=task)
        messages = self._build_initial_messages(task, session_id)

        for step in range(self.max_turns):
            # Get COS for context
            cos_prompt = self.spine.cos.render_prompt(session_id)

            # Update system message with current COS
            messages[0] = Message(
                role="system",
                content=self._build_system_content(cos_prompt),
            )

            # Call LLM
            response = await self.chat.chat(messages, json_mode=True)
            action = self._parse_action(response.content)

            turn = AgentTurn(
                step=step,
                reasoning=action.params.get("reasoning", action.params.get("thought", "")),
                action=action,
                llm_response=response.content,
                cos_snapshot=cos_prompt,
            )

            # Execute action through C3
            action_result = await self._execute_action(action, session_id, result)

            # Update COS
            new_facts = []
            resolved = []
            if action.action == "write_knowledge":
                title = action.params.get("title", "")
                new_facts.append(f"Wrote knowledge: {title}")
            elif action.action == "add_evidence":
                claim = action.params.get("claim", "")
                new_facts.append(f"Evidence recorded: {claim[:80]}")
            elif action.action == "done":
                resolved = [task]

            reasoning_text = action.params.get("reasoning", action.params.get("thought", ""))
            self.spine.cos.update(
                session_id,
                f"Step {step}: {action.action} — {reasoning_text[:100]}",
                new_facts=new_facts if new_facts else None,
                resolved_questions=resolved if resolved else None,
            )

            # Add assistant response and action result to conversation
            messages.append(Message(role="assistant", content=response.content))
            messages.append(Message(role="user", content=action_result))

            result.turns.append(turn)

            if action.action == "done":
                result.final_answer = action.params.get("answer", "")
                break

        # Finalize
        self.spine.end_session(session_id)
        result.llm_stats = self.chat.stats

        # Store session log
        self.spine.vault.store_raw_log(
            self._render_log(result),
            session_id,
            "agent_session.md",
        )

        return result

    def _build_system_content(self, cos_prompt: str) -> str:
        parts = [SYSTEM_PROMPT]
        if self.identity:
            parts.append(f"\n\n## Identity\n{self.identity}")
        if cos_prompt:
            parts.append(f"\n\n{cos_prompt}")
        return "\n".join(parts)

    def _build_initial_messages(self, task: str, session_id: str) -> list[Message]:
        return [
            Message(role="system", content=self._build_system_content("")),
            Message(role="user", content=f"Task: {task}\n\nBegin reasoning. Respond with a JSON action."),
        ]

    def _parse_action(self, content: str) -> AgentAction:
        """Parse LLM JSON response into an AgentAction."""
        try:
            # Try to extract JSON from the response
            text = content.strip()
            # Handle markdown code blocks
            if "```json" in text:
                text = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL).group(1)
            elif "```" in text:
                text = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL).group(1)
            data = json.loads(text)
            return AgentAction(
                action=data.get("action", "think"),
                params={
                    "reasoning": data.get("reasoning", ""),
                    **data.get("params", {}),
                },
            )
        except (json.JSONDecodeError, AttributeError):
            return AgentAction(action="think", params={"thought": content, "reasoning": "Failed to parse JSON"})

    async def _execute_action(
        self, action: AgentAction, session_id: str, result: AgentResult
    ) -> str:
        """Execute an action through C3 and return feedback for the LLM."""
        if action.action == "search":
            query = action.params.get("query", "")
            results = await self.spine.search(query, top_k=10)
            if results:
                formatted = "\n".join(
                    f"[{i+1}] (score={r.score:.3f}) {r.content[:200]}"
                    for i, r in enumerate(results[:5])
                )
                return f"Search results for '{query}':\n{formatted}"
            return f"No results found for '{query}'. Memory is empty on this topic."

        elif action.action == "add_evidence":
            claim = action.params.get("claim", "")
            sources = action.params.get("sources", [])
            confidence = action.params.get("confidence", 0.5)
            reasoning = action.params.get("reasoning", "")
            pack = self.spine.add_evidence(claim, sources, confidence, reasoning)
            result.evidence_created.append(pack.id)
            return f"Evidence recorded (id={pack.id}). Claim: '{claim}'. You can reference this evidence_id in write_knowledge."

        elif action.action == "write_knowledge":
            title = action.params.get("title", "")
            content_text = action.params.get("content", "")
            tags = action.params.get("tags", [])
            evidence_ids = action.params.get("evidence_ids", [])
            try:
                entry = await self.spine.add_knowledge(
                    title=title, content=content_text,
                    tags=tags, evidence_ids=evidence_ids,
                    session_id=session_id,
                )
                result.knowledge_written.append(entry.id)
                return f"Knowledge written successfully (id={entry.id}): '{title}'"
            except Exception as e:
                result.governance_blocks.append(str(e))
                return f"GOVERNANCE BLOCKED: {e}. You must provide evidence_ids from previous add_evidence actions."

        elif action.action == "think":
            return "Continue reasoning. What's your next action?"

        elif action.action == "done":
            return "Session complete."

        return "Unknown action. Use: search, add_evidence, write_knowledge, think, or done."

    def _render_log(self, result: AgentResult) -> str:
        parts = [f"# Agent Session: {result.task}\n"]
        parts.append(f"Session ID: {result.session_id}")
        parts.append(f"Turns: {len(result.turns)}")
        parts.append(f"Knowledge written: {len(result.knowledge_written)}")
        parts.append(f"Evidence created: {len(result.evidence_created)}")
        parts.append(f"Governance blocks: {len(result.governance_blocks)}")
        parts.append(f"LLM stats: {result.llm_stats}")
        parts.append("")
        for turn in result.turns:
            parts.append(f"## Turn {turn.step}: {turn.action.action}")
            parts.append(f"Reasoning: {turn.reasoning}")
            parts.append(f"Raw: {turn.llm_response[:300]}")
            parts.append("")
        if result.final_answer:
            parts.append(f"## Final Answer\n{result.final_answer}")
        return "\n".join(parts)
