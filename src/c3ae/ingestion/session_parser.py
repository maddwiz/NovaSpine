"""Parse Claude Code and OpenClaw session JSONL files into searchable chunks.

Supports two formats:
- Claude Code: messages with type "user"/"assistant", content in message.content
- OpenClaw: messages with type "message", role in message.role
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionChunk:
    """A meaningful chunk extracted from a session transcript."""
    role: str           # "user", "assistant", "tool_call", "tool_result"
    content: str        # The actual text content
    session_id: str     # Source session identifier
    source_file: str    # Original file path
    index: int          # Position in session (for ordering)
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionParser:
    """Parse agent session JSONL files into searchable chunks."""

    # Skip tool results longer than this (they're usually file dumps)
    MAX_TOOL_RESULT_LEN = 2000
    # Skip very short content
    MIN_CONTENT_LEN = 20

    def parse_file(self, path: Path) -> list[SessionChunk]:
        """Parse a session file, auto-detecting format."""
        lines = path.read_text(errors="replace").strip().split("\n")
        if not lines:
            return []

        # Detect format from first valid JSON line
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                first = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        else:
            return []

        session_id = path.stem
        source_file = str(path)

        # OpenClaw format has "type":"session" or "type":"message" with message.role
        if first.get("type") == "session" or (
            first.get("type") == "message" and "message" in first and "role" in first.get("message", {})
        ):
            return self._parse_openclaw(lines, session_id, source_file)

        # Claude Code format has "type":"user"/"assistant" with message.role
        if first.get("type") in ("user", "assistant", "file-history-snapshot"):
            return self._parse_claude_code(lines, session_id, source_file)

        # Generic JSONL format:
        # {"role":"user|assistant", "content":"..."}
        if "role" in first and "content" in first:
            return self._parse_role_content_jsonl(lines, session_id, source_file)

        # Unknown format — try to extract any text content
        return self._parse_generic(lines, session_id, source_file)

    def _parse_openclaw(self, lines: list[str], session_id: str, source_file: str) -> list[SessionChunk]:
        """Parse OpenClaw session format."""
        chunks = []
        idx = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("type") != "message":
                continue

            msg = entry.get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                text = self._extract_text(content)
                if text and len(text) >= self.MIN_CONTENT_LEN:
                    chunks.append(SessionChunk(
                        role="user", content=text,
                        session_id=session_id, source_file=source_file,
                        index=idx,
                    ))
                    idx += 1

            elif role == "assistant":
                text = self._extract_text(content)
                # Extract tool calls separately
                tool_calls = self._extract_tool_calls(content)
                if text and len(text) >= self.MIN_CONTENT_LEN:
                    chunks.append(SessionChunk(
                        role="assistant", content=text,
                        session_id=session_id, source_file=source_file,
                        index=idx,
                    ))
                    idx += 1
                for tc in tool_calls:
                    chunks.append(SessionChunk(
                        role="tool_call", content=tc,
                        session_id=session_id, source_file=source_file,
                        index=idx,
                    ))
                    idx += 1

            elif role == "toolResult":
                text = self._extract_text(content)
                if text and self.MIN_CONTENT_LEN <= len(text) <= self.MAX_TOOL_RESULT_LEN:
                    tool_name = msg.get("toolName", "unknown")
                    chunks.append(SessionChunk(
                        role="tool_result", content=text,
                        session_id=session_id, source_file=source_file,
                        index=idx,
                        metadata={"tool": tool_name},
                    ))
                    idx += 1

        return chunks

    def _parse_claude_code(self, lines: list[str], session_id: str, source_file: str) -> list[SessionChunk]:
        """Parse Claude Code session format."""
        chunks = []
        idx = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")
            msg = entry.get("message", {})

            if entry_type == "user":
                content = msg.get("content", "")
                text = self._extract_text(content)
                if text and len(text) >= self.MIN_CONTENT_LEN:
                    chunks.append(SessionChunk(
                        role="user", content=text,
                        session_id=session_id, source_file=source_file,
                        index=idx,
                    ))
                    idx += 1

            elif entry_type == "assistant":
                content = msg.get("content", "")
                text = self._extract_text(content)
                tool_calls = self._extract_tool_calls(content)
                if text and len(text) >= self.MIN_CONTENT_LEN:
                    chunks.append(SessionChunk(
                        role="assistant", content=text,
                        session_id=session_id, source_file=source_file,
                        index=idx,
                    ))
                    idx += 1
                for tc in tool_calls:
                    chunks.append(SessionChunk(
                        role="tool_call", content=tc,
                        session_id=session_id, source_file=source_file,
                        index=idx,
                    ))
                    idx += 1

        return chunks

    def _parse_generic(self, lines: list[str], session_id: str, source_file: str) -> list[SessionChunk]:
        """Fallback parser for unknown formats — extract any text content."""
        chunks = []
        idx = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = self._extract_text_deep(entry)
            if text and len(text) >= self.MIN_CONTENT_LEN:
                chunks.append(SessionChunk(
                    role="unknown", content=text[:5000],
                    session_id=session_id, source_file=source_file,
                    index=idx,
                ))
                idx += 1
        return chunks

    def _parse_role_content_jsonl(
        self, lines: list[str], session_id: str, source_file: str
    ) -> list[SessionChunk]:
        """Parse generic JSONL chat rows with role/content fields."""
        chunks = []
        idx = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = str(entry.get("role", "unknown")).strip().lower()
            content = self._extract_text(entry.get("content", ""))
            if not content or len(content) < self.MIN_CONTENT_LEN:
                continue

            if role in {"tool", "tool_call", "tool_use"}:
                normalized_role = "tool_call"
            elif role in {"tool_result", "toolresponse", "tool-response"}:
                normalized_role = "tool_result"
            elif role in {"assistant", "user", "system"}:
                normalized_role = role
            else:
                normalized_role = "unknown"

            chunks.append(
                SessionChunk(
                    role=normalized_role,
                    content=content,
                    session_id=session_id,
                    source_file=source_file,
                    index=idx,
                    metadata={
                        k: v
                        for k, v in entry.items()
                        if k not in {"role", "content"}
                    },
                )
            )
            idx += 1
        return chunks

    def _extract_text(self, content: Any) -> str:
        """Extract plain text from message content (string or content blocks)."""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        # Claude Code tool result format
                        inner = block.get("content", "")
                        if isinstance(inner, str) and len(inner) <= self.MAX_TOOL_RESULT_LEN:
                            parts.append(inner)
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts).strip()
        return ""

    def _extract_tool_calls(self, content: Any) -> list[str]:
        """Extract tool call descriptions from content blocks."""
        calls = []
        if not isinstance(content, list):
            return calls
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                name = block.get("name", "unknown")
                args = block.get("input", {})
                # Create a concise summary
                summary = f"Tool: {name}"
                if isinstance(args, dict):
                    for k, v in args.items():
                        val = str(v)[:200]
                        summary += f"\n  {k}: {val}"
                calls.append(summary)
            elif block.get("type") == "toolCall":
                name = block.get("name", "unknown")
                args_str = block.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {"raw": args_str}
                summary = f"Tool: {name}"
                if isinstance(args, dict):
                    for k, v in args.items():
                        val = str(v)[:200]
                        summary += f"\n  {k}: {val}"
                calls.append(summary)
        return calls

    def _extract_text_deep(self, obj: Any, max_depth: int = 3) -> str:
        """Recursively extract text from nested structures."""
        if max_depth <= 0:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            # Prefer "content", "text", "summary" keys
            for key in ("content", "text", "summary", "message"):
                if key in obj:
                    result = self._extract_text_deep(obj[key], max_depth - 1)
                    if result:
                        return result
        if isinstance(obj, list):
            parts = []
            for item in obj[:10]:  # Limit list traversal
                result = self._extract_text_deep(item, max_depth - 1)
                if result:
                    parts.append(result)
            return "\n".join(parts)
        return ""
