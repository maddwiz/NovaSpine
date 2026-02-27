from __future__ import annotations

import asyncio
import json

from c3ae.config import Config
from c3ae.llm.venice_chat import ChatResponse
from c3ae.memory_spine.spine import MemorySpine


class _FakeChat:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    async def chat(self, messages, temperature=None, max_tokens=None, json_mode=False):  # noqa: ANN001
        payload = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return ChatResponse(content=payload)

    async def close(self) -> None:
        return None

    @property
    def stats(self) -> dict:
        return {"calls": self._idx}


def test_llm_policy_noop_and_delete_actions(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.memory_manager.use_llm_policy = True
        cfg.ensure_dirs()
        spine = MemorySpine(cfg)
        try:
            ev = spine.add_evidence(
                claim="Desmond prefers concise status updates.",
                sources=["unit-test"],
                confidence=0.95,
                reasoning="seed evidence",
            )
            first = await spine.add_knowledge(
                title="Preference",
                content="Desmond prefers concise status updates.",
                evidence_ids=[ev.id],
            )

            # LLM policy routes duplicate to NOOP.
            spine.write_manager._chat = _FakeChat(
                [f'{{"action":"NOOP","target_id":"{first.id}","reason":"duplicate"}}']
            )
            duplicate = await spine.add_knowledge(
                title="Preference",
                content="Desmond prefers concise status updates.",
                evidence_ids=[ev.id],
            )
            assert duplicate.id == first.id

            # LLM policy can retract stale entry and allow a fresh ADD.
            spine.write_manager._chat = _FakeChat(
                [f'{{"action":"DELETE","target_id":"{first.id}","reason":"stale"}}']
            )
            replacement = await spine.add_knowledge(
                title="Preference",
                content="Desmond now prefers weekly status summaries.",
                evidence_ids=[ev.id],
            )
            assert replacement.id != first.id
            first_after = spine.bank.get(first.id)
            assert first_after is not None
            assert first_after.status.value == "retracted"
            assert replacement.status.value == "active"
        finally:
            await spine.close()

    asyncio.run(_run())


def test_learned_policy_file_routes_duplicate_to_noop(tmp_path):
    async def _run() -> None:
        policy_path = tmp_path / "policy.json"
        policy_path.write_text(
            json.dumps(
                {
                    "memory_manager": {
                        "actions": ["ADD", "UPDATE", "DELETE", "NOOP"],
                        "feature_order": [
                            "similarity",
                            "title_similarity",
                            "content_similarity",
                            "evidence_ratio",
                            "length_norm",
                            "negation",
                        ],
                        "weights": [
                            [-3.0, -2.0, -2.0, 0.0, 0.0, 0.0],  # ADD
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],      # UPDATE
                            [-1.0, -1.0, -1.0, 0.0, 0.0, 2.0],   # DELETE
                            [4.0, 2.0, 3.0, 0.0, 0.0, 0.0],      # NOOP
                        ],
                        "bias": [0.0, 0.0, -0.5, 0.0],
                    }
                }
            ),
            encoding="utf-8",
        )

        cfg = Config()
        cfg.data_dir = tmp_path / "data"
        cfg.memory_manager.use_learned_policy = True
        cfg.memory_manager.learned_policy_path = str(policy_path)
        cfg.memory_manager.learned_policy_min_confidence = 0.2
        cfg.ensure_dirs()
        spine = MemorySpine(cfg)
        try:
            ev = spine.add_evidence(
                claim="User prefers concise updates.",
                sources=["unit-test"],
                confidence=0.9,
                reasoning="seed",
            )
            first = await spine.add_knowledge(
                title="Preference",
                content="User prefers concise updates.",
                evidence_ids=[ev.id],
            )
            second = await spine.add_knowledge(
                title="Preference",
                content="User prefers concise updates.",
                evidence_ids=[ev.id],
            )
            assert second.id == first.id
        finally:
            await spine.close()

    asyncio.run(_run())
