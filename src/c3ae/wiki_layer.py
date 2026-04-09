"""NovaSpine-backed wiki views compiled from durable facts and syntheses."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from c3ae.utils import iso_str, utcnow


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "untitled"


def _safe_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _relation_label(value: str) -> str:
    return _safe_text((value or "").replace("_", " "))


def _truncate(value: str, limit: int = 220) -> str:
    text = _safe_text(value)
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)]}…"


class NovaSpineWiki:
    """Compiled wiki-style views over NovaSpine facts and syntheses."""

    def __init__(self, spine: Any) -> None:
        self.spine = spine
        self.data_dir = Path(spine.config.data_dir).resolve()
        self.profile_root = self._resolve_profile_root(self.data_dir)
        workspace = self.profile_root / "workspace"
        self.workspace_root = workspace if workspace.exists() else self.data_dir
        self.vault_root = self.workspace_root / "wiki"
        self.entities_dir = self.vault_root / "entities"
        self.syntheses_dir = self.vault_root / "syntheses"
        self.reports_dir = self.vault_root / "reports"
        self.cache_dir = self.vault_root / ".openclaw-wiki" / "cache"
        self.manual_dir = self.vault_root / ".openclaw-wiki" / "manual"

    @staticmethod
    def _resolve_profile_root(data_dir: Path) -> Path:
        parts = data_dir.parts
        if len(parts) >= 3 and tuple(parts[-3:]) == ("sidecars", "novaspine", "data"):
            return data_dir.parent.parent.parent
        if data_dir.name == "novaspine-memory-data":
            return data_dir.parent
        return data_dir

    def _ensure_dirs(self) -> None:
        for target in (
            self.vault_root,
            self.entities_dir,
            self.syntheses_dir,
            self.reports_dir,
            self.cache_dir,
            self.manual_dir,
        ):
            target.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_if_changed(target: Path, content: str) -> None:
        existing = target.read_text("utf-8") if target.exists() else None
        if existing == content:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def _manual_path(self, entity: str) -> Path:
        return self.manual_dir / f"{_slugify(entity)}.json"

    def _load_manual_page(self, entity: str) -> dict[str, Any]:
        target = self._manual_path(entity)
        if not target.exists():
            return {"summary": "", "note": "", "open_questions": [], "tags": [], "updated_at": ""}
        try:
            payload = json.loads(target.read_text("utf-8"))
        except Exception:
            return {"summary": "", "note": "", "open_questions": [], "tags": [], "updated_at": ""}
        return {
            "summary": _safe_text(payload.get("summary")),
            "note": _safe_text(payload.get("note")),
            "open_questions": [
                _safe_text(item)
                for item in payload.get("open_questions", [])
                if _safe_text(item)
            ],
            "tags": [
                _safe_text(item)
                for item in payload.get("tags", [])
                if _safe_text(item)
            ],
            "updated_at": _safe_text(payload.get("updated_at")),
        }

    def apply_page_update(
        self,
        *,
        entity: str,
        summary: str | None = None,
        note: str | None = None,
        open_questions: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        clean_entity = _safe_text(entity)
        if not clean_entity:
            raise ValueError("entity is required")
        self._ensure_dirs()
        current = self._load_manual_page(clean_entity)
        if summary is not None:
            current["summary"] = _safe_text(summary)
        if note is not None:
            current["note"] = _safe_text(note)
        if open_questions is not None:
            current["open_questions"] = [_safe_text(item) for item in open_questions if _safe_text(item)]
        if tags is not None:
            current["tags"] = [_safe_text(item) for item in tags if _safe_text(item)]
        current["updated_at"] = iso_str(utcnow())
        self._manual_path(clean_entity).write_text(f"{json.dumps(current, indent=2)}\n", encoding="utf-8")
        self.compile()
        page = self.get_page(entity=clean_entity)
        if not page:
            raise ValueError(f"Could not build wiki page for {clean_entity}")
        return page

    def _provenance_for_chunk(self, chunk_id: str) -> dict[str, Any]:
        chunk = self.spine.sqlite.get_chunk(chunk_id)
        if not chunk:
            return {
                "chunk_id": chunk_id,
                "chunk_found": False,
                "source_id": "",
                "source_file": "",
                "session_id": "",
                "role": "",
            }
        metadata = dict(chunk.metadata or {})
        return {
            "chunk_id": chunk.id,
            "chunk_found": True,
            "source_id": _safe_text(chunk.source_id),
            "source_file": _safe_text(metadata.get("source_file")),
            "session_id": _safe_text(metadata.get("session_id")),
            "role": _safe_text(metadata.get("role")),
            "created_at": iso_str(chunk.created_at),
            "preview": _truncate(chunk.content, 220),
        }

    def _claim_status(self, fact: dict[str, Any], fallback: str) -> str:
        metadata = dict(fact.get("metadata") or {})
        status = _safe_text(metadata.get("fact_status")).lower()
        return status if status in {"current", "historical"} else fallback

    def _build_page_records(self) -> list[dict[str, Any]]:
        fact_count = max(1, int(self.spine.sqlite.count_structured_facts()))
        groups = self.spine.sqlite.list_structured_truth(limit=max(100, min(2000, fact_count * 6)))
        entity_map: dict[str, dict[str, Any]] = {}
        for group in groups:
            entity = _safe_text(group.get("entity"))
            relation = _safe_text(group.get("relation"))
            if not entity or not relation:
                continue
            slug = _slugify(entity)
            page = entity_map.setdefault(
                slug,
                {
                    "id": f"entity:{slug}",
                    "entity": entity,
                    "slug": slug,
                    "path": f"entities/{slug}.md",
                    "abs_path": str(self.entities_dir / f"{slug}.md"),
                    "relations": [],
                    "current_claims": [],
                    "historical_claims": [],
                },
            )
            current_claims: list[dict[str, Any]] = []
            for fact in group.get("current_facts", []):
                claim = {
                    "id": _safe_text(fact.get("id")),
                    "entity": entity,
                    "relation": relation,
                    "value": _safe_text(fact.get("value")),
                    "date": _safe_text(fact.get("date")),
                    "confidence": float(fact.get("confidence") or 0.0),
                    "status": self._claim_status(fact, "current"),
                    "metadata": dict(fact.get("metadata") or {}),
                    "created_at": _safe_text(fact.get("created_at")),
                    "source_chunk_id": _safe_text(fact.get("source_chunk_id")),
                    "provenance": self._provenance_for_chunk(_safe_text(fact.get("source_chunk_id"))),
                }
                current_claims.append(claim)
                page["current_claims"].append(claim)
            historical_claims: list[dict[str, Any]] = []
            for fact in group.get("historical_facts", []):
                claim = {
                    "id": _safe_text(fact.get("id")),
                    "entity": entity,
                    "relation": relation,
                    "value": _safe_text(fact.get("value")),
                    "date": _safe_text(fact.get("date")),
                    "confidence": float(fact.get("confidence") or 0.0),
                    "status": self._claim_status(fact, "historical"),
                    "metadata": dict(fact.get("metadata") or {}),
                    "created_at": _safe_text(fact.get("created_at")),
                    "source_chunk_id": _safe_text(fact.get("source_chunk_id")),
                    "provenance": self._provenance_for_chunk(_safe_text(fact.get("source_chunk_id"))),
                }
                historical_claims.append(claim)
                page["historical_claims"].append(claim)
            page["relations"].append(
                {
                    "relation": relation,
                    "current_claims": current_claims,
                    "historical_claims": historical_claims,
                    "has_conflict": len({_safe_text(claim["value"]) for claim in current_claims if _safe_text(claim["value"])}) > 1,
                }
            )

        pages = sorted(entity_map.values(), key=lambda item: (len(item["current_claims"]), item["entity"].lower()), reverse=True)
        for page in pages:
            manual = self._load_manual_page(page["entity"])
            page["manual"] = manual
            page["summary"] = manual["summary"] or self._auto_summary(page)
            page["claim_count"] = len(page["current_claims"]) + len(page["historical_claims"])
            page["conflict_relations"] = [
                relation["relation"] for relation in page["relations"] if relation["has_conflict"]
            ]
        return pages

    @staticmethod
    def _auto_summary(page: dict[str, Any]) -> str:
        facts = page.get("current_claims") or []
        if not facts:
            return f"{page.get('entity', 'This entity')} has no current durable claims yet."
        top = []
        for fact in facts[:3]:
            value = _safe_text(fact.get("value"))
            relation = _relation_label(_safe_text(fact.get("relation")))
            if value and relation:
                top.append(f"{relation}: {value}")
        if not top:
            return f"{page.get('entity', 'This entity')} has {len(facts)} current durable claim(s)."
        return "; ".join(top)

    def _render_page(self, page: dict[str, Any], generated_at: str) -> str:
        lines = [
            "---",
            f"title: {page['entity']}",
            f"page_id: {page['id']}",
            f"generated_at: {generated_at}",
            f"current_claims: {len(page['current_claims'])}",
            f"historical_claims: {len(page['historical_claims'])}",
            f"conflict_relations: {json.dumps(page['conflict_relations'])}",
            f"tags: {json.dumps(page['manual']['tags'])}",
            "---",
            "",
            f"# {page['entity']}",
            "",
            page["summary"],
            "",
        ]
        if page["manual"]["note"]:
            lines.extend(["## Curated Notes", "", page["manual"]["note"], ""])
        if page["manual"]["open_questions"]:
            lines.append("## Open Questions")
            lines.append("")
            for question in page["manual"]["open_questions"]:
                lines.append(f"- {question}")
            lines.append("")
        lines.append("## Current Claims")
        lines.append("")
        if page["current_claims"]:
            for claim in page["current_claims"]:
                source = claim["provenance"]
                source_bits = [claim["source_chunk_id"]]
                if source.get("source_file"):
                    source_bits.append(str(source["source_file"]))
                elif source.get("source_id"):
                    source_bits.append(str(source["source_id"]))
                lines.append(
                    f"- `{claim['id'][:8]}` {_relation_label(claim['relation'])}: **{claim['value']}**"
                    f" [confidence={claim['confidence']:.2f}, status={claim['status']}]"
                )
                lines.append(f"  - source: {' | '.join(bit for bit in source_bits if bit)}")
                if source.get("preview"):
                    lines.append(f"  - evidence: {source['preview']}")
        else:
            lines.append("- No current claims yet.")
        lines.append("")
        lines.append("## Historical Claims")
        lines.append("")
        if page["historical_claims"]:
            for claim in page["historical_claims"][:25]:
                lines.append(
                    f"- `{claim['id'][:8]}` {_relation_label(claim['relation'])}: {claim['value']}"
                    f" [date={claim['date'] or 'unknown'}, confidence={claim['confidence']:.2f}]"
                )
        else:
            lines.append("- No historical claims yet.")
        lines.append("")
        if page["conflict_relations"]:
            lines.append("## Conflicts")
            lines.append("")
            for relation in page["conflict_relations"]:
                lines.append(f"- {_relation_label(relation)}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def compile(self) -> dict[str, Any]:
        self._ensure_dirs()
        generated_at = iso_str(utcnow())
        pages = self._build_page_records()
        current_claims = [claim for page in pages for claim in page["current_claims"]]
        historical_claims = [claim for page in pages for claim in page["historical_claims"]]
        conflicts = self.spine.sqlite.list_structured_fact_conflicts(limit=200)
        low_confidence = [
            claim for claim in current_claims if float(claim.get("confidence") or 0.0) < 0.65
        ]
        open_questions = [
            {"entity": page["entity"], "questions": page["manual"]["open_questions"]}
            for page in pages
            if page["manual"]["open_questions"]
        ]

        self._write_if_changed(
            self.vault_root / "AGENTS.md",
            "# NovaSpine Wiki\n\nThis vault is compiled from NovaSpine durable memory.\n",
        )
        self._write_if_changed(
            self.vault_root / "WIKI.md",
            "# NovaSpine Wiki\n\nUse entity pages for durable claims, reports for health checks, and cache files for machine-readable digests.\n",
        )

        index_lines = [
            "# NovaSpine Wiki Index",
            "",
            f"- Generated at: {generated_at}",
            f"- Entity pages: {len(pages)}",
            f"- Current claims: {len(current_claims)}",
            f"- Historical claims: {len(historical_claims)}",
            f"- Conflicts: {len(conflicts)}",
            "",
            "## Entities",
            "",
        ]
        if pages:
            for page in pages:
                index_lines.append(
                    f"- [{page['entity']}]({page['path']})"
                    f" — {page['claim_count']} claims, {len(page['conflict_relations'])} conflict relation(s)"
                )
        else:
            index_lines.append("- No durable entity pages have been compiled yet.")
        self._write_if_changed(self.vault_root / "index.md", "\n".join(index_lines).rstrip() + "\n")

        for page in pages:
            self._write_if_changed(self.entities_dir / f"{page['slug']}.md", self._render_page(page, generated_at))

        syntheses = self.spine.sqlite.list_consolidated_memories(limit=25)
        synthesis_lines = [
            "# Recent Syntheses",
            "",
            f"- Generated at: {generated_at}",
            "",
        ]
        if syntheses:
            for item in syntheses:
                synthesis_lines.extend(
                    [
                        f"## {item['id'][:8]}",
                        "",
                        _safe_text(item.get("summary")),
                        "",
                    ]
                )
        else:
            synthesis_lines.append("No consolidated memories yet.")
        self._write_if_changed(self.syntheses_dir / "recent-consolidated.md", "\n".join(synthesis_lines).rstrip() + "\n")

        contradiction_lines = [
            "# Contradictions",
            "",
            f"- Generated at: {generated_at}",
            "",
        ]
        if conflicts:
            for group in conflicts:
                values = ", ".join(sorted({_safe_text(item.get('value')) for item in group.get("current_facts", []) if _safe_text(item.get("value"))}))
                contradiction_lines.append(
                    f"- **{group['entity']} / {_relation_label(group['relation'])}** — {values}"
                )
        else:
            contradiction_lines.append("No current contradictions detected.")
        self._write_if_changed(self.reports_dir / "contradictions.md", "\n".join(contradiction_lines).rstrip() + "\n")

        low_conf_lines = [
            "# Low Confidence Claims",
            "",
            f"- Generated at: {generated_at}",
            "",
        ]
        if low_confidence:
            for claim in low_confidence[:100]:
                low_conf_lines.append(
                    f"- `{claim['id'][:8]}` {claim['entity']} / {_relation_label(claim['relation'])} = {claim['value']} "
                    f"(confidence={claim['confidence']:.2f})"
                )
        else:
            low_conf_lines.append("No low-confidence current claims detected.")
        self._write_if_changed(self.reports_dir / "low-confidence.md", "\n".join(low_conf_lines).rstrip() + "\n")

        questions_lines = [
            "# Open Questions",
            "",
            f"- Generated at: {generated_at}",
            "",
        ]
        if open_questions:
            for item in open_questions:
                questions_lines.append(f"## {item['entity']}")
                questions_lines.append("")
                for question in item["questions"]:
                    questions_lines.append(f"- {question}")
                questions_lines.append("")
        else:
            questions_lines.append("No curated open questions yet.")
        self._write_if_changed(self.reports_dir / "open-questions.md", "\n".join(questions_lines).rstrip() + "\n")

        health_lines = [
            "# Claim Health",
            "",
            f"- Generated at: {generated_at}",
            f"- Entity pages: {len(pages)}",
            f"- Current claims: {len(current_claims)}",
            f"- Historical claims: {len(historical_claims)}",
            f"- Conflicts: {len(conflicts)}",
            f"- Low confidence: {len(low_confidence)}",
            "",
        ]
        self._write_if_changed(self.reports_dir / "claim-health.md", "\n".join(health_lines).rstrip() + "\n")

        digest = {
            "generated_at": generated_at,
            "page_count": len(pages),
            "current_claim_count": len(current_claims),
            "historical_claim_count": len(historical_claims),
            "conflict_count": len(conflicts),
            "low_confidence_count": len(low_confidence),
            "open_question_count": sum(len(item["questions"]) for item in open_questions),
            "pages": [
                {
                    "id": page["id"],
                    "entity": page["entity"],
                    "path": page["path"],
                    "summary": page["summary"],
                    "current_claim_count": len(page["current_claims"]),
                    "historical_claim_count": len(page["historical_claims"]),
                    "conflict_relations": page["conflict_relations"],
                    "tags": page["manual"]["tags"],
                }
                for page in pages[:50]
            ],
        }
        self._write_if_changed(self.cache_dir / "agent-digest.json", f"{json.dumps(digest, indent=2)}\n")

        claims_target = self.cache_dir / "claims.jsonl"
        claim_lines = []
        for claim in [*current_claims, *historical_claims]:
            claim_lines.append(
                json.dumps(
                    {
                        "id": claim["id"],
                        "entity": claim["entity"],
                        "relation": claim["relation"],
                        "value": claim["value"],
                        "status": claim["status"],
                        "confidence": claim["confidence"],
                        "date": claim["date"],
                        "path": f"entities/{_slugify(claim['entity'])}.md",
                        "source_chunk_id": claim["source_chunk_id"],
                        "generated_at": generated_at,
                    },
                    sort_keys=True,
                )
            )
        self._write_if_changed(claims_target, ("\n".join(claim_lines) + "\n") if claim_lines else "")

        return {
            "service": "novaspine",
            "generated_at": generated_at,
            "vault_root": str(self.vault_root),
            "entity_pages": len(pages),
            "current_claims": len(current_claims),
            "historical_claims": len(historical_claims),
            "conflicts": len(conflicts),
            "low_confidence": len(low_confidence),
            "open_questions": sum(len(item["questions"]) for item in open_questions),
            "reports": {
                "contradictions": str(self.reports_dir / "contradictions.md"),
                "low_confidence": str(self.reports_dir / "low-confidence.md"),
                "open_questions": str(self.reports_dir / "open-questions.md"),
                "claim_health": str(self.reports_dir / "claim-health.md"),
            },
            "cache": {
                "agent_digest": str(self.cache_dir / "agent-digest.json"),
                "claims_jsonl": str(self.cache_dir / "claims.jsonl"),
            },
        }

    def search(self, query: str, *, limit: int = 10) -> dict[str, Any]:
        status = self.compile()
        pages = self._build_page_records()
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        query_tokens = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 1]

        for page in pages:
            haystack = " ".join(
                [
                    page["entity"],
                    page["summary"],
                    " ".join(claim["value"] for claim in page["current_claims"][:12]),
                    " ".join(claim["relation"] for claim in page["current_claims"][:12]),
                    " ".join(page["manual"]["tags"]),
                ]
            ).lower()
            score = sum(1 for token in query_tokens if token in haystack)
            if not score:
                continue
            key = ("page", page["id"])
            seen.add(key)
            results.append(
                {
                    "kind": "page",
                    "id": page["id"],
                    "title": page["entity"],
                    "path": page["path"],
                    "score": float(200 + (score * 10) + len(page["current_claims"])),
                    "preview": page["summary"],
                    "metadata": {
                        "current_claims": len(page["current_claims"]),
                        "historical_claims": len(page["historical_claims"]),
                        "conflict_relations": page["conflict_relations"],
                    },
                }
            )

        for entity in self.spine.sqlite.search_entities(query, limit=max(limit, 5)):
            key = ("page", _safe_text(entity.get("normalized_name")))
            if key in seen:
                continue
            seen.add(key)
            name = _safe_text(entity.get("name"))
            slug = _slugify(name)
            results.append(
                {
                    "kind": "page",
                    "id": f"entity:{slug}",
                    "title": name,
                    "path": f"entities/{slug}.md",
                    "score": 180.0,
                    "preview": f"Entity page for {name}",
                    "metadata": dict(entity.get("metadata") or {}),
                }
            )

        for result in self.spine.sqlite.search_structured_facts_fts(query, limit=max(limit * 2, 10)):
            metadata = dict(result.metadata or {})
            entity = _safe_text(metadata.get("entity"))
            relation = _safe_text(metadata.get("relation"))
            value = _safe_text(metadata.get("value")) or _truncate(result.content, 180)
            key = ("claim", result.id)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "kind": "claim",
                    "id": result.id,
                    "title": f"{entity} / {_relation_label(relation)}" if entity and relation else "Durable claim",
                    "path": f"entities/{_slugify(entity)}.md" if entity else "",
                    "score": float(120.0 + float(result.score)),
                    "preview": value,
                    "metadata": metadata,
                }
            )

        for result in self.spine.sqlite.search_consolidated_fts(query, limit=max(limit, 5)):
            key = ("synthesis", result.id)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "kind": "synthesis",
                    "id": result.id,
                    "title": f"Consolidated memory {result.id[:8]}",
                    "path": "syntheses/recent-consolidated.md",
                    "score": float(40.0 + float(result.score)),
                    "preview": _truncate(result.content, 220),
                    "metadata": dict(result.metadata or {}),
                }
            )

        results.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return {
            "ok": True,
            "query": query,
            "count": min(len(results), max(1, limit)),
            "results": results[: max(1, limit)],
            "status": status,
        }

    def get_page(
        self,
        *,
        entity: str = "",
        path: str = "",
        claim_id: str = "",
    ) -> dict[str, Any] | None:
        self.compile()
        target_entity = _safe_text(entity)
        if claim_id:
            fact = self.spine.sqlite.get_structured_fact(claim_id)
            if not fact:
                return None
            target_entity = _safe_text(fact.get("entity"))
        if path and not target_entity:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = self.vault_root / candidate
            if candidate.exists():
                text = candidate.read_text("utf-8")
                return {
                    "ok": True,
                    "id": _safe_text(candidate.stem),
                    "entity": "",
                    "path": str(candidate.relative_to(self.vault_root)),
                    "absolute_path": str(candidate),
                    "content": text,
                    "claims": [],
                    "manual": {},
                }
            return None
        if not target_entity:
            return None
        pages = {page["entity"].lower(): page for page in self._build_page_records()}
        page = pages.get(target_entity.lower())
        if not page:
            manual = self._load_manual_page(target_entity)
            page = {
                "id": f"entity:{_slugify(target_entity)}",
                "entity": target_entity,
                "slug": _slugify(target_entity),
                "path": f"entities/{_slugify(target_entity)}.md",
                "abs_path": str(self.entities_dir / f"{_slugify(target_entity)}.md"),
                "relations": [],
                "current_claims": [],
                "historical_claims": [],
                "manual": manual,
                "summary": manual["summary"] or f"{target_entity} has no current durable claims yet.",
                "claim_count": 0,
                "conflict_relations": [],
            }
        page_path = Path(page["abs_path"])
        content = page_path.read_text("utf-8") if page_path.exists() else self._render_page(page, iso_str(utcnow()))
        return {
            "ok": True,
            "id": page["id"],
            "entity": page["entity"],
            "title": page["entity"],
            "path": page["path"],
            "absolute_path": str(page_path),
            "content": content,
            "summary": page["summary"],
            "claims": [*page["current_claims"], *page["historical_claims"]],
            "current_claims": page["current_claims"],
            "historical_claims": page["historical_claims"],
            "conflict_relations": page["conflict_relations"],
            "manual": page["manual"],
        }

    def lint(self, *, limit: int = 50) -> dict[str, Any]:
        status = self.compile()
        conflicts = self.spine.sqlite.list_structured_fact_conflicts(limit=max(1, limit))
        pages = self._build_page_records()
        current_claims = [claim for page in pages for claim in page["current_claims"]]
        low_confidence = [
            {
                "id": claim["id"],
                "entity": claim["entity"],
                "relation": claim["relation"],
                "value": claim["value"],
                "confidence": claim["confidence"],
                "path": f"entities/{_slugify(claim['entity'])}.md",
            }
            for claim in current_claims
            if float(claim.get("confidence") or 0.0) < 0.65
        ]
        missing_evidence = [
            {
                "id": claim["id"],
                "entity": claim["entity"],
                "relation": claim["relation"],
                "value": claim["value"],
                "source_chunk_id": claim["source_chunk_id"],
                "path": f"entities/{_slugify(claim['entity'])}.md",
            }
            for claim in current_claims
            if not bool(claim["provenance"].get("chunk_found"))
        ]
        return {
            "ok": True,
            "status": status,
            "counts": {
                "conflicts": len(conflicts),
                "low_confidence": len(low_confidence),
                "missing_evidence": len(missing_evidence),
                "open_questions": status["open_questions"],
            },
            "conflicts": conflicts[: max(1, limit)],
            "low_confidence": low_confidence[: max(1, limit)],
            "missing_evidence": missing_evidence[: max(1, limit)],
            "reports": status["reports"],
        }
