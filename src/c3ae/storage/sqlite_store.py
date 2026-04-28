"""SQLite storage with FTS5 full-text search."""

from __future__ import annotations

from datetime import timedelta
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from c3ae.types import (
    AuditEvent,
    CarryOverSummary,
    Chunk,
    EvidencePack,
    ReasoningEntry,
    SearchResult,
    SkillCapsule,
)
from c3ae.utils import (
    extract_benchmark_case_token,
    iso_str,
    json_dumps,
    json_loads,
    parse_iso,
    utcnow,
)

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS cos (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    summary TEXT NOT NULL,
    key_facts TEXT NOT NULL DEFAULT '[]',
    open_questions TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_cos_session ON cos(session_id, sequence);

CREATE TABLE IF NOT EXISTS reasoning_bank (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    evidence_ids TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by TEXT,
    session_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rb_status ON reasoning_bank(status);

CREATE VIRTUAL TABLE IF NOT EXISTS reasoning_bank_fts USING fts5(
    title, content, tags,
    content=reasoning_bank, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS evidence_packs (
    id TEXT PRIMARY KEY,
    claim TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.0,
    reasoning TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_capsules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    procedure TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    version INTEGER NOT NULL DEFAULT 1,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS skill_capsules_fts USING fts5(
    name, description, procedure, tags,
    content=skill_capsules, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content=chunks, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT 'ok',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(created_at);

CREATE TABLE IF NOT EXISTS memory_access (
    memory_id TEXT PRIMARY KEY,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL DEFAULT 'unknown',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_chunk ON entity_mentions(chunk_id);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    src_entity_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    dst_entity_id TEXT NOT NULL,
    source_chunk_id TEXT NOT NULL DEFAULT '',
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    confidence REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'active',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (src_entity_id) REFERENCES entities(id),
    FOREIGN KEY (dst_entity_id) REFERENCES entities(id)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_entity_id, status);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_entity_id, status);
CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(relation, status);

CREATE TABLE IF NOT EXISTS consolidated_memories (
    id TEXT PRIMARY KEY,
    cluster_key TEXT NOT NULL,
    summary TEXT NOT NULL,
    facts TEXT NOT NULL DEFAULT '[]',
    source_chunk_ids TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_consolidated_cluster ON consolidated_memories(cluster_key);

CREATE VIRTUAL TABLE IF NOT EXISTS consolidated_memories_fts USING fts5(
    summary, facts,
    content=consolidated_memories, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS structured_facts (
    id TEXT PRIMARY KEY,
    source_chunk_id TEXT NOT NULL,
    entity TEXT NOT NULL,
    relation TEXT NOT NULL,
    value TEXT NOT NULL,
    fact_date TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.0,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_structured_facts_chunk ON structured_facts(source_chunk_id);
CREATE INDEX IF NOT EXISTS idx_structured_facts_entity ON structured_facts(entity);
CREATE INDEX IF NOT EXISTS idx_structured_facts_relation ON structured_facts(relation);
CREATE INDEX IF NOT EXISTS idx_structured_facts_date ON structured_facts(fact_date);

CREATE VIRTUAL TABLE IF NOT EXISTS structured_facts_fts USING fts5(
    entity, relation, value,
    content=structured_facts, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    mime_type TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""

# FTS5 trigger SQL — keeps FTS in sync with content tables
_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS reasoning_bank_ai AFTER INSERT ON reasoning_bank BEGIN
    INSERT INTO reasoning_bank_fts(rowid, title, content, tags)
    VALUES (new.rowid, new.title, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS reasoning_bank_ad AFTER DELETE ON reasoning_bank BEGIN
    INSERT INTO reasoning_bank_fts(reasoning_bank_fts, rowid, title, content, tags)
    VALUES ('delete', old.rowid, old.title, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS reasoning_bank_au AFTER UPDATE ON reasoning_bank BEGIN
    INSERT INTO reasoning_bank_fts(reasoning_bank_fts, rowid, title, content, tags)
    VALUES ('delete', old.rowid, old.title, old.content, old.tags);
    INSERT INTO reasoning_bank_fts(rowid, title, content, tags)
    VALUES (new.rowid, new.title, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS skill_capsules_ai AFTER INSERT ON skill_capsules BEGIN
    INSERT INTO skill_capsules_fts(rowid, name, description, procedure, tags)
    VALUES (new.rowid, new.name, new.description, new.procedure, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS skill_capsules_ad AFTER DELETE ON skill_capsules BEGIN
    INSERT INTO skill_capsules_fts(skill_capsules_fts, rowid, name, description, procedure, tags)
    VALUES ('delete', old.rowid, old.name, old.description, old.procedure, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS skill_capsules_au AFTER UPDATE ON skill_capsules BEGIN
    INSERT INTO skill_capsules_fts(skill_capsules_fts, rowid, name, description, procedure, tags)
    VALUES ('delete', old.rowid, old.name, old.description, old.procedure, old.tags);
    INSERT INTO skill_capsules_fts(rowid, name, description, procedure, tags)
    VALUES (new.rowid, new.name, new.description, new.procedure, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content)
    VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
    INSERT INTO chunks_fts(rowid, content)
    VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS consolidated_memories_ai AFTER INSERT ON consolidated_memories BEGIN
    INSERT INTO consolidated_memories_fts(rowid, summary, facts)
    VALUES (new.rowid, new.summary, new.facts);
END;
CREATE TRIGGER IF NOT EXISTS consolidated_memories_ad AFTER DELETE ON consolidated_memories BEGIN
    INSERT INTO consolidated_memories_fts(consolidated_memories_fts, rowid, summary, facts)
    VALUES ('delete', old.rowid, old.summary, old.facts);
END;
CREATE TRIGGER IF NOT EXISTS consolidated_memories_au AFTER UPDATE ON consolidated_memories BEGIN
    INSERT INTO consolidated_memories_fts(consolidated_memories_fts, rowid, summary, facts)
    VALUES ('delete', old.rowid, old.summary, old.facts);
    INSERT INTO consolidated_memories_fts(rowid, summary, facts)
    VALUES (new.rowid, new.summary, new.facts);
END;

CREATE TRIGGER IF NOT EXISTS structured_facts_ai AFTER INSERT ON structured_facts BEGIN
    INSERT INTO structured_facts_fts(rowid, entity, relation, value)
    VALUES (new.rowid, new.entity, new.relation, new.value);
END;
CREATE TRIGGER IF NOT EXISTS structured_facts_ad AFTER DELETE ON structured_facts BEGIN
    INSERT INTO structured_facts_fts(structured_facts_fts, rowid, entity, relation, value)
    VALUES ('delete', old.rowid, old.entity, old.relation, old.value);
END;
CREATE TRIGGER IF NOT EXISTS structured_facts_au AFTER UPDATE ON structured_facts BEGIN
    INSERT INTO structured_facts_fts(structured_facts_fts, rowid, entity, relation, value)
    VALUES ('delete', old.rowid, old.entity, old.relation, old.value);
    INSERT INTO structured_facts_fts(rowid, entity, relation, value)
    VALUES (new.rowid, new.entity, new.relation, new.value);
END;
"""


def _sanitize_fts_query(query: str) -> str:
    """Sanitize a query for FTS5 MATCH syntax.

    Strips special FTS5 operators and builds token-safe MATCH queries.
    For longer natural-language queries, use OR semantics to avoid
    over-constraining recall.
    """
    # Remove FTS5 special chars while preserving alnum/underscore tokens.
    cleaned = re.sub(r"[^\w\s]", " ", query)
    raw_tokens = cleaned.split()
    tokens: list[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        t = tok.strip().strip("_")
        if not t:
            continue
        # Ignore ultra-short tokens except numeric anchors like years.
        if len(t) < 2 and not t.isdigit():
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(t)
    if not tokens:
        return '""'
    # Cap term count to keep MATCH plans predictable.
    tokens = tokens[:24]

    def _fts_term(tok: str, use_prefix: bool) -> str:
        t = tok.lower()
        if use_prefix and t.isalpha() and len(t) >= 4:
            return f"{t}*"
        return t

    case_terms = [t for t in tokens if re.fullmatch(r"[a-z0-9_]*case_\d+", t.lower())]
    if case_terms:
        # Keep benchmark case token mandatory to avoid cross-case contamination.
        case_expr = _benchmark_case_fts_expression(case_terms[0])
        if not case_expr:
            case_expr = _fts_term(case_terms[0], use_prefix=False)
        tail = [t for t in tokens if t != case_terms[0]]
        if not tail:
            return case_expr
        if len(tail) <= 3:
            tail_query = " ".join(_fts_term(t, use_prefix=False) for t in tail)
        else:
            tail_query = " OR ".join(_fts_term(t, use_prefix=True) for t in tail)
        return f"({case_expr}) AND ({tail_query})"

    # Precision for terse queries (IDs/names). Broad recall for long NL queries.
    if len(tokens) <= 3:
        return " ".join(_fts_term(t, use_prefix=False) for t in tokens)
    return " OR ".join(_fts_term(t, use_prefix=True) for t in tokens)


def _benchmark_case_fts_expression(case_token: str) -> str:
    token = (case_token or "").strip().strip("_").lower()
    if not token:
        return ""
    parts = [p for p in re.split(r"[_\s]+", token) if p]
    if not parts:
        return ""
    # unicode61 tokenizes "__DMR_CASE_0042__" into separate terms ("dmr", "case", "0042"),
    # so the benchmark filter must require those terms together.
    return " ".join(f'"{p}"' for p in parts)


def _normalize_entity_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    return re.sub(r"[^a-z0-9\-\._ ]+", "", cleaned)


class SQLiteStore:
    """Main SQLite storage backend."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False,
                                     timeout=30.0)
        self._write_lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._init_schema()

    def _commit_locked(self, retries: int = 3) -> None:
        """Commit with retry on database-locked errors.

        Caller must already hold ``self._write_lock``.
        """
        import time as _time
        for attempt in range(retries):
            try:
                self._conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < retries - 1:
                    _time.sleep(0.1 * (2 ** attempt))
                    continue
                raise

    def _commit(self, retries: int = 3) -> None:
        with self._write_lock:
            self._commit_locked(retries)

    def _execute_write(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._write_lock:
            cur = self._conn.execute(sql, params)
            self._commit_locked()
            return cur

    def _init_schema(self) -> None:
        import time as _time
        for attempt in range(5):
            try:
                with self._write_lock:
                    cur = self._conn.cursor()
                    cur.executescript(_SCHEMA)
                    cur.executescript(_FTS_TRIGGERS)
                    cur.execute(
                        "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                        ("schema_version", str(SCHEMA_VERSION)),
                    )
                    self._commit_locked()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < 4:
                    _time.sleep(2 * (attempt + 1))
                    continue
                raise

    def close(self) -> None:
        self._conn.close()

    # --- Sessions ---

    def create_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> str:
        self._execute_write(
            "INSERT INTO sessions(id, started_at, metadata) VALUES (?, ?, ?)",
            (session_id, iso_str(utcnow()), json_dumps(metadata or {})),
        )
        return session_id

    def end_session(self, session_id: str) -> None:
        self._execute_write(
            "UPDATE sessions SET ended_at=? WHERE id=?",
            (iso_str(utcnow()), session_id),
        )

    # --- Chunks ---

    def insert_chunk(self, chunk: Chunk) -> str:
        self._execute_write(
            "INSERT INTO chunks(id, source_id, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (chunk.id, chunk.source_id, chunk.content, json_dumps(chunk.metadata), iso_str(chunk.created_at)),
        )
        return chunk.id

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        if not row:
            return None
        return self._row_to_chunk(row)

    def get_chunk_rowid(self, chunk_id: str) -> int | None:
        row = self._conn.execute("SELECT rowid FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return row[0] if row else None

    def get_chunk_id_by_rowid(self, rowid: int) -> str | None:
        row = self._conn.execute("SELECT id FROM chunks WHERE rowid=?", (rowid,)).fetchone()
        return row[0] if row else None

    def delete_chunk(self, chunk_id: str) -> bool:
        cur = self._execute_write("DELETE FROM chunks WHERE id=?", (chunk_id,))
        return cur.rowcount > 0

    def search_chunks_fts(self, query: str, limit: int = 20) -> list[SearchResult]:
        def _run(match_query: str) -> list[sqlite3.Row]:
            return self._conn.execute(
                """SELECT c.id, c.content, c.metadata, c.created_at,
                          bm25(chunks_fts) AS score
                   FROM chunks_fts f
                   JOIN chunks c ON c.rowid = f.rowid
                   WHERE chunks_fts MATCH ?
                   ORDER BY score
                   LIMIT ?""",
                (match_query, limit),
            ).fetchall()

        fts_query = _sanitize_fts_query(query)
        rows = _run(fts_query)
        # Benchmark-case queries can miss when tail terms don't appear in the source chunk.
        # Fall back to case-token-only retrieval to preserve deterministic case scoping.
        if not rows:
            case_token = extract_benchmark_case_token(query)
            if case_token:
                fallback_query = _benchmark_case_fts_expression(case_token)
                if fallback_query:
                    if fallback_query != fts_query:
                        rows = _run(fallback_query)
        return [
            SearchResult(
                id=r["id"],
                content=r["content"],
                score=-r["score"],  # bm25 returns negative scores, lower = better
                source="fts5",
                metadata={
                    **json_loads(r["metadata"]),
                    "_created_at": r["created_at"],
                    "_source_kind": "chunk",
                },
            )
            for r in rows
        ]

    def count_chunks(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0]

    def list_chunk_ids(self, limit: int = 1_000_000) -> list[dict[str, str]]:
        rows = self._conn.execute(
            "SELECT id FROM chunks ORDER BY created_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [{"id": str(r["id"])} for r in rows]

    def list_chunks(
        self,
        limit: int = 200,
        older_than: str | None = None,
        session_id: str | None = None,
    ) -> list[Chunk]:
        where: list[str] = []
        params: list[Any] = []
        if older_than:
            where.append("c.created_at <= ?")
            params.append(older_than)
        if session_id:
            where.append("json_extract(c.metadata, '$.session_id') = ?")
            params.append(session_id)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        rows = self._conn.execute(
            f"""SELECT c.*
                FROM chunks c
                {where_sql}
                ORDER BY c.created_at DESC
                LIMIT ?""",
            (*params, limit),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def list_chunks_with_access(
        self,
        limit: int = 200,
        older_than: str | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if older_than:
            where.append("c.created_at <= ?")
            params.append(older_than)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        rows = self._conn.execute(
            f"""SELECT c.id, c.source_id, c.content, c.metadata, c.created_at,
                       COALESCE(ma.access_count, 0) AS access_count
                FROM chunks c
                LEFT JOIN memory_access ma ON ma.memory_id = c.id
                {where_sql}
                ORDER BY c.created_at ASC
                LIMIT ?""",
            (*params, limit),
        ).fetchall()
        return [
            {
                "id": str(r["id"]),
                "source_id": str(r["source_id"]),
                "content": str(r["content"]),
                "metadata": json_loads(r["metadata"]),
                "created_at": str(r["created_at"]),
                "access_count": int(r["access_count"]),
            }
            for r in rows
        ]

    # --- Structured Facts ---

    def insert_structured_fact(
        self,
        *,
        source_chunk_id: str,
        entity: str,
        relation: str,
        value: str,
        fact_date: str = "",
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        entity_clean = re.sub(r"\s+", " ", (entity or "").strip())
        relation_clean = re.sub(r"\s+", "_", (relation or "").strip().lower())
        value_clean = re.sub(r"\s+", " ", (value or "").strip())
        if not source_chunk_id or not entity_clean or not relation_clean or not value_clean:
            raise ValueError("structured fact requires source_chunk_id/entity/relation/value")
        now = iso_str(utcnow())

        # Upsert-by-content for idempotent re-ingestion.
        with self._write_lock:
            row = self._conn.execute(
                """SELECT id FROM structured_facts
                   WHERE source_chunk_id=? AND entity=? AND relation=? AND value=? AND fact_date=?""",
                (
                    source_chunk_id,
                    entity_clean,
                    relation_clean,
                    value_clean,
                    fact_date.strip(),
                ),
            ).fetchone()
            if row:
                fact_id = str(row["id"])
                self._conn.execute(
                    """UPDATE structured_facts
                       SET confidence=?, metadata=?, created_at=?
                       WHERE id=?""",
                    (
                        float(confidence),
                        json_dumps(metadata or {}),
                        now,
                        fact_id,
                    ),
                )
                self._commit_locked()
                return fact_id

            fact_id = uuid4().hex
            self._conn.execute(
                """INSERT INTO structured_facts(
                    id, source_chunk_id, entity, relation, value, fact_date, confidence, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact_id,
                    source_chunk_id,
                    entity_clean,
                    relation_clean,
                    value_clean,
                    fact_date.strip(),
                    float(confidence),
                    json_dumps(metadata or {}),
                    now,
                ),
            )
            self._commit_locked()
            return fact_id

    def list_structured_facts(
        self,
        *,
        entity: str = "",
        relation: str = "",
        source_chunk_id: str = "",
        order_by_date_desc: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if entity.strip():
            where.append("entity = ?")
            params.append(re.sub(r"\s+", " ", entity.strip()))
        if relation.strip():
            where.append("relation = ?")
            params.append(re.sub(r"\s+", "_", relation.strip().lower()))
        if source_chunk_id.strip():
            where.append("source_chunk_id = ?")
            params.append(source_chunk_id.strip())
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        order_sql = "ORDER BY fact_date DESC, created_at DESC" if order_by_date_desc else "ORDER BY created_at DESC"
        rows = self._conn.execute(
            f"""SELECT id, source_chunk_id, entity, relation, value, fact_date, confidence, metadata, created_at
                FROM structured_facts
                {where_sql}
                {order_sql}
                LIMIT ?""",
            (*params, max(1, int(limit))),
        ).fetchall()
        return [
            {
                "id": str(r["id"]),
                "source_chunk_id": str(r["source_chunk_id"]),
                "entity": str(r["entity"]),
                "relation": str(r["relation"]),
                "value": str(r["value"]),
                "date": str(r["fact_date"]),
                "confidence": float(r["confidence"]),
                "metadata": json_loads(r["metadata"]),
                "created_at": str(r["created_at"]),
            }
            for r in rows
        ]

    @staticmethod
    def _fact_status(fact: dict[str, Any]) -> str:
        metadata = dict(fact.get("metadata") or {})
        status = str(metadata.get("fact_status", "")).strip().lower()
        return status if status in {"current", "historical"} else ""

    @staticmethod
    def _fact_sort_key(fact: dict[str, Any]) -> tuple[str, str, float]:
        return (
            str(fact.get("date") or ""),
            str(fact.get("created_at") or ""),
            float(fact.get("confidence") or 0.0),
        )

    def _facts_by_group(
        self,
        *,
        entity: str = "",
        relation: str = "",
        limit: int = 200,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for fact in self.list_structured_facts(
            entity=entity,
            relation=relation,
            order_by_date_desc=True,
            limit=max(1, limit),
        ):
            key = (str(fact["entity"]).lower(), str(fact["relation"]).lower())
            groups.setdefault(key, []).append(fact)
        return groups

    def get_structured_fact(self, fact_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """SELECT id, source_chunk_id, entity, relation, value, fact_date, confidence, metadata, created_at
               FROM structured_facts
               WHERE id=?""",
            (fact_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": str(row["id"]),
            "source_chunk_id": str(row["source_chunk_id"]),
            "entity": str(row["entity"]),
            "relation": str(row["relation"]),
            "value": str(row["value"]),
            "date": str(row["fact_date"]),
            "confidence": float(row["confidence"]),
            "metadata": json_loads(row["metadata"]),
            "created_at": str(row["created_at"]),
        }

    def update_structured_fact_metadata(self, fact_id: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
        now = iso_str(utcnow())
        self._execute_write(
            "UPDATE structured_facts SET metadata=?, created_at=? WHERE id=?",
            (json_dumps(metadata), now, fact_id),
        )
        return self.get_structured_fact(fact_id)

    def list_current_structured_facts(
        self,
        *,
        entity: str = "",
        relation: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        current: list[dict[str, Any]] = []
        for facts in self._facts_by_group(entity=entity, relation=relation, limit=max(limit * 4, 200)).values():
            explicit_current = [fact for fact in facts if self._fact_status(fact) == "current"]
            if explicit_current:
                winners = explicit_current
            else:
                latest_date = next((str(fact.get("date") or "") for fact in facts if str(fact.get("date") or "")), "")
                if latest_date:
                    winners = [fact for fact in facts if str(fact.get("date") or "") == latest_date]
                else:
                    winners = [sorted(facts, key=self._fact_sort_key, reverse=True)[0]]
            current.extend(sorted(winners, key=self._fact_sort_key, reverse=True))
        current.sort(key=self._fact_sort_key, reverse=True)
        return current[: max(1, limit)]

    def list_structured_truth(
        self,
        *,
        entity: str = "",
        relation: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for (_, _), facts in self._facts_by_group(entity=entity, relation=relation, limit=max(limit * 6, 200)).items():
            current = self.list_current_structured_facts(
                entity=str(facts[0]["entity"]),
                relation=str(facts[0]["relation"]),
                limit=max(len(facts), 4),
            )
            current_ids = {str(fact["id"]) for fact in current}
            historical = [fact for fact in facts if str(fact["id"]) not in current_ids]
            groups.append(
                {
                    "entity": str(facts[0]["entity"]),
                    "relation": str(facts[0]["relation"]),
                    "current_facts": current,
                    "historical_facts": sorted(historical, key=self._fact_sort_key, reverse=True),
                }
            )
        groups.sort(
            key=lambda item: max(
                [self._fact_sort_key(fact) for fact in [*item["current_facts"], *item["historical_facts"]]],
                default=("", "", 0.0),
            ),
            reverse=True,
        )
        return groups[: max(1, limit)]

    def list_structured_fact_conflicts(self, *, limit: int = 50) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        for group in self.list_structured_truth(limit=max(limit * 4, 100)):
            current = list(group["current_facts"])
            values = {str(fact.get("value") or "").strip() for fact in current if str(fact.get("value") or "").strip()}
            if len(values) <= 1:
                continue
            conflicts.append(
                {
                    "entity": group["entity"],
                    "relation": group["relation"],
                    "value_count": len(values),
                    "current_facts": current,
                    "historical_facts": group["historical_facts"],
                }
            )
        return conflicts[: max(1, limit)]

    def resolve_structured_fact_conflict(
        self,
        *,
        winner_fact_id: str,
        loser_fact_ids: list[str],
        reason: str = "",
        user_confirmation: str = "",
        resolution_ticket_id: str = "",
    ) -> dict[str, Any]:
        winner = self.get_structured_fact(winner_fact_id)
        if not winner:
            raise ValueError(f"Unknown fact id: {winner_fact_id}")
        related = [winner]
        for fact_id in loser_fact_ids:
            fact = self.get_structured_fact(fact_id)
            if not fact:
                raise ValueError(f"Unknown fact id: {fact_id}")
            related.append(fact)
        entity = str(winner["entity"]).lower()
        relation = str(winner["relation"]).lower()
        for fact in related[1:]:
            if str(fact["entity"]).lower() != entity or str(fact["relation"]).lower() != relation:
                raise ValueError("All resolved facts must share the same entity and relation")

        resolved_at = iso_str(utcnow())
        resolution_id = resolution_ticket_id or uuid4().hex
        updated_winner_meta = dict(winner.get("metadata") or {})
        updated_winner_meta.update(
            {
                "fact_status": "current",
                "resolved_at": resolved_at,
                "resolution_id": resolution_id,
                "resolution_reason": reason,
                "user_confirmation": user_confirmation,
            }
        )
        updated_winner = self.update_structured_fact_metadata(winner_fact_id, updated_winner_meta)
        superseded: list[dict[str, Any]] = []
        for fact in related[1:]:
            metadata = dict(fact.get("metadata") or {})
            metadata.update(
                {
                    "fact_status": "historical",
                    "resolved_at": resolved_at,
                    "resolution_id": resolution_id,
                    "resolution_reason": reason,
                    "user_confirmation": user_confirmation,
                    "superseded_by_fact_id": winner_fact_id,
                }
            )
            updated = self.update_structured_fact_metadata(str(fact["id"]), metadata)
            if updated:
                superseded.append(updated)
        return {
            "winner_fact": updated_winner,
            "superseded_facts": superseded,
            "resolved_at": resolved_at,
            "resolution_id": resolution_id,
        }

    def search_structured_facts_fts(self, query: str, limit: int = 20) -> list[SearchResult]:
        fts_query = _sanitize_fts_query(query)
        rows = self._conn.execute(
            """SELECT sf.id, sf.source_chunk_id, sf.entity, sf.relation, sf.value,
                      sf.fact_date, sf.confidence, sf.metadata, sf.created_at,
                      bm25(structured_facts_fts) AS score
               FROM structured_facts_fts f
               JOIN structured_facts sf ON sf.rowid = f.rowid
               WHERE structured_facts_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (fts_query, limit),
        ).fetchall()
        return [
            SearchResult(
                id=str(r["id"]),
                content=f"{r['entity']} {r['relation']} {r['value']}",
                score=-float(r["score"]),
                source="structured_fact",
                metadata={
                    **json_loads(r["metadata"]),
                    "source_chunk_id": str(r["source_chunk_id"]),
                    "entity": str(r["entity"]),
                    "relation": str(r["relation"]),
                    "value": str(r["value"]),
                    "fact_date": str(r["fact_date"]),
                    "fact_confidence": float(r["confidence"]),
                    "_created_at": str(r["created_at"]),
                    "_source_kind": "structured_fact",
                },
            )
            for r in rows
        ]

    def count_structured_facts(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM structured_facts").fetchone()
        return int(row["n"]) if row else 0

    # --- Reasoning Bank ---

    def insert_reasoning_entry(self, entry: ReasoningEntry) -> str:
        self._execute_write(
            """INSERT INTO reasoning_bank(id, title, content, tags, evidence_ids,
               status, superseded_by, session_id, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id, entry.title, entry.content,
                json_dumps(entry.tags), json_dumps(entry.evidence_ids),
                entry.status.value, entry.superseded_by, entry.session_id,
                json_dumps(entry.metadata), iso_str(entry.created_at),
            ),
        )
        return entry.id

    def get_reasoning_entry(self, entry_id: str) -> ReasoningEntry | None:
        row = self._conn.execute("SELECT * FROM reasoning_bank WHERE id=?", (entry_id,)).fetchone()
        if not row:
            return None
        return self._row_to_reasoning_entry(row)

    def supersede_reasoning_entry(self, old_id: str, new_entry: ReasoningEntry) -> str:
        with self._write_lock:
            self._conn.execute(
                "UPDATE reasoning_bank SET status='superseded', superseded_by=? WHERE id=?",
                (new_entry.id, old_id),
            )
            self._conn.execute(
                """INSERT INTO reasoning_bank(id, title, content, tags, evidence_ids,
                   status, superseded_by, session_id, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_entry.id,
                    new_entry.title,
                    new_entry.content,
                    json_dumps(new_entry.tags),
                    json_dumps(new_entry.evidence_ids),
                    new_entry.status.value,
                    new_entry.superseded_by,
                    new_entry.session_id,
                    json_dumps(new_entry.metadata),
                    iso_str(new_entry.created_at),
                ),
            )
            self._commit_locked()
            return new_entry.id

    def list_reasoning_entries(self, status: str = "active", limit: int = 100) -> list[ReasoningEntry]:
        rows = self._conn.execute(
            "SELECT * FROM reasoning_bank WHERE status=? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
        return [self._row_to_reasoning_entry(r) for r in rows]

    def list_reasoning_entries_any_status(self, limit: int = 1_000_000) -> list[ReasoningEntry]:
        rows = self._conn.execute(
            "SELECT * FROM reasoning_bank ORDER BY created_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [self._row_to_reasoning_entry(r) for r in rows]

    def list_reasoning_chunk_links(self, limit: int = 1_000_000) -> list[dict[str, str]]:
        rows = self._conn.execute(
            """SELECT id, json_extract(metadata, '$.reasoning_entry_id') AS reasoning_entry_id
               FROM chunks
               WHERE json_extract(metadata, '$.type') = 'reasoning_entry'
               LIMIT ?""",
            (max(1, int(limit)),),
        ).fetchall()
        return [
            {
                "chunk_id": str(r["id"]),
                "reasoning_entry_id": str(r["reasoning_entry_id"] or ""),
            }
            for r in rows
        ]

    def mark_reasoning_chunks_superseded(self, reasoning_entry_id: str, superseded_by: str) -> int:
        rows = self._conn.execute(
            """SELECT id, metadata FROM chunks
               WHERE json_extract(metadata, '$.reasoning_entry_id') = ?""",
            (reasoning_entry_id,),
        ).fetchall()
        if not rows:
            return 0
        with self._write_lock:
            for row in rows:
                metadata = json_loads(row["metadata"])
                metadata["entry_status"] = "superseded"
                metadata["superseded_by"] = superseded_by
                self._conn.execute(
                    "UPDATE chunks SET metadata=? WHERE id=?",
                    (json_dumps(metadata), row["id"]),
                )
            self._commit_locked()
        return len(rows)

    def search_reasoning_fts(self, query: str, limit: int = 20) -> list[SearchResult]:
        fts_query = _sanitize_fts_query(query)
        rows = self._conn.execute(
            """SELECT r.id, r.content, r.title, r.metadata, r.created_at,
                      bm25(reasoning_bank_fts) AS score
               FROM reasoning_bank_fts f
               JOIN reasoning_bank r ON r.rowid = f.rowid
               WHERE reasoning_bank_fts MATCH ?
               AND r.status = 'active'
               ORDER BY score
               LIMIT ?""",
            (fts_query, limit),
        ).fetchall()
        return [
            SearchResult(
                id=r["id"],
                content=r["content"],
                score=-r["score"],
                source="reasoning_bank",
                metadata={
                    **json_loads(r["metadata"]),
                    "_created_at": r["created_at"],
                    "_source_kind": "reasoning_entry",
                },
            )
            for r in rows
        ]

    # --- Evidence Packs ---

    def insert_evidence_pack(self, pack: EvidencePack) -> str:
        self._execute_write(
            """INSERT INTO evidence_packs(id, claim, sources, confidence, reasoning, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                pack.id, pack.claim, json_dumps(pack.sources),
                pack.confidence, pack.reasoning,
                json_dumps(pack.metadata), iso_str(pack.created_at),
            ),
        )
        return pack.id

    def get_evidence_pack(self, pack_id: str) -> EvidencePack | None:
        row = self._conn.execute("SELECT * FROM evidence_packs WHERE id=?", (pack_id,)).fetchone()
        if not row:
            return None
        return EvidencePack(
            id=row["id"],
            claim=row["claim"],
            sources=json_loads(row["sources"]),
            confidence=row["confidence"],
            reasoning=row["reasoning"],
            metadata=json_loads(row["metadata"]),
            created_at=parse_iso(row["created_at"]),
        )

    # --- COS ---

    def insert_cos(self, cos: CarryOverSummary) -> str:
        self._execute_write(
            """INSERT INTO cos(id, session_id, sequence, summary, key_facts, open_questions, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cos.id, cos.session_id, cos.sequence, cos.summary,
                json_dumps(cos.key_facts), json_dumps(cos.open_questions),
                json_dumps(cos.metadata), iso_str(cos.created_at),
            ),
        )
        return cos.id

    def get_latest_cos(self, session_id: str) -> CarryOverSummary | None:
        row = self._conn.execute(
            "SELECT * FROM cos WHERE session_id=? ORDER BY sequence DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_cos(row)

    def list_cos(self, session_id: str) -> list[CarryOverSummary]:
        rows = self._conn.execute(
            "SELECT * FROM cos WHERE session_id=? ORDER BY sequence ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_cos(r) for r in rows]

    # --- Skill Capsules ---

    def insert_skill_capsule(self, capsule: SkillCapsule) -> str:
        self._execute_write(
            """INSERT INTO skill_capsules(id, name, description, procedure, tags, version, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                capsule.id, capsule.name, capsule.description, capsule.procedure,
                json_dumps(capsule.tags), capsule.version,
                json_dumps(capsule.metadata), iso_str(capsule.created_at),
            ),
        )
        return capsule.id

    def get_skill_capsule(self, capsule_id: str) -> SkillCapsule | None:
        row = self._conn.execute("SELECT * FROM skill_capsules WHERE id=?", (capsule_id,)).fetchone()
        if not row:
            return None
        return self._row_to_skill_capsule(row)

    def search_skills_fts(self, query: str, limit: int = 10) -> list[SearchResult]:
        fts_query = _sanitize_fts_query(query)
        rows = self._conn.execute(
            """SELECT s.id, s.description, s.name, s.metadata,
                      bm25(skill_capsules_fts) AS score
               FROM skill_capsules_fts f
               JOIN skill_capsules s ON s.rowid = f.rowid
               WHERE skill_capsules_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (fts_query, limit),
        ).fetchall()
        return [
            SearchResult(
                id=r["id"],
                content=r["description"],
                score=-r["score"],
                source="skill_capsules",
                metadata=json_loads(r["metadata"]),
            )
            for r in rows
        ]

    def list_skill_capsules(self, limit: int = 100) -> list[SkillCapsule]:
        rows = self._conn.execute(
            "SELECT * FROM skill_capsules ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_skill_capsule(r) for r in rows]

    # --- Consolidated Memories ---

    def upsert_consolidated_memory(
        self,
        cluster_key: str,
        summary: str,
        facts: list[str],
        source_chunk_ids: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        now = iso_str(utcnow())
        with self._write_lock:
            row = self._conn.execute(
                "SELECT id FROM consolidated_memories WHERE cluster_key=?",
                (cluster_key,),
            ).fetchone()
            if row:
                memory_id = str(row["id"])
                self._conn.execute(
                    """UPDATE consolidated_memories
                       SET summary=?, facts=?, source_chunk_ids=?, metadata=?, updated_at=?
                       WHERE id=?""",
                    (
                        summary,
                        json_dumps(facts),
                        json_dumps(source_chunk_ids),
                        json_dumps(metadata or {}),
                        now,
                        memory_id,
                    ),
                )
                self._commit_locked()
                return memory_id
            memory_id = uuid4().hex
            self._conn.execute(
                """INSERT INTO consolidated_memories(
                    id, cluster_key, summary, facts, source_chunk_ids, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory_id,
                    cluster_key,
                    summary,
                    json_dumps(facts),
                    json_dumps(source_chunk_ids),
                    json_dumps(metadata or {}),
                    now,
                    now,
                ),
            )
            self._commit_locked()
            return memory_id

    def search_consolidated_fts(self, query: str, limit: int = 20) -> list[SearchResult]:
        fts_query = _sanitize_fts_query(query)
        rows = self._conn.execute(
            """SELECT c.id, c.summary, c.facts, c.metadata, c.updated_at,
                      bm25(consolidated_memories_fts) AS score
               FROM consolidated_memories_fts f
               JOIN consolidated_memories c ON c.rowid = f.rowid
               WHERE consolidated_memories_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (fts_query, limit),
        ).fetchall()
        out: list[SearchResult] = []
        for r in rows:
            facts = json_loads(r["facts"])
            fact_text = "; ".join(str(x) for x in facts[:5]) if isinstance(facts, list) else ""
            content = f"{r['summary']} | facts: {fact_text}" if fact_text else str(r["summary"])
            out.append(
                SearchResult(
                    id=str(r["id"]),
                    content=content,
                    score=-float(r["score"]),
                    source="consolidated",
                    metadata={
                        **json_loads(r["metadata"]),
                        "_created_at": str(r["updated_at"]),
                        "_source_kind": "consolidated",
                    },
                )
            )
        return out

    def list_consolidated_memories(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT id, cluster_key, summary, facts, source_chunk_ids, metadata, created_at, updated_at
               FROM consolidated_memories
               ORDER BY updated_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "id": str(r["id"]),
                "cluster_key": str(r["cluster_key"]),
                "summary": str(r["summary"]),
                "facts": json_loads(r["facts"]),
                "source_chunk_ids": json_loads(r["source_chunk_ids"]),
                "metadata": json_loads(r["metadata"]),
                "created_at": str(r["created_at"]),
                "updated_at": str(r["updated_at"]),
            }
            for r in rows
        ]

    def count_consolidated_memories(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM consolidated_memories").fetchone()
        return int(row["n"]) if row else 0

    # --- Knowledge Graph ---

    def upsert_entity(
        self,
        name: str,
        entity_type: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        normalized = _normalize_entity_name(name)
        if not normalized:
            raise ValueError("Entity name cannot be empty")
        now = iso_str(utcnow())
        with self._write_lock:
            row = self._conn.execute(
                "SELECT id, metadata FROM entities WHERE normalized_name=?",
                (normalized,),
            ).fetchone()
            if row:
                entity_id = str(row["id"])
                current_meta = json_loads(row["metadata"])
                merged_meta = {**current_meta, **(metadata or {})}
                self._conn.execute(
                    "UPDATE entities SET name=?, entity_type=?, metadata=?, updated_at=? WHERE id=?",
                    (name.strip(), entity_type, json_dumps(merged_meta), now, entity_id),
                )
                self._commit_locked()
                return entity_id

            entity_id = uuid4().hex
            self._conn.execute(
                """INSERT INTO entities(id, name, normalized_name, entity_type, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entity_id,
                    name.strip(),
                    normalized,
                    entity_type,
                    json_dumps(metadata or {}),
                    now,
                    now,
                ),
            )
            self._commit_locked()
            return entity_id

    def get_entity_by_name(self, name: str) -> dict[str, Any] | None:
        normalized = _normalize_entity_name(name)
        if not normalized:
            return None
        row = self._conn.execute(
            """SELECT id, name, normalized_name, entity_type, metadata, created_at, updated_at
               FROM entities
               WHERE normalized_name=?""",
            (normalized,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": str(row["id"]),
            "name": str(row["name"]),
            "normalized_name": str(row["normalized_name"]),
            "entity_type": str(row["entity_type"]),
            "metadata": json_loads(row["metadata"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def search_entities(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        token = _normalize_entity_name(query)
        if not token:
            return []
        rows = self._conn.execute(
            """SELECT id, name, normalized_name, entity_type, metadata, created_at, updated_at
               FROM entities
               WHERE normalized_name LIKE ?
               ORDER BY LENGTH(name) ASC
               LIMIT ?""",
            (f"%{token}%", limit),
        ).fetchall()
        return [
            {
                "id": str(r["id"]),
                "name": str(r["name"]),
                "normalized_name": str(r["normalized_name"]),
                "entity_type": str(r["entity_type"]),
                "metadata": json_loads(r["metadata"]),
                "created_at": str(r["created_at"]),
                "updated_at": str(r["updated_at"]),
            }
            for r in rows
        ]

    def add_entity_mention(self, entity_id: str, chunk_id: str, confidence: float = 0.5) -> str:
        mention_id = uuid4().hex
        self._execute_write(
            """INSERT INTO entity_mentions(id, entity_id, chunk_id, confidence, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (mention_id, entity_id, chunk_id, float(confidence), iso_str(utcnow())),
        )
        return mention_id

    def get_entities_for_chunk_ids(self, chunk_ids: list[str]) -> dict[str, set[str]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._conn.execute(
            f"""SELECT m.chunk_id, e.name
                FROM entity_mentions m
                JOIN entities e ON e.id = m.entity_id
                WHERE m.chunk_id IN ({placeholders})""",
            tuple(chunk_ids),
        ).fetchall()
        out: dict[str, set[str]] = {cid: set() for cid in chunk_ids}
        for row in rows:
            cid = str(row["chunk_id"])
            name = str(row["name"]).strip().lower()
            if not name:
                continue
            out.setdefault(cid, set()).add(name)
        return out

    def add_edge(
        self,
        src_entity_id: str,
        relation: str,
        dst_entity_id: str,
        source_chunk_id: str = "",
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
        invalidate_existing_relation: bool = False,
    ) -> str:
        now = iso_str(utcnow())
        relation_norm = relation.strip().lower().replace(" ", "_")
        with self._write_lock:
            if invalidate_existing_relation:
                self._conn.execute(
                    """UPDATE edges
                       SET status='inactive', valid_to=?
                       WHERE src_entity_id=? AND relation=? AND status='active' AND dst_entity_id != ?""",
                    (now, src_entity_id, relation_norm, dst_entity_id),
                )
            edge_id = uuid4().hex
            self._conn.execute(
                """INSERT INTO edges(
                    id, src_entity_id, relation, dst_entity_id, source_chunk_id,
                    valid_from, valid_to, confidence, status, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, 'active', ?, ?)""",
                (
                    edge_id,
                    src_entity_id,
                    relation_norm,
                    dst_entity_id,
                    source_chunk_id,
                    now,
                    float(confidence),
                    json_dumps(metadata or {}),
                    now,
                ),
            )
            self._commit_locked()
            return edge_id

    def query_graph(self, entity: str, depth: int = 2, limit: int = 200) -> dict[str, Any]:
        seed = self.get_entity_by_name(entity)
        if not seed:
            matches = self.search_entities(entity, limit=1)
            if not matches:
                return {"entity": entity, "depth": depth, "nodes": [], "edges": []}
            seed = matches[0]

        depth = max(1, int(depth))
        frontier = {seed["id"]}
        visited = set(frontier)
        collected_edges: list[dict[str, Any]] = []
        collected_nodes: dict[str, dict[str, Any]] = {seed["id"]: seed}

        for _ in range(depth):
            if not frontier:
                break
            placeholders = ",".join("?" for _ in frontier)
            rows = self._conn.execute(
                f"""SELECT e.id, e.src_entity_id, e.relation, e.dst_entity_id,
                           e.source_chunk_id, e.valid_from, e.valid_to, e.confidence,
                           e.status, e.metadata, e.created_at,
                           s.name AS src_name, d.name AS dst_name
                    FROM edges e
                    JOIN entities s ON s.id = e.src_entity_id
                    JOIN entities d ON d.id = e.dst_entity_id
                    WHERE e.status='active'
                    AND (e.src_entity_id IN ({placeholders}) OR e.dst_entity_id IN ({placeholders}))
                    ORDER BY e.created_at DESC
                    LIMIT ?""",
                (*frontier, *frontier, limit),
            ).fetchall()
            frontier = set()
            for r in rows:
                edge = {
                    "id": str(r["id"]),
                    "src_entity_id": str(r["src_entity_id"]),
                    "src_name": str(r["src_name"]),
                    "relation": str(r["relation"]),
                    "dst_entity_id": str(r["dst_entity_id"]),
                    "dst_name": str(r["dst_name"]),
                    "source_chunk_id": str(r["source_chunk_id"]),
                    "valid_from": str(r["valid_from"]),
                    "valid_to": r["valid_to"],
                    "confidence": float(r["confidence"]),
                    "status": str(r["status"]),
                    "metadata": json_loads(r["metadata"]),
                    "created_at": str(r["created_at"]),
                }
                collected_edges.append(edge)
                for node_id, node_name in (
                    (edge["src_entity_id"], edge["src_name"]),
                    (edge["dst_entity_id"], edge["dst_name"]),
                ):
                    if node_id not in collected_nodes:
                        collected_nodes[node_id] = {
                            "id": node_id,
                            "name": node_name,
                        }
                    if node_id not in visited:
                        frontier.add(node_id)
                        visited.add(node_id)

        return {
            "entity": seed["name"],
            "depth": depth,
            "nodes": list(collected_nodes.values()),
            "edges": collected_edges,
        }

    def search_graph_context(self, query: str, limit: int = 20) -> list[SearchResult]:
        entities = self.search_entities(query, limit=max(1, limit))
        results: list[SearchResult] = []
        for ent in entities:
            graph = self.query_graph(ent["name"], depth=1, limit=max(5, limit))
            node_count = len(graph.get("nodes", []))
            edge_count = len(graph.get("edges", []))
            score = 1.0 + min(edge_count / 10.0, 1.0)
            content = f"Entity '{ent['name']}' linked to {node_count} nodes via {edge_count} edges"
            results.append(
                SearchResult(
                    id=str(ent["id"]),
                    content=content,
                    score=score,
                    source="graph",
                    metadata={
                        "entity_name": ent["name"],
                        "node_count": node_count,
                        "edge_count": edge_count,
                        "_source_kind": "graph",
                        "_created_at": ent["updated_at"],
                    },
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def count_entities(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()
        return int(row["n"]) if row else 0

    def count_edges(self, active_only: bool = True) -> int:
        if active_only:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM edges WHERE status='active'"
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM edges").fetchone()
        return int(row["n"]) if row else 0

    def list_graph_contradictions(
        self,
        lookback_hours: int = 24 * 30,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        since = iso_str(utcnow() - timedelta(hours=max(1, int(lookback_hours))))
        rows = self._conn.execute(
            """SELECT e.src_entity_id, s.name AS src_name, e.relation,
                      COUNT(DISTINCT e.dst_entity_id) AS dst_count,
                      MAX(e.created_at) AS last_seen
               FROM edges e
               JOIN entities s ON s.id = e.src_entity_id
               WHERE e.created_at >= ?
               GROUP BY e.src_entity_id, e.relation
               HAVING COUNT(DISTINCT e.dst_entity_id) > 1
               ORDER BY last_seen DESC
               LIMIT ?""",
            (since, limit),
        ).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            src_id = str(row["src_entity_id"])
            relation = str(row["relation"])
            evid_rows = self._conn.execute(
                """SELECT e.dst_entity_id, d.name AS dst_name, e.status, e.confidence,
                          e.valid_from, e.valid_to, e.source_chunk_id, e.created_at
                   FROM edges e
                   JOIN entities d ON d.id = e.dst_entity_id
                   WHERE e.src_entity_id=? AND e.relation=?
                   ORDER BY e.created_at DESC
                   LIMIT 8""",
                (src_id, relation),
            ).fetchall()
            evidence = [
                {
                    "dst_entity_id": str(e["dst_entity_id"]),
                    "dst_name": str(e["dst_name"]),
                    "status": str(e["status"]),
                    "confidence": float(e["confidence"]),
                    "valid_from": str(e["valid_from"]),
                    "valid_to": e["valid_to"],
                    "source_chunk_id": str(e["source_chunk_id"]),
                    "created_at": str(e["created_at"]),
                }
                for e in evid_rows
            ]
            out.append(
                {
                    "src_entity_id": src_id,
                    "src_name": str(row["src_name"]),
                    "relation": relation,
                    "dst_count": int(row["dst_count"]),
                    "last_seen": str(row["last_seen"]),
                    "evidence": evidence,
                }
            )
        return out

    # --- Audit Log ---

    def insert_audit_event(self, event: AuditEvent) -> str:
        self._execute_write(
            """INSERT INTO audit_log(id, action, target_type, target_id, detail, outcome, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event.id, event.action, event.target_type, event.target_id,
                event.detail, event.outcome, iso_str(event.created_at),
            ),
        )
        return event.id

    def list_audit_events(self, limit: int = 100, target_type: str | None = None) -> list[AuditEvent]:
        if target_type:
            rows = self._conn.execute(
                "SELECT * FROM audit_log WHERE target_type=? ORDER BY created_at DESC LIMIT ?",
                (target_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            AuditEvent(
                id=r["id"], action=r["action"], target_type=r["target_type"],
                target_id=r["target_id"], detail=r["detail"], outcome=r["outcome"],
                created_at=parse_iso(r["created_at"]),
            )
            for r in rows
        ]

    # --- Memory Access Tracking ---

    def increment_memory_access(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        ts = iso_str(utcnow())
        with self._write_lock:
            for memory_id in memory_ids:
                self._conn.execute(
                    """INSERT INTO memory_access(memory_id, access_count, last_accessed)
                       VALUES (?, 1, ?)
                       ON CONFLICT(memory_id) DO UPDATE SET
                         access_count = access_count + 1,
                         last_accessed = excluded.last_accessed""",
                    (memory_id, ts),
                )
            self._commit_locked()

    def get_memory_access_count(self, memory_id: str) -> int:
        row = self._conn.execute(
            "SELECT access_count FROM memory_access WHERE memory_id=?",
            (memory_id,),
        ).fetchone()
        return int(row["access_count"]) if row else 0

    def get_memory_access_counts(self, memory_ids: list[str]) -> dict[str, int]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._conn.execute(
            f"SELECT memory_id, access_count FROM memory_access WHERE memory_id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()
        return {str(r["memory_id"]): int(r["access_count"]) for r in rows}

    # --- Embedding Cache ---

    def get_cached_embedding(self, text_hash: str) -> bytes | None:
        row = self._conn.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash=?", (text_hash,)
        ).fetchone()
        return row["embedding"] if row else None

    def cache_embedding(self, text_hash: str, embedding: bytes, model: str) -> None:
        self._execute_write(
            "INSERT OR REPLACE INTO embedding_cache(text_hash, embedding, model, created_at) VALUES (?, ?, ?, ?)",
            (text_hash, embedding, model, iso_str(utcnow())),
        )

    # --- Files ---

    def insert_file(self, file_id: str, path: str, content_hash: str,
                    size_bytes: int, mime_type: str, metadata: dict[str, Any] | None = None) -> str:
        self._execute_write(
            """INSERT INTO files(id, path, content_hash, size_bytes, mime_type, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (file_id, path, content_hash, size_bytes, mime_type,
             json_dumps(metadata or {}), iso_str(utcnow())),
        )
        return file_id

    def get_file_by_path(self, path: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()
        return dict(row) if row else None

    # --- Row Converters ---

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            source_id=row["source_id"],
            content=row["content"],
            metadata=json_loads(row["metadata"]),
            created_at=parse_iso(row["created_at"]),
        )

    @staticmethod
    def _row_to_reasoning_entry(row: sqlite3.Row) -> ReasoningEntry:
        return ReasoningEntry(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            tags=json_loads(row["tags"]),
            evidence_ids=json_loads(row["evidence_ids"]),
            status=row["status"],
            superseded_by=row["superseded_by"],
            session_id=row["session_id"],
            metadata=json_loads(row["metadata"]),
            created_at=parse_iso(row["created_at"]),
        )

    @staticmethod
    def _row_to_cos(row: sqlite3.Row) -> CarryOverSummary:
        return CarryOverSummary(
            id=row["id"],
            session_id=row["session_id"],
            sequence=row["sequence"],
            summary=row["summary"],
            key_facts=json_loads(row["key_facts"]),
            open_questions=json_loads(row["open_questions"]),
            metadata=json_loads(row["metadata"]),
            created_at=parse_iso(row["created_at"]),
        )

    @staticmethod
    def _row_to_skill_capsule(row: sqlite3.Row) -> SkillCapsule:
        return SkillCapsule(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            procedure=row["procedure"],
            tags=json_loads(row["tags"]),
            version=row["version"],
            metadata=json_loads(row["metadata"]),
            created_at=parse_iso(row["created_at"]),
        )
