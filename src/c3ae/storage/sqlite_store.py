"""SQLite storage with FTS5 full-text search."""

from __future__ import annotations

from datetime import timedelta
import re
import sqlite3
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
from c3ae.utils import iso_str, json_dumps, json_loads, parse_iso, utcnow

SCHEMA_VERSION = 1

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
    title, content, tags, content=reasoning_bank, content_rowid=rowid
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
    name, description, procedure, tags, content=skill_capsules, content_rowid=rowid
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, content=chunks, content_rowid=rowid
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
    summary, facts, content=consolidated_memories, content_rowid=rowid
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
"""


def _sanitize_fts_query(query: str) -> str:
    """Sanitize a query for FTS5 MATCH syntax.

    Strips special FTS5 operators and wraps each token in double quotes
    to prevent syntax errors from punctuation like '?'.
    """
    # Remove FTS5 special chars
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    tokens = cleaned.split()
    if not tokens:
        return '""'
    # Quote each token to prevent operator interpretation
    return " ".join(f'"{t}"' for t in tokens)


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
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._init_schema()

    def _commit(self, retries: int = 3) -> None:
        """Commit with retry on database-locked errors."""
        import time as _time
        for attempt in range(retries):
            try:
                self._conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < retries - 1:
                    _time.sleep(1.0 * (attempt + 1))
                    continue
                raise

    def _init_schema(self) -> None:
        import time as _time
        for attempt in range(5):
            try:
                cur = self._conn.cursor()
                cur.executescript(_SCHEMA)
                cur.executescript(_FTS_TRIGGERS)
                cur.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                    ("schema_version", str(SCHEMA_VERSION)),
                )
                self._conn.commit()
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
        self._conn.execute(
            "INSERT INTO sessions(id, started_at, metadata) VALUES (?, ?, ?)",
            (session_id, iso_str(utcnow()), json_dumps(metadata or {})),
        )
        self._commit()
        return session_id

    def end_session(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET ended_at=? WHERE id=?",
            (iso_str(utcnow()), session_id),
        )
        self._commit()

    # --- Chunks ---

    def insert_chunk(self, chunk: Chunk) -> str:
        self._conn.execute(
            "INSERT INTO chunks(id, source_id, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (chunk.id, chunk.source_id, chunk.content, json_dumps(chunk.metadata), iso_str(chunk.created_at)),
        )
        self._commit()
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
        cur = self._conn.execute("DELETE FROM chunks WHERE id=?", (chunk_id,))
        self._commit()
        return cur.rowcount > 0

    def search_chunks_fts(self, query: str, limit: int = 20) -> list[SearchResult]:
        fts_query = _sanitize_fts_query(query)
        rows = self._conn.execute(
            """SELECT c.id, c.content, c.metadata, c.created_at,
                      bm25(chunks_fts) AS score
               FROM chunks_fts f
               JOIN chunks c ON c.rowid = f.rowid
               WHERE chunks_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (fts_query, limit),
        ).fetchall()
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

    # --- Reasoning Bank ---

    def insert_reasoning_entry(self, entry: ReasoningEntry) -> str:
        self._conn.execute(
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
        self._commit()
        return entry.id

    def get_reasoning_entry(self, entry_id: str) -> ReasoningEntry | None:
        row = self._conn.execute("SELECT * FROM reasoning_bank WHERE id=?", (entry_id,)).fetchone()
        if not row:
            return None
        return self._row_to_reasoning_entry(row)

    def supersede_reasoning_entry(self, old_id: str, new_entry: ReasoningEntry) -> str:
        self._conn.execute(
            "UPDATE reasoning_bank SET status='superseded', superseded_by=? WHERE id=?",
            (new_entry.id, old_id),
        )
        return self.insert_reasoning_entry(new_entry)

    def list_reasoning_entries(self, status: str = "active", limit: int = 100) -> list[ReasoningEntry]:
        rows = self._conn.execute(
            "SELECT * FROM reasoning_bank WHERE status=? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
        return [self._row_to_reasoning_entry(r) for r in rows]

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
        self._conn.execute(
            """INSERT INTO evidence_packs(id, claim, sources, confidence, reasoning, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                pack.id, pack.claim, json_dumps(pack.sources),
                pack.confidence, pack.reasoning,
                json_dumps(pack.metadata), iso_str(pack.created_at),
            ),
        )
        self._commit()
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
        self._conn.execute(
            """INSERT INTO cos(id, session_id, sequence, summary, key_facts, open_questions, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cos.id, cos.session_id, cos.sequence, cos.summary,
                json_dumps(cos.key_facts), json_dumps(cos.open_questions),
                json_dumps(cos.metadata), iso_str(cos.created_at),
            ),
        )
        self._commit()
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
        self._conn.execute(
            """INSERT INTO skill_capsules(id, name, description, procedure, tags, version, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                capsule.id, capsule.name, capsule.description, capsule.procedure,
                json_dumps(capsule.tags), capsule.version,
                json_dumps(capsule.metadata), iso_str(capsule.created_at),
            ),
        )
        self._commit()
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
            self._commit()
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
        self._commit()
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
            self._commit()
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
        self._commit()
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
        self._conn.execute(
            """INSERT INTO entity_mentions(id, entity_id, chunk_id, confidence, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (mention_id, entity_id, chunk_id, float(confidence), iso_str(utcnow())),
        )
        self._commit()
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
        self._commit()
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
        self._conn.execute(
            """INSERT INTO audit_log(id, action, target_type, target_id, detail, outcome, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event.id, event.action, event.target_type, event.target_id,
                event.detail, event.outcome, iso_str(event.created_at),
            ),
        )
        self._commit()
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
        for memory_id in memory_ids:
            self._conn.execute(
                """INSERT INTO memory_access(memory_id, access_count, last_accessed)
                   VALUES (?, 1, ?)
                   ON CONFLICT(memory_id) DO UPDATE SET
                     access_count = access_count + 1,
                     last_accessed = excluded.last_accessed""",
                (memory_id, ts),
            )
        self._commit()

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
        self._conn.execute(
            "INSERT OR REPLACE INTO embedding_cache(text_hash, embedding, model, created_at) VALUES (?, ?, ?, ?)",
            (text_hash, embedding, model, iso_str(utcnow())),
        )
        self._commit()

    # --- Files ---

    def insert_file(self, file_id: str, path: str, content_hash: str,
                    size_bytes: int, mime_type: str, metadata: dict[str, Any] | None = None) -> str:
        self._conn.execute(
            """INSERT INTO files(id, path, content_hash, size_bytes, mime_type, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (file_id, path, content_hash, size_bytes, mime_type,
             json_dumps(metadata or {}), iso_str(utcnow())),
        )
        self._commit()
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
