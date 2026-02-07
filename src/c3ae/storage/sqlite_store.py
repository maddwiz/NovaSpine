"""SQLite storage with FTS5 full-text search."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

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
        self._conn.commit()
        return session_id

    def end_session(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET ended_at=? WHERE id=?",
            (iso_str(utcnow()), session_id),
        )
        self._conn.commit()

    # --- Chunks ---

    def insert_chunk(self, chunk: Chunk) -> str:
        self._conn.execute(
            "INSERT INTO chunks(id, source_id, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (chunk.id, chunk.source_id, chunk.content, json_dumps(chunk.metadata), iso_str(chunk.created_at)),
        )
        self._conn.commit()
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
        self._conn.commit()
        return cur.rowcount > 0

    def search_chunks_fts(self, query: str, limit: int = 20) -> list[SearchResult]:
        fts_query = _sanitize_fts_query(query)
        rows = self._conn.execute(
            """SELECT c.id, c.content, c.metadata,
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
                metadata=json_loads(r["metadata"]),
            )
            for r in rows
        ]

    def count_chunks(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0]

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
        self._conn.commit()
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
            """SELECT r.id, r.content, r.title, r.metadata,
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
                metadata=json_loads(r["metadata"]),
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
        self._conn.commit()
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
        self._conn.commit()
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
        self._conn.commit()
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
        self._conn.commit()
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
        self._conn.commit()

    # --- Files ---

    def insert_file(self, file_id: str, path: str, content_hash: str,
                    size_bytes: int, mime_type: str, metadata: dict[str, Any] | None = None) -> str:
        self._conn.execute(
            """INSERT INTO files(id, path, content_hash, size_bytes, mime_type, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (file_id, path, content_hash, size_bytes, mime_type,
             json_dumps(metadata or {}), iso_str(utcnow())),
        )
        self._conn.commit()
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
