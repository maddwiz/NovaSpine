#!/usr/bin/env python3
"""Nova Session Compression Watcher — watches for new sessions and compresses them.

Monitors the OpenClaw and Claude Code session directories for new/modified .jsonl files
and compresses them through the cognitive dedup pipeline.

Also ingests sessions into searchable memory (FTS5 + FAISS vectors).

Runs as a systemd user service for persistent operation.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import json
import hashlib
from pathlib import Path

# Ensure imports
_src = str(Path(__file__).resolve().parents[1] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine

# Directories to watch
WATCH_DIRS = [
    Path.home() / ".openclaw" / "agents",
    Path.home() / ".claude" / "projects",
]

# Where to store compressed blobs
OUTPUT_DIR = Path.home() / "Nova-v1" / "data" / "compressed"

# State files to track what's been compressed/ingested
STATE_FILE = Path.home() / "Nova-v1" / "data" / "compress-state.json"
INGEST_STATE_FILE = Path.home() / "Nova-v1" / "data" / "ingest-state.json"

# Minimum file size to bother compressing
MIN_SIZE = 2000

# How often to scan (seconds)
SCAN_INTERVAL = 300  # 5 minutes


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def file_hash(path: Path) -> str:
    """Quick hash of file size + mtime for change detection."""
    st = path.stat()
    return hashlib.md5(f"{st.st_size}:{st.st_mtime}".encode()).hexdigest()


def find_sessions() -> list[Path]:
    """Find all .jsonl session files."""
    sessions = []
    for watch_dir in WATCH_DIRS:
        if watch_dir.exists():
            for f in watch_dir.rglob("*.jsonl"):
                if f.stat().st_size >= MIN_SIZE:
                    sessions.append(f)
    return sessions


def load_ingest_state() -> dict:
    if INGEST_STATE_FILE.exists():
        return json.loads(INGEST_STATE_FILE.read_text())
    return {}


def save_ingest_state(state: dict) -> None:
    INGEST_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    INGEST_STATE_FILE.write_text(json.dumps(state, indent=2))


def compress_session(spine: MemorySpine, session_file: Path) -> dict | None:
    """Compress a single session file. Returns result dict or None on error."""
    try:
        data = session_file.read_bytes()
        session_id = session_file.stem
        result = spine.compress_session(data, session_id=session_id)

        # Save compressed blob
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{session_id}.ucog"
        out_path.write_bytes(result["blob"])

        ratio = result["original_size"] / max(1, result["compressed_size"])
        return {
            "file": str(session_file),
            "original_size": result["original_size"],
            "compressed_size": result["compressed_size"],
            "ratio": round(ratio, 1),
            "output": str(out_path),
        }
    except Exception as e:
        print(f"  ERROR compressing {session_file.name}: {e}", file=sys.stderr)
        return None


def ingest_session(spine: MemorySpine, session_file: Path) -> dict | None:
    """Ingest a session into searchable memory with optional embedding.

    Ingests into FTS5 for keyword search, then tries to embed into FAISS
    for vector search if Venice API key is available.
    """
    try:
        result = spine.ingest_session(session_file)
        # Try to embed the newly ingested chunks into FAISS
        if result and result["chunks_ingested"] > 0 and os.environ.get("VENICE_API_KEY"):
            try:
                _embed_recent_chunks(spine, session_file.stem, result["chunks_ingested"])
            except Exception as e:
                print(f"  WARNING: Embedding failed for {session_file.name}: {e}",
                      file=sys.stderr)
        return result
    except Exception as e:
        print(f"  ERROR ingesting {session_file.name}: {e}", file=sys.stderr)
        return None


def _embed_recent_chunks(spine: MemorySpine, session_id: str, expected_count: int):
    """Embed recently ingested chunks for a session into FAISS."""
    rows = spine.sqlite._conn.execute(
        """SELECT id, content FROM chunks
           WHERE source_id = ?
           ORDER BY rowid DESC LIMIT ?""",
        (f"session:{session_id}", expected_count),
    ).fetchall()

    if not rows:
        return

    # Filter out chunks already in FAISS
    existing = set(spine.faiss._id_map)
    to_embed = [(r["id"], r["content"]) for r in rows if r["id"] not in existing]
    if not to_embed:
        return

    chunk_ids = [cid for cid, _ in to_embed]
    texts = [content for _, content in to_embed]

    asyncio.run(spine._embed_and_index(chunk_ids, texts))
    if spine.config.faiss_dir:
        spine.faiss.save()
    print(f"  Embedded {len(to_embed)} chunks for {session_id}")


def run_once(spine: MemorySpine) -> tuple[int, int]:
    """Scan for new/changed sessions, compress and ingest them.

    Returns (count_compressed, count_ingested).
    """
    compress_state = load_state()
    ingest_state = load_ingest_state()
    sessions = find_sessions()
    compressed = 0
    ingested = 0

    for session_file in sessions:
        fpath = str(session_file)
        fhash = file_hash(session_file)

        # Compress if needed
        if fpath not in compress_state or compress_state[fpath] != fhash:
            result = compress_session(spine, session_file)
            if result:
                compress_state[fpath] = fhash
                compressed += 1
                print(f"  Compressed {session_file.name}: "
                      f"{result['original_size']:,} -> {result['compressed_size']:,} "
                      f"({result['ratio']}x)")

        # Ingest into searchable memory if needed
        if fpath not in ingest_state or ingest_state[fpath] != fhash:
            result = ingest_session(spine, session_file)
            if result and result["chunks_ingested"] > 0:
                ingest_state[fpath] = fhash
                ingested += 1
                print(f"  Ingested {session_file.name}: "
                      f"{result['chunks_ingested']} chunks "
                      f"({result['roles']})")

    if compressed:
        save_state(compress_state)
    if ingested:
        save_ingest_state(ingest_state)

    return compressed, ingested


def main():
    daemon = "--daemon" in sys.argv

    # Init spine
    data_dir = Path.home() / "Nova-v1" / "data"
    config = Config()
    config.data_dir = data_dir
    config.ensure_dirs()
    spine = MemorySpine(config)

    print(f"Nova compression watcher started")
    print(f"  Watching: {[str(d) for d in WATCH_DIRS]}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Mode: {'daemon' if daemon else 'one-shot'}")

    if daemon:
        while True:
            try:
                n_compressed, n_ingested = run_once(spine)
                if n_compressed or n_ingested:
                    cs = spine.cogstore.stats()
                    chunks_in_db = spine.sqlite.count_chunks()
                    print(f"  [{time.strftime('%H:%M')}] "
                          f"Compressed {n_compressed}, ingested {n_ingested}, "
                          f"cogstore: {cs.get('unique_chunks', 0)}, "
                          f"searchable: {chunks_in_db} chunks")
            except Exception as e:
                print(f"  Error in scan: {e}", file=sys.stderr)
            time.sleep(SCAN_INTERVAL)
    else:
        n_compressed, n_ingested = run_once(spine)
        cs = spine.cogstore.stats()
        chunks_in_db = spine.sqlite.count_chunks()
        print(f"\nDone: {n_compressed} compressed, {n_ingested} ingested, "
              f"cogstore: {cs.get('unique_chunks', 0)}, "
              f"searchable: {chunks_in_db} chunks")

    spine.sqlite.close()


if __name__ == "__main__":
    main()
