#!/usr/bin/env python3
"""Backfill FAISS vector embeddings for existing session chunks.

Reads all chunks from SQLite that don't yet have FAISS vectors,
embeds them in batches via Venice API, and inserts into FAISS.
Handles rate limiting and saves progress incrementally.

Usage:
    python scripts/backfill-embeddings.py [--batch-size 32] [--delay 0.5]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine


def _normalize_text(text: str, max_chars: int) -> str:
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


async def _embed_batch(spine: MemorySpine, batch_ids: list[str], batch_texts: list[str]) -> int:
    before_size = spine.faiss.size
    await spine._embed_and_index(batch_ids, batch_texts)
    return max(0, spine.faiss.size - before_size)


async def backfill(batch_size: int = 32, delay: float = 0.5, limit: int = 0, max_chars: int = 0):
    config = Config()
    spine = MemorySpine(config)
    provider = (config.venice.embedding_provider or "venice").strip().lower()
    model = config.venice.embedding_model
    dims = config.venice.embedding_dims

    # Get all chunk IDs from SQLite
    rows = spine.sqlite._conn.execute(
        "SELECT id, content FROM chunks ORDER BY rowid"
    ).fetchall()
    all_chunk_ids = [(r["id"], _normalize_text(r["content"], max_chars)) for r in rows]
    total_chunks = len(all_chunk_ids)

    # Get IDs already in FAISS
    existing_ids = set(spine.faiss._id_map)
    print(f"Total chunks in SQLite: {total_chunks}")
    print(f"Already in FAISS: {len(existing_ids)}")

    # Filter to only un-embedded chunks
    to_embed = [(cid, content) for cid, content in all_chunk_ids if cid not in existing_ids]
    print(f"Chunks needing embeddings: {len(to_embed)}")

    if not to_embed:
        print("Nothing to do!")
        return

    if limit > 0:
        to_embed = to_embed[:limit]
        print(f"Limiting to first {limit} chunks")

    print(f"\nStarting backfill with batch_size={batch_size}, delay={delay}s")
    print(f"Embedding provider: {provider}")
    print(f"Embedding model: {model}")
    print(f"Embedding dims: {dims}")
    print(f"Max chars per chunk: {max_chars or 'unbounded'}")
    print(f"FAISS dir: {config.faiss_dir}")
    print()

    embedded = 0
    errors = 0
    start_time = time.time()

    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i:i + batch_size]
        batch_ids = [cid for cid, _ in batch]
        batch_texts = [content for _, content in batch]

        try:
            inserted = await _embed_batch(spine, batch_ids, batch_texts)
            if inserted == 0 and len(batch_ids) > 1:
                for single_id, single_text in zip(batch_ids, batch_texts):
                    try:
                        inserted += await _embed_batch(spine, [single_id], [single_text])
                    except Exception as single_exc:
                        print(f"    skipped {single_id}: {single_exc}")
            if inserted == 0:
                raise RuntimeError(
                    f"No vectors inserted for batch {i // batch_size} using provider={provider}; "
                    "check embedding backend configuration."
                )
            embedded += inserted

            elapsed = time.time() - start_time
            rate = embedded / elapsed if elapsed > 0 else 0
            remaining = (len(to_embed) - embedded) / rate if rate > 0 else 0
            print(f"  [{embedded}/{len(to_embed)}] "
                  f"{rate:.1f} chunks/s, ~{remaining:.0f}s remaining")

        except Exception as e:
            errors += 1
            print(f"  ERROR on batch {i//batch_size}: {e}")
            if errors > 5:
                print("Too many errors, stopping.")
                break

        # Rate limit delay
        if delay > 0 and i + batch_size < len(to_embed):
            await asyncio.sleep(delay)

    # Save final state
    if spine.config.faiss_dir:
        spine.faiss.save()
        print(f"\nFAISS index saved to {spine.config.faiss_dir}")

    elapsed = time.time() - start_time
    print(f"\nDone! Embedded {embedded} chunks in {elapsed:.1f}s")
    print(f"FAISS now has {spine.faiss.size} vectors")
    if errors:
        print(f"Errors: {errors}")

    await spine.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill FAISS embeddings")
    parser.add_argument("--batch-size", type=int, default=32, help="Chunks per API call")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between batches")
    parser.add_argument("--limit", type=int, default=0, help="Max chunks to embed (0=all)")
    parser.add_argument("--max-chars", type=int, default=0, help="Truncate chunk text to this many chars before embedding (0=unbounded)")
    args = parser.parse_args()

    asyncio.run(backfill(args.batch_size, args.delay, args.limit, args.max_chars))


if __name__ == "__main__":
    main()
