#!/usr/bin/env python3
"""USC efficiency showcase benchmark (compression ratio + throughput)."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark USC compression efficiency.")
    p.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input text/session file path (repeatable).",
    )
    p.add_argument(
        "--input-glob",
        default="",
        help="Optional glob pattern for input files (for example: 'sessions/*.jsonl').",
    )
    p.add_argument(
        "--synthetic-count",
        type=int,
        default=10,
        help="Synthetic sessions generated when no input files are provided.",
    )
    p.add_argument(
        "--synthetic-size-kb",
        type=int,
        default=32,
        help="Approximate size of each synthetic session.",
    )
    p.add_argument("--data-dir", default="", help="Optional data dir for spine/cogstore.")
    p.add_argument(
        "--verify-roundtrip",
        action="store_true",
        help="Verify decompression integrity for each compressed session.",
    )
    p.add_argument("--out", default="", help="Optional output JSON path.")
    return p.parse_args()


def _synthetic_payload(session_idx: int, target_bytes: int) -> tuple[str, bytes]:
    sid = f"synthetic-{session_idx:04d}"
    lines = [
        "[USER] remind me of all project deadlines this week",
        "[ASSISTANT] checking previous sprint notes and planning board",
        "[TOOL_CALL] search(query='project deadlines current sprint')",
        "[TOOL_RESULT] sprint-42 deadlines: api freeze Friday, QA signoff Saturday",
        "[ASSISTANT] noted; api freeze Friday and QA signoff Saturday",
    ]
    repeated = []
    while len("\n".join(repeated).encode("utf-8")) < target_bytes:
        repeated.extend(lines)
    payload = ("\n".join(repeated)[:target_bytes]).encode("utf-8", errors="ignore")
    return sid, payload


def _load_payloads(args: argparse.Namespace) -> list[tuple[str, bytes]]:
    paths: list[Path] = []
    for raw in args.input:
        p = Path(raw)
        if p.exists() and p.is_file():
            paths.append(p)
    if args.input_glob:
        for p in sorted(Path(".").glob(args.input_glob)):
            if p.exists() and p.is_file():
                paths.append(p)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_paths: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(p)
    if unique_paths:
        out: list[tuple[str, bytes]] = []
        for p in unique_paths:
            out.append((p.stem, p.read_bytes()))
        return out

    target_bytes = max(1024, int(args.synthetic_size_kb) * 1024)
    return [_synthetic_payload(i, target_bytes) for i in range(max(1, int(args.synthetic_count)))]


def _print_summary(summary: dict[str, Any]) -> None:
    print("USC Efficiency Summary")
    print(
        "sessions={sessions} raw_bytes={raw} compressed_bytes={compressed} "
        "ratio={ratio:.2f}x encode_s={encode:.3f} decode_s={decode:.3f}".format(
            sessions=summary["sessions"],
            raw=summary["total_raw_bytes"],
            compressed=summary["total_compressed_bytes"],
            ratio=summary["overall_ratio"],
            encode=summary["total_encode_seconds"],
            decode=summary["total_decode_seconds"],
        )
    )
    print(
        "throughput_mb_s={thr:.2f} roundtrip_verified={verified}".format(
            thr=summary["encode_throughput_mb_s"],
            verified=summary["roundtrip_verified"],
        )
    )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = Config()
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    payloads = _load_payloads(args)

    rows: list[dict[str, Any]] = []
    total_raw = 0
    total_compressed = 0
    total_encode_s = 0.0
    total_decode_s = 0.0
    verified = 0
    try:
        for sid, data in payloads:
            total_raw += len(data)
            t0 = time.perf_counter()
            packed = spine.compress_session(data, session_id=sid)
            encode_s = max(0.0, time.perf_counter() - t0)
            total_encode_s += encode_s

            blob = packed.get("blob", b"")
            stats = packed.get("stats", {}) if isinstance(packed.get("stats", {}), dict) else {}
            compressed_size = int(packed.get("compressed_size", len(blob)))
            total_compressed += compressed_size
            ratio = float(len(data) / max(1, compressed_size))

            decode_s = 0.0
            if args.verify_roundtrip:
                t1 = time.perf_counter()
                restored = spine.decompress_with_dedup(
                    blob,
                    expected_hash=str(stats.get("integrity_hash", "")),
                )
                decode_s = max(0.0, time.perf_counter() - t1)
                total_decode_s += decode_s
                if restored != data:
                    raise RuntimeError(f"Roundtrip verification failed for session: {sid}")
                verified += 1

            rows.append(
                {
                    "session_id": sid,
                    "raw_bytes": len(data),
                    "compressed_bytes": compressed_size,
                    "ratio": ratio,
                    "encode_seconds": encode_s,
                    "decode_seconds": decode_s,
                    "stats": {
                        "ref": int(stats.get("ref", 0)),
                        "delta": int(stats.get("delta", 0)),
                        "full": int(stats.get("full", 0)),
                        "temporal_motifs": int(stats.get("temporal_motifs", 0)),
                    },
                }
            )

        overall_ratio = float(total_raw / max(1, total_compressed))
        encode_mb_s = (total_raw / (1024 * 1024)) / max(1e-9, total_encode_s)
        cogstore_stats = spine.cogstore.stats()
        summary = {
            "sessions": len(payloads),
            "total_raw_bytes": total_raw,
            "total_compressed_bytes": total_compressed,
            "overall_ratio": overall_ratio,
            "total_encode_seconds": total_encode_s,
            "total_decode_seconds": total_decode_s,
            "encode_throughput_mb_s": encode_mb_s,
            "roundtrip_verified": verified,
            "cogstore": cogstore_stats,
            "results": rows,
        }
    finally:
        await spine.close()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = _parse_args()
    result = asyncio.run(_run(args))
    _print_summary(result)
    if args.out:
        print(f"saved={args.out}")


if __name__ == "__main__":
    main()
