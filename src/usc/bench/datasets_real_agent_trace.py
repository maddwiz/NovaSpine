from __future__ import annotations

import json
import random
import time
import uuid
from datetime import datetime, timezone


TOOLS = [
    "web.search_query",
    "file_search.msearch",
    "file_search.mclick",
    "python.exec",
    "container.exec",
    "gcal.search_events",
    "gmail.search_email_ids",
]

HOSTS = ["spark-4c54", "mbp-des", "node-a", "node-b"]
PATHS = ["/home/maddwiz/usc", "/mnt/data/run", "/tmp/cache", "/var/log/syslog"]
MODELS = ["gpt-4o", "llama-3-70b", "qwen2.5-32b"]


def _ts():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _rand_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _rand_hash(n=8):
    return "".join(random.choice("0123456789abcdef") for _ in range(n))


def _maybe_stacktrace():
    if random.random() < 0.08:
        return (
            "Traceback (most recent call last):\n"
            "  File \"runner.py\", line 201, in run\n"
            "    main()\n"
            "  File \"core.py\", line 88, in main\n"
            "    raise RuntimeError(\"OOM in KV cache\")\n"
            "RuntimeError: OOM in KV cache\n"
        )
    return ""


def _tool_payload(tool: str, step: int):
    rid = str(uuid.uuid4())
    if tool == "web.search_query":
        q = random.choice([
            "latest transformer kv cache compression paper",
            "drain3 log template mining",
            "zstd trained dictionary json logs",
            "arithmetic coding neural compressor",
        ])
        return {"request_id": rid, "q": q, "recency": random.choice([30, 60, 180])}

    if tool == "file_search.msearch":
        q = random.choice(["chunking best size", "template mining", "delta encoding", "mtf"])
        return {"request_id": rid, "queries": [q], "intent": "nav"}

    if tool == "file_search.mclick":
        return {"request_id": rid, "pointers": [f"{random.randint(1,9)}:{random.randint(0,9)}"]}

    if tool == "python.exec":
        code = random.choice([
            "import torch\nprint(torch.__version__)",
            "import numpy as np\nprint(np.mean([1,2,3]))",
            "x = 1\nfor _ in range(1000): x = (x*1103515245 + 12345) & 0xFFFFFFFF\nprint(x)",
        ])
        return {"request_id": rid, "code": code}

    if tool == "container.exec":
        cmd = random.choice([
            ["python", "-m", "py_compile", "src/usc/mem/stream_proto.py"],
            ["pytest", "-q"],
            ["git", "status"],
            ["python", "-m", "usc.bench.stream_bench10_v3b_windows"],
        ])
        return {"request_id": rid, "cmd": cmd}

    if tool == "gcal.search_events":
        return {"request_id": rid, "time_min": "2026-01-01T00:00:00", "time_max": "2026-02-01T00:00:00"}

    if tool == "gmail.search_email_ids":
        return {"request_id": rid, "query": "subject:(invoice OR receipt) newer_than:30d", "max_results": 10}

    return {"request_id": rid, "step": step}


def real_agent_trace(loops: int = 200, seed: int = 7) -> str:
    random.seed(seed)

    lines = []
    session = _rand_hash(12)
    model = random.choice(MODELS)

    for i in range(loops):
        tool = random.choice(TOOLS)
        host = random.choice(HOSTS)
        ip = _rand_ip()
        pid = random.randint(1200, 9500)

        payload = _tool_payload(tool, i)
        payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

        # multiple line styles to simulate real messy logs
        style = random.random()

        if style < 0.35:
            lines.append(f"{_ts()} INFO host={host} ip={ip} pid={pid} session={session} tool={tool} args={payload_json}")
        elif style < 0.70:
            lines.append(f"[{_ts()}] tool_call::{tool} rid={payload.get('request_id')} payload={payload_json}")
        else:
            lines.append(f"{_ts()} | {model} | {tool} | {payload_json}")

        # occasional second line per step (results / timing)
        if random.random() < 0.55:
            ms = random.randint(5, 1800)
            tok = random.randint(20, 2500)
            lines.append(f"{_ts()} DEBUG tool_result ok=true latency_ms={ms} tokens={tok} bytes_out={random.randint(64, 16384)}")

        st = _maybe_stacktrace()
        if st:
            lines.append(st.rstrip("\n"))

        # occasional path + hash chatter
        if random.random() < 0.22:
            p = random.choice(PATHS)
            h = _rand_hash(16)
            lines.append(f"{_ts()} WARN cache_miss path={p} sha={h} retry={random.randint(0,3)}")

    return "\n".join(lines) + "\n"
