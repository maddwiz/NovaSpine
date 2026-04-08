"""NovaSpine CLI."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import click

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine


def _get_spine(data_dir: str | None = None) -> MemorySpine:
    config = Config()
    if data_dir:
        config.data_dir = Path(data_dir)
    return MemorySpine(config)


def _default_data_dir(data_dir: str | None) -> Path:
    config = Config()
    if data_dir:
        config.data_dir = Path(data_dir)
    return config.data_dir


def _default_openclaw_install_root() -> Path:
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return Path(os.environ.get("NOVASPINE_OPENCLAW_HOME", base / "novaspine" / "openclaw"))


def _default_openclaw_config_path() -> Path:
    return Path(os.environ.get("OPENCLAW_CONFIG_PATH", Path.home() / ".openclaw" / "openclaw.json"))


def _find_plugin_entry(entries: object, plugin_id: str) -> dict | None:
    if isinstance(entries, dict):
        value = entries.get(plugin_id)
        return value if isinstance(value, dict) else None
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            candidate = item.get("name") or item.get("id") or item.get("plugin")
            if candidate == plugin_id:
                return item
    return None


def _probe_json(url: str, timeout: float = 2.0) -> tuple[bool, str]:
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return True, json.dumps(payload, sort_keys=True)
    except HTTPError as error:
        return False, f"HTTP {error.code}"
    except URLError as error:
        return False, str(error.reason)
    except Exception as error:  # pragma: no cover - defensive
        return False, str(error)


def _doctor_check(name: str, level: str, detail: str) -> dict[str, str]:
    return {"name": name, "level": level, "detail": detail}


def _preview_text(text: str, limit: int = 220) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3].rstrip()}..."


@click.group()
@click.option("--data-dir", envvar="C3AE_DATA_DIR", default=None, help="Data directory")
@click.pass_context
def main(ctx: click.Context, data_dir: str | None) -> None:
    """NovaSpine CLI for storing, recalling, and inspecting memory.

    Use `novaspine ingest` to store memory, `novaspine recall` to retrieve it,
    `novaspine search` for lower-level inspection, `novaspine status` to inspect
    state, `novaspine doctor` to verify installs, and `novaspine serve` to run
    the HTTP API.
    """
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show NovaSpine memory status."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    st = spine.status()
    click.echo("NovaSpine Status")
    click.echo(f"  Chunks:            {st['chunks']}")
    click.echo(f"  Vectors:           {st['vectors']}")
    click.echo(f"  Reasoning entries: {st['reasoning_entries']}")
    click.echo(f"  Skills:            {st['skills']}")
    click.echo(f"  Vault documents:   {st['vault_documents']}")
    spine.sqlite.close()


@main.command()
@click.argument("query")
@click.option("--top-k", "-k", default=10, help="Number of results")
@click.option("--keyword-only", "-K", is_flag=True, help="Keyword search only (no embeddings)")
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int, keyword_only: bool) -> None:
    """Inspect lower-level hybrid search results for debugging and manual review."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    if keyword_only:
        results = spine.search_keyword(query, top_k=top_k)
    else:
        results = asyncio.run(spine.search(query, top_k=top_k))
    if not results:
        click.echo("No search results found.")
    else:
        click.echo(f"NovaSpine Search: {query}")
        for i, r in enumerate(results, 1):
            click.echo(f"\n--- Result {i} (score: {r.score:.4f}, source: {r.source}) ---")
            click.echo(f"ID: {r.id}")
            preview = _preview_text(r.content)
            click.echo(f"{preview}")
    spine.sqlite.close()


@main.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of memories to return")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON")
@click.pass_context
def recall(ctx: click.Context, query: str, top_k: int, json_output: bool) -> None:
    """Recall useful memories for a query.

    This is the preferred high-level retrieval command for humans and agents.
    Use `search` when you want lower-level hybrid search inspection instead.
    """
    spine = _get_spine(ctx.obj.get("data_dir"))
    rows = asyncio.run(spine.recall(query, top_k=top_k))
    if json_output:
        click.echo(json.dumps(rows, indent=2))
    elif not rows:
        click.echo("No memories recalled.")
    else:
        click.echo(f"NovaSpine Recall: {query}")
        for i, row in enumerate(rows, 1):
            metadata = row.get("metadata") or {}
            score = float(row.get("score", 0.0))
            source = str(row.get("source", "") or "memory")
            click.echo(f"\n--- Memory {i} (score: {score:.4f}, source: {source}) ---")
            click.echo(f"ID: {row.get('id', '')}")
            source_id = str(metadata.get("source_id", "")).strip()
            if source_id:
                click.echo(f"Source ID: {source_id}")
            click.echo(_preview_text(str(row.get("content", ""))))
    spine.sqlite.close()


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--source-id", "-s", default="", help="Source identifier")
@click.pass_context
def ingest(ctx: click.Context, path: str, source_id: str) -> None:
    """Ingest a file into NovaSpine memory."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    file_path = Path(path)
    chunk_ids = asyncio.run(spine.ingest_file(file_path, source_id=source_id))
    click.echo(f"Ingested {file_path.name}: {len(chunk_ids)} chunks indexed")
    spine.sqlite.close()


@main.command()
@click.argument("task")
@click.option("--max-steps", "-n", default=1, help="Max reasoning steps")
@click.pass_context
def reason(ctx: click.Context, task: str, max_steps: int) -> None:
    """Start a reasoning session."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    from c3ae.pipeline.loop import PipelineLoop
    pipeline = PipelineLoop(spine, max_steps=max_steps)
    result = asyncio.run(pipeline.run(task))
    click.echo(f"Session: {result.session.session_id}")
    click.echo(f"Steps: {len(result.session.steps)}")
    click.echo(f"Context hits: {len(result.search_results)}")
    if result.entries_written:
        click.echo(f"Entries written: {len(result.entries_written)}")
    if result.entries_blocked:
        click.echo(f"Entries blocked: {len(result.entries_blocked)}")
    if result.session.final_answer:
        click.echo(f"\nResult: {result.session.final_answer}")
    spine.sqlite.close()


@main.command()
@click.option("--limit", "-l", default=20, help="Number of entries")
@click.pass_context
def bank(ctx: click.Context, limit: int) -> None:
    """List reasoning bank entries."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    entries = spine.bank.list_active(limit=limit)
    if not entries:
        click.echo("No active reasoning entries.")
    else:
        for e in entries:
            click.echo(f"  [{e.id[:8]}] {e.title} ({len(e.evidence_ids)} evidence)")
    spine.sqlite.close()


@main.command(name="cos")
@click.argument("session_id")
@click.pass_context
def cos_cmd(ctx: click.Context, session_id: str) -> None:
    """Show carry-over summary for a session."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    prompt = spine.cos.render_prompt(session_id)
    if prompt:
        click.echo(prompt)
    else:
        click.echo(f"No COS found for session {session_id}")
    spine.sqlite.close()


@main.command()
@click.option("--host", "-h", default="127.0.0.1", help="Bind host")
@click.option("--port", "-p", default=8420, type=int, help="Bind port")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Start the NovaSpine HTTP API server."""
    import uvicorn
    from c3ae.api.routes import create_app

    data_dir = ctx.obj.get("data_dir")
    app = create_app(data_dir=data_dir)
    click.echo(f"Starting NovaSpine API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command(name="cogdedup-stats")
@click.pass_context
def cogdedup_stats(ctx: click.Context) -> None:
    """Show cognitive deduplication statistics."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    st = spine.status()

    click.echo("Cognitive Deduplication Stats")
    click.echo("=" * 50)

    if "cogdedup" in st:
        cs = st["cogdedup"]
        click.echo(f"  Chunks stored:     {cs.get('total_chunks', 0)}")
        click.echo(f"  Total references:  {cs.get('total_references', 0)}")
        click.echo(f"  Hot tier:          {cs.get('hot_chunks', 0)}")
        click.echo(f"  Cold archived:     {cs.get('cold_chunks', 0)}")
    else:
        click.echo("  (cogstore not initialized — no sessions compressed yet)")

    if "anomaly" in st:
        a = st["anomaly"]
        click.echo(f"\nAnomaly Detection")
        click.echo(f"  Observations:      {a['total_observations']}")
        click.echo(f"  Alerts fired:      {a['alerts']}")
        click.echo(f"  Mean ratio:        {a['mean_ratio']:.2f}x")

    if "temporal" in st:
        click.echo(f"\nTemporal Patterns")
        click.echo(f"  Motifs detected:   {st['temporal']['motifs_detected']}")

    spine.sqlite.close()


@main.command(name="compress-session")
@click.argument("path", type=click.Path(exists=True))
@click.option("--session-id", "-s", default="", help="Session ID")
@click.option("--output", "-o", default="", help="Output path for compressed blob")
@click.pass_context
def compress_session_cmd(ctx: click.Context, path: str, session_id: str, output: str) -> None:
    """Compress a session file using cognitive deduplication."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    file_path = Path(path)
    data = file_path.read_bytes()

    if not session_id:
        session_id = file_path.stem

    result = spine.compress_session(data, session_id=session_id)
    ratio = result["original_size"] / max(1, result["compressed_size"])

    click.echo(f"Compressed: {file_path.name}")
    click.echo(f"  Original:    {result['original_size']:,} bytes")
    click.echo(f"  Compressed:  {result['compressed_size']:,} bytes")
    click.echo(f"  Ratio:       {ratio:.1f}x")

    stats = result["stats"]
    if "anomaly_alert" in stats:
        a = stats["anomaly_alert"]
        click.echo(f"  ANOMALY:     {a['type']} (z={a['z_score']:.2f})")
    if "temporal_motifs" in stats:
        click.echo(f"  Motifs:      {stats['temporal_motifs']}")

    if output:
        out_path = Path(output)
        out_path.write_bytes(result["blob"])
        click.echo(f"  Saved to:    {out_path}")

    spine.sqlite.close()


@main.command(name="anomaly-report")
@click.pass_context
def anomaly_report(ctx: click.Context) -> None:
    """Show anomaly detection drift report."""
    spine = _get_spine(ctx.obj.get("data_dir"))

    # Access the anomaly detector (triggers lazy init)
    detector = spine._anomaly_detector
    report = detector.drift_report()

    click.echo("Anomaly Detection Drift Report")
    click.echo("=" * 50)
    click.echo(f"  Observations:        {detector._observation_count}")
    click.echo(f"  Alerts fired:        {report.alerts_count}")
    click.echo(f"  Mean ratio:          {report.current_mean:.2f}x")
    click.echo(f"  Std deviation:       {report.current_std:.4f}")
    click.echo(f"  Drifting:            {'YES' if report.is_drifting else 'no'}")

    if detector._alerts:
        click.echo(f"\nRecent Alerts ({len(detector._alerts)}):")
        for alert in detector._alerts[-10:]:
            click.echo(f"  [{alert.severity}] z={alert.z_score:.2f} "
                       f"ratio={alert.ratio:.2f} label={alert.label}")
    else:
        click.echo("\n  No alerts recorded.")

    spine.sqlite.close()


@main.command(name="graph")
@click.argument("entity")
@click.option("--depth", "-d", default=2, type=int, help="Traversal depth")
@click.pass_context
def graph_cmd(ctx: click.Context, entity: str, depth: int) -> None:
    """Query memory graph around an entity."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    graph = asyncio.run(spine.graph_query(entity, depth=depth))
    click.echo(f"Entity: {graph.get('entity', entity)}")
    click.echo(f"Nodes: {len(graph.get('nodes', []))}")
    click.echo(f"Edges: {len(graph.get('edges', []))}")
    for e in graph.get("edges", [])[:10]:
        click.echo(f"  {e.get('src_name')} -[{e.get('relation')}]-> {e.get('dst_name')}")
    spine.sqlite.close()


@main.command(name="consolidate")
@click.option("--session-id", default=None, help="Optional session id filter")
@click.option("--max-chunks", default=1000, type=int, help="Max chunks to process")
@click.pass_context
def consolidate_cmd(ctx: click.Context, session_id: str | None, max_chunks: int) -> None:
    """Run episodic->semantic consolidation."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    try:
        result = asyncio.run(spine.consolidate_async(session_id=session_id, max_chunks=max_chunks))
        click.echo(result)
    finally:
        asyncio.run(spine.close())


@main.command(name="dream")
@click.pass_context
def dream_cmd(ctx: click.Context) -> None:
    """Run an offline-style dream consolidation pass."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    try:
        result = asyncio.run(spine.dream_consolidate_async())
        click.echo(result)
    finally:
        asyncio.run(spine.close())


@main.command(name="doctor")
@click.option("--api-url", default="http://127.0.0.1:8420", show_default=True, help="NovaSpine API URL to probe")
@click.option("--openclaw-config", default=None, help="Path to openclaw.json to inspect")
@click.option("--install-root", default=None, help="NovaSpine OpenClaw install root to inspect")
@click.option("--skip-api-check", is_flag=True, help="Do not probe the HTTP health endpoint")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON")
@click.pass_context
def doctor_cmd(
    ctx: click.Context,
    api_url: str,
    openclaw_config: str | None,
    install_root: str | None,
    skip_api_check: bool,
    json_output: bool,
) -> None:
    """Inspect NovaSpine and OpenClaw integration health."""
    data_dir = _default_data_dir(ctx.obj.get("data_dir"))
    install_root_path = Path(install_root).expanduser() if install_root else _default_openclaw_install_root()
    openclaw_config_path = Path(openclaw_config).expanduser() if openclaw_config else _default_openclaw_config_path()

    spine = _get_spine(ctx.obj.get("data_dir"))
    try:
        status = spine.status()
    finally:
        spine.sqlite.close()

    checks: list[dict[str, str]] = []
    checks.append(_doctor_check("data-dir", "ok" if data_dir.exists() else "fail", str(data_dir)))
    checks.append(_doctor_check(
        "memory-core",
        "ok",
        f"chunks={status.get('chunks', 0)} consolidated={status.get('consolidated_memories', 0)} graph_entities={((status.get('graph') or {}).get('entities', 0))}",
    ))

    required_install_paths = [
        install_root_path / "packages" / "openclaw-memory-plugin",
        install_root_path / "packages" / "openclaw-context-engine",
        install_root_path / "packages" / "openclaw-consciousness",
        install_root_path / "scripts" / "run-memory-maintenance.sh",
        install_root_path / "scripts" / "run-consciousness-suite.sh",
    ]
    missing_install_paths = [str(path) for path in required_install_paths if not path.exists()]
    checks.append(_doctor_check(
        "openclaw-install-root",
        "ok" if not missing_install_paths else "warn",
        str(install_root_path) if not missing_install_paths else f"missing: {', '.join(missing_install_paths)}",
    ))

    if skip_api_check:
        checks.append(_doctor_check("api-health", "warn", "skipped"))
    else:
        ok, detail = _probe_json(f"{api_url.rstrip('/')}/api/v1/health")
        checks.append(_doctor_check("api-health", "ok" if ok else "warn", detail))

    if not openclaw_config_path.exists():
        checks.append(_doctor_check("openclaw-config", "warn", f"not found: {openclaw_config_path}"))
    else:
        try:
            config_data = json.loads(openclaw_config_path.read_text())
            plugins = config_data.get("plugins") if isinstance(config_data, dict) else {}
            allow = (plugins.get("allow") or []) if isinstance(plugins, dict) else []
            load = (plugins.get("load") or {}) if isinstance(plugins, dict) else {}
            slots = (plugins.get("slots") or {}) if isinstance(plugins, dict) else {}
            paths = (load.get("paths") or []) if isinstance(load, dict) else []
            entries = (plugins.get("entries") or {}) if isinstance(plugins, dict) else {}

            expected_plugin_ids = ("novaspine-memory", "novaspine-context", "nova-consciousness")
            missing_allow = [plugin_id for plugin_id in expected_plugin_ids if plugin_id not in allow]
            missing_paths = [
                str(candidate)
                for candidate in [
                    install_root_path / "packages" / "openclaw-memory-plugin",
                    install_root_path / "packages" / "openclaw-context-engine",
                    install_root_path / "packages" / "openclaw-consciousness",
                ]
                if str(candidate) not in paths
            ]
            bad_slots = []
            if slots.get("memory") != "novaspine-memory":
                bad_slots.append(f"memory={slots.get('memory')!r}")
            if slots.get("contextEngine") != "novaspine-context":
                bad_slots.append(f"contextEngine={slots.get('contextEngine')!r}")

            entry_notes = []
            for plugin_id in expected_plugin_ids:
                entry = _find_plugin_entry(entries, plugin_id)
                if not entry:
                    entry_notes.append(f"{plugin_id}:missing")
                    continue
                enabled = entry.get("enabled")
                config = entry.get("config") if isinstance(entry.get("config"), dict) else {}
                base_url = config.get("baseUrl", "")
                entry_notes.append(f"{plugin_id}:{'on' if enabled is not False else 'off'}:{base_url or 'no-base-url'}")

            detail_parts = []
            if missing_allow:
                detail_parts.append(f"allow missing {missing_allow}")
            if missing_paths:
                detail_parts.append(f"paths missing {missing_paths}")
            if bad_slots:
                detail_parts.append(f"slots {'; '.join(bad_slots)}")
            detail_parts.append(", ".join(entry_notes))

            level = "ok" if not missing_allow and not missing_paths and not bad_slots else "fail"
            checks.append(_doctor_check("openclaw-config", level, " | ".join(detail_parts)))
        except Exception as error:
            checks.append(_doctor_check("openclaw-config", "fail", f"{openclaw_config_path}: {error}"))

    counts = {
        "ok": sum(1 for check in checks if check["level"] == "ok"),
        "warn": sum(1 for check in checks if check["level"] == "warn"),
        "fail": sum(1 for check in checks if check["level"] == "fail"),
    }
    payload = {
        "data_dir": str(data_dir),
        "install_root": str(install_root_path),
        "openclaw_config": str(openclaw_config_path),
        "memory_status": status,
        "checks": checks,
        "summary": counts,
    }

    if json_output:
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo("NovaSpine Doctor")
        click.echo(f"  data dir:         {payload['data_dir']}")
        click.echo(f"  install root:     {payload['install_root']}")
        click.echo(f"  openclaw config:  {payload['openclaw_config']}")
        click.echo(
            "  memory status:    "
            f"chunks={status.get('chunks', 0)} "
            f"consolidated={status.get('consolidated_memories', 0)} "
            f"reasoning={status.get('reasoning_entries', 0)} "
            f"skills={status.get('skills', 0)}"
        )
        for check in checks:
            icon = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}[check["level"]]
            click.echo(f"  [{icon}] {check['name']}: {check['detail']}")
        click.echo(
            f"  summary: ok={counts['ok']} warn={counts['warn']} fail={counts['fail']}"
        )

    if counts["fail"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
