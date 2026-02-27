"""C3/Ae CLI."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine


def _get_spine(data_dir: str | None = None) -> MemorySpine:
    config = Config()
    if data_dir:
        config.data_dir = Path(data_dir)
    return MemorySpine(config)


@click.group()
@click.option("--data-dir", envvar="C3AE_DATA_DIR", default=None, help="Data directory")
@click.pass_context
def main(ctx: click.Context, data_dir: str | None) -> None:
    """C3/Ae Memory Stack — persistent cognitive memory system."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show memory system status."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    st = spine.status()
    click.echo("C3/Ae Memory Status")
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
    """Search memory."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    if keyword_only:
        results = spine.search_keyword(query, top_k=top_k)
    else:
        results = asyncio.run(spine.search(query, top_k=top_k))
    if not results:
        click.echo("No results found.")
    else:
        for i, r in enumerate(results, 1):
            click.echo(f"\n--- Result {i} (score: {r.score:.4f}, source: {r.source}) ---")
            click.echo(f"ID: {r.id}")
            preview = r.content[:200].replace("\n", " ")
            click.echo(f"{preview}")
    spine.sqlite.close()


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--source-id", "-s", default="", help="Source identifier")
@click.pass_context
def ingest(ctx: click.Context, path: str, source_id: str) -> None:
    """Ingest a file into memory."""
    spine = _get_spine(ctx.obj.get("data_dir"))
    file_path = Path(path)
    chunk_ids = asyncio.run(spine.ingest_file(file_path))
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
    """Start the HTTP API server."""
    import uvicorn
    from c3ae.api.routes import create_app

    data_dir = ctx.obj.get("data_dir")
    app = create_app(data_dir=data_dir)
    click.echo(f"Starting C3/Ae API on {host}:{port}")
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


if __name__ == "__main__":
    main()
