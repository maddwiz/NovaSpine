"""C3/Ae CLI."""

from __future__ import annotations

import asyncio
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


@main.command(name="rebuild-index")
@click.option(
    "--provider",
    type=click.Choice(["venice", "openai", "ollama", "local"], case_sensitive=False),
    default=None,
    help="Override C3AE_EMBEDDING_PROVIDER for this rebuild.",
)
@click.pass_context
def rebuild_index(ctx: click.Context, provider: str | None) -> None:
    """Rebuild FAISS index from all stored chunks."""
    import os
    import faiss
    from c3ae.storage.faiss_store import FAISSStore

    if provider:
        os.environ["C3AE_EMBEDDING_PROVIDER"] = provider

    spine = _get_spine(ctx.obj.get("data_dir"))
    rows = spine.sqlite._conn.execute(
        "SELECT id, content FROM chunks ORDER BY created_at ASC"
    ).fetchall()
    if not rows:
        click.echo("No chunks found; nothing to index.")
        spine.sqlite.close()
        return

    chunk_ids = [str(r["id"]) for r in rows]
    texts = [str(r["content"]) for r in rows]

    vectors = asyncio.run(spine.embedder.embed(texts))
    new_store = FAISSStore(
        dims=spine.embedder.dimensions,
        faiss_dir=spine.config.faiss_dir,
        ivf_threshold=spine.config.retrieval.faiss_ivf_threshold,
    )
    new_store._index = faiss.IndexFlatIP(spine.embedder.dimensions)
    new_store._id_map = []
    new_store.add_batch(vectors, chunk_ids)
    new_store.save()

    click.echo(
        f"Rebuilt FAISS index with {len(chunk_ids)} vectors "
        f"({spine.embedder.dimensions} dims, provider={spine.config.embedding.provider})."
    )
    spine.sqlite.close()


if __name__ == "__main__":
    main()
