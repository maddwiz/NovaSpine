from __future__ import annotations

import numpy as np

from c3ae.embeddings.cache import EmbeddingCache
from c3ae.storage.sqlite_store import SQLiteStore


def test_embedding_cache_is_scoped_by_model(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "cache.db")
    text = "same text"
    vec_a = np.array([1.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0], dtype=np.float32)

    cache_a = EmbeddingCache(store, model="model-a")
    cache_b = EmbeddingCache(store, model="model-b")

    cache_a.put(text, vec_a)
    cache_b.put(text, vec_b)

    got_a = cache_a.get(text)
    got_b = cache_b.get(text)

    assert got_a is not None
    assert got_b is not None
    assert np.allclose(got_a, vec_a)
    assert np.allclose(got_b, vec_b)
