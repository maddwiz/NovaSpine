import json

import numpy as np

from c3ae.storage.faiss_store import FAISSStore, faiss


def test_add_replaces_existing_external_id():
    store = FAISSStore(dims=4)
    first = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    second = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    store.add(first, "chunk-1")
    store.add(second, "chunk-1")

    assert store.size == 1
    vec = store.get_vector_by_external_id("chunk-1")
    assert vec is not None
    assert np.allclose(vec, second)


def test_add_batch_dedupes_incoming_and_existing_ids():
    store = FAISSStore(dims=4)
    store.add(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), "a")
    store.add(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), "b")

    batch = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    positions = store.add_batch(batch, ["b", "c", "c", "a"])

    assert len(positions) == 3
    assert store.size == 3
    assert store.get_vector_by_external_id("a") is not None
    assert store.get_vector_by_external_id("b") is not None
    c_vec = store.get_vector_by_external_id("c")
    assert c_vec is not None
    expected_c = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    expected_c /= np.linalg.norm(expected_c)
    assert np.allclose(c_vec, expected_c)


def test_load_repairs_duplicate_ids(tmp_path):
    faiss_dir = tmp_path / "faiss"
    faiss_dir.mkdir()

    index = faiss.IndexFlatIP(4)
    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faiss.normalize_L2(vecs)
    index.add(vecs)
    faiss.write_index(index, str(faiss_dir / "memory.index"))
    (faiss_dir / "memory.idmap").write_text(json.dumps(["dup", "dup", "other"]))

    store = FAISSStore(dims=4, faiss_dir=faiss_dir)

    assert store.size == 2
    assert store._id_map == ["dup", "other"]
    dup_vec = store.get_vector_by_external_id("dup")
    assert dup_vec is not None
    assert np.allclose(dup_vec, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
