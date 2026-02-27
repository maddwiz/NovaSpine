"""FAISS vector index wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from c3ae.exceptions import StorageError


class FAISSStore:
    """FAISS vector index with ID mapping and persistence."""

    def __init__(self, dims: int = 1024, faiss_dir: Path | str | None = None,
                 ivf_threshold: int = 50_000) -> None:
        self.dims = dims
        self.faiss_dir = Path(faiss_dir) if faiss_dir else None
        self.ivf_threshold = ivf_threshold
        # rowid → external ID mapping
        self._id_map: list[str] = []
        self._id_to_pos: dict[str, int] = {}
        self._index: faiss.Index = faiss.IndexFlatIP(dims)
        self._trained = True
        if self.faiss_dir:
            self.faiss_dir.mkdir(parents=True, exist_ok=True)
            self._try_load()

    @property
    def size(self) -> int:
        return self._index.ntotal

    def add(self, vector: np.ndarray, external_id: str) -> int:
        """Add a single L2-normalized vector. Returns internal index."""
        vec = np.ascontiguousarray(vector.reshape(1, -1).astype(np.float32))
        faiss.normalize_L2(vec)
        idx = self._index.ntotal
        self._index.add(vec)
        self._id_map.append(external_id)
        self._id_to_pos[external_id] = idx
        return idx

    def add_batch(self, vectors: np.ndarray, external_ids: list[str]) -> list[int]:
        """Add multiple L2-normalized vectors."""
        if len(vectors) != len(external_ids):
            raise StorageError("vectors and external_ids length mismatch")
        vecs = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(vecs)
        start_idx = self._index.ntotal
        self._index.add(vecs)
        self._id_map.extend(external_ids)
        for offset, external_id in enumerate(external_ids):
            self._id_to_pos[external_id] = start_idx + offset
        return list(range(start_idx, start_idx + len(external_ids)))

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        """Search for nearest neighbors. Returns [(external_id, score), ...]."""
        if self._index.ntotal == 0:
            return []
        vec = np.ascontiguousarray(query_vector.reshape(1, -1).astype(np.float32))
        faiss.normalize_L2(vec)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    def get_vector_by_external_id(self, external_id: str) -> np.ndarray | None:
        """Return stored normalized vector for an external ID if present."""
        idx = self._id_to_pos.get(external_id)
        if idx is None:
            return None
        if idx < 0 or idx >= self._index.ntotal:
            return None
        try:
            return self._index.reconstruct(int(idx)).astype(np.float32)
        except Exception:
            return None

    def get_vectors_by_external_ids(self, external_ids: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for eid in external_ids:
            vec = self.get_vector_by_external_id(eid)
            if vec is not None:
                out[eid] = vec
        return out

    def remove(self, external_id: str) -> bool:
        """Remove by external ID. Rebuilds index (expensive)."""
        if external_id not in self._id_map:
            return False
        idx = self._id_map.index(external_id)
        # Reconstruct all vectors except the one to remove
        n = self._index.ntotal
        if n <= 1:
            self._index = faiss.IndexFlatIP(self.dims)
            self._id_map = []
            self._id_to_pos = {}
            return True
        all_vecs = np.zeros((n, self.dims), dtype=np.float32)
        for i in range(n):
            all_vecs[i] = self._index.reconstruct(i)
        keep_mask = list(range(n))
        keep_mask.pop(idx)
        keep_vecs = all_vecs[keep_mask]
        self._id_map.pop(idx)
        self._id_to_pos = {eid: i for i, eid in enumerate(self._id_map)}
        self._index = faiss.IndexFlatIP(self.dims)
        if len(keep_vecs) > 0:
            self._index.add(keep_vecs)
        return True

    def maybe_upgrade_to_ivf(self) -> bool:
        """Upgrade to IVF index if threshold exceeded."""
        if self._index.ntotal < self.ivf_threshold:
            return False
        if not isinstance(self._index, faiss.IndexFlatIP):
            return False  # Already upgraded
        n = self._index.ntotal
        all_vecs = np.zeros((n, self.dims), dtype=np.float32)
        for i in range(n):
            all_vecs[i] = self._index.reconstruct(i)
        nlist = max(int(np.sqrt(n)), 16)
        quantizer = faiss.IndexFlatIP(self.dims)
        ivf_index = faiss.IndexIVFFlat(quantizer, self.dims, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf_index.train(all_vecs)
        ivf_index.add(all_vecs)
        ivf_index.nprobe = 16
        self._index = ivf_index
        self._trained = True
        return True

    def save(self) -> None:
        """Persist index and ID map to disk."""
        if not self.faiss_dir:
            raise StorageError("No faiss_dir configured")
        faiss.write_index(self._index, str(self.faiss_dir / "memory.index"))
        with open(self.faiss_dir / "memory.idmap", "w") as f:
            json.dump(self._id_map, f)

    def _try_load(self) -> None:
        """Load index from disk if files exist."""
        index_path = self.faiss_dir / "memory.index"
        idmap_path = self.faiss_dir / "memory.idmap"
        if index_path.exists() and idmap_path.exists():
            self._index = faiss.read_index(str(index_path))
            with open(idmap_path) as f:
                self._id_map = json.load(f)
            self._id_to_pos = {eid: i for i, eid in enumerate(self._id_map)}
