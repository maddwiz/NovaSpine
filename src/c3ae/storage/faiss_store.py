"""FAISS vector index wrapper."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"builtin type SwigPyPacked has no __module__ attribute",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"builtin type SwigPyObject has no __module__ attribute",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"builtin type swigvarlink has no __module__ attribute",
        category=DeprecationWarning,
    )
    import faiss

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

    def _snapshot_rows(self) -> tuple[list[str], np.ndarray]:
        count = min(len(self._id_map), self._index.ntotal)
        if count <= 0:
            return [], np.zeros((0, self.dims), dtype=np.float32)
        vecs = np.zeros((count, self.dims), dtype=np.float32)
        for i in range(count):
            vecs[i] = self._index.reconstruct(i)
        return list(self._id_map[:count]), vecs

    def _dedupe_rows(
        self, external_ids: list[str], vectors: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        if not external_ids:
            return [], np.zeros((0, self.dims), dtype=np.float32)
        last_pos: dict[str, int] = {}
        for idx, external_id in enumerate(external_ids):
            last_pos[external_id] = idx
        keep_positions = sorted(last_pos.values())
        keep_ids = [external_ids[i] for i in keep_positions]
        keep_vecs = vectors[keep_positions] if len(keep_positions) > 0 else np.zeros(
            (0, self.dims), dtype=np.float32
        )
        return keep_ids, keep_vecs

    def _rebuild_index(self, vectors: np.ndarray, external_ids: list[str]) -> None:
        self._index = faiss.IndexFlatIP(self.dims)
        self._id_map = list(external_ids)
        self._id_to_pos = {eid: i for i, eid in enumerate(self._id_map)}
        if len(external_ids) == 0:
            return
        vecs = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(vecs)
        self._index.add(vecs)
        self.maybe_upgrade_to_ivf()

    def _repair_loaded_state(self) -> None:
        row_count = min(len(self._id_map), self._index.ntotal)
        duplicate_ids = len(set(self._id_map[:row_count])) != row_count
        has_mismatch = self._index.ntotal != len(self._id_map)
        if not duplicate_ids and not has_mismatch:
            self._id_to_pos = {eid: i for i, eid in enumerate(self._id_map)}
            return
        ext_ids, vecs = self._snapshot_rows()
        ext_ids, vecs = self._dedupe_rows(ext_ids, vecs)
        self._rebuild_index(vecs, ext_ids)

    def add(self, vector: np.ndarray, external_id: str) -> int:
        """Add a single L2-normalized vector. Returns internal index."""
        return self.add_batch(vector.reshape(1, -1), [external_id])[0]

    def add_batch(self, vectors: np.ndarray, external_ids: list[str]) -> list[int]:
        """Add multiple L2-normalized vectors."""
        if len(vectors) != len(external_ids):
            raise StorageError("vectors and external_ids length mismatch")
        if len(external_ids) == 0:
            return []
        vecs = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(vecs)
        incoming_ids, incoming_vecs = self._dedupe_rows(external_ids, vecs)
        needs_rebuild = (
            len(incoming_ids) != len(external_ids)
            or any(external_id in self._id_to_pos for external_id in incoming_ids)
            or self._index.ntotal != len(self._id_map)
            or len(self._id_map) != len(self._id_to_pos)
        )
        if not needs_rebuild:
            start_idx = self._index.ntotal
            self._index.add(incoming_vecs)
            self._id_map.extend(incoming_ids)
            for offset, external_id in enumerate(incoming_ids):
                self._id_to_pos[external_id] = start_idx + offset
            return [self._id_to_pos[external_id] for external_id in incoming_ids]

        keep_ids, keep_vecs = self._snapshot_rows()
        incoming_set = set(incoming_ids)
        filtered_positions = [i for i, external_id in enumerate(keep_ids) if external_id not in incoming_set]
        filtered_ids = [keep_ids[i] for i in filtered_positions]
        filtered_vecs = keep_vecs[filtered_positions] if filtered_positions else np.zeros(
            (0, self.dims), dtype=np.float32
        )
        combined_ids = filtered_ids + incoming_ids
        if len(filtered_ids) > 0:
            combined_vecs = np.vstack([filtered_vecs, incoming_vecs])
        else:
            combined_vecs = incoming_vecs
        combined_ids, combined_vecs = self._dedupe_rows(combined_ids, combined_vecs)
        self._rebuild_index(combined_vecs, combined_ids)
        return [self._id_to_pos[external_id] for external_id in incoming_ids]

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
        keep_ids, keep_vecs = self._snapshot_rows()
        filtered_positions = [i for i, eid in enumerate(keep_ids) if eid != external_id]
        filtered_ids = [keep_ids[i] for i in filtered_positions]
        filtered_vecs = keep_vecs[filtered_positions] if filtered_positions else np.zeros(
            (0, self.dims), dtype=np.float32
        )
        self._rebuild_index(filtered_vecs, filtered_ids)
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
        index_path = self.faiss_dir / "memory.index"
        idmap_path = self.faiss_dir / "memory.idmap"
        index_tmp = self.faiss_dir / "memory.index.tmp"
        idmap_tmp = self.faiss_dir / "memory.idmap.tmp"
        faiss.write_index(self._index, str(index_tmp))
        with open(idmap_tmp, "w") as f:
            json.dump(self._id_map, f)
        index_tmp.replace(index_path)
        idmap_tmp.replace(idmap_path)

    def _try_load(self) -> None:
        """Load index from disk if files exist."""
        index_path = self.faiss_dir / "memory.index"
        idmap_path = self.faiss_dir / "memory.idmap"
        if index_path.exists() and idmap_path.exists():
            try:
                self._index = faiss.read_index(str(index_path))
                with open(idmap_path) as f:
                    self._id_map = json.load(f)
                self._repair_loaded_state()
            except Exception:
                self._index = faiss.IndexFlatIP(self.dims)
                self._id_map = []
                self._id_to_pos = {}
