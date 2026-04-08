from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from c3ae.exceptions import StorageError
from c3ae.storage.faiss_store import FAISSStore


def test_faiss_dimension_mismatch_raises(tmp_path: Path) -> None:
    store = FAISSStore(dims=8, faiss_dir=tmp_path)
    store.add(np.ones(8, dtype=np.float32), "id-1")
    store.save()

    with pytest.raises(StorageError, match="dimension mismatch"):
        FAISSStore(dims=16, faiss_dir=tmp_path)
