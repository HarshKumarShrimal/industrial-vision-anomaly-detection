"""Nearest-neighbour search backend for PatchCore.

Uses FAISS when available (fast, used on Colab/Kaggle Linux) and falls back to a
chunked ``torch.cdist`` implementation otherwise (e.g. on macOS where
``faiss-cpu`` wheels are flaky). The two paths return identical results.
"""
from __future__ import annotations

import torch

try:  # pragma: no cover - exercised implicitly by environment
    import faiss

    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    _HAS_FAISS = False


class NearestNeighbors:
    """Exact L2 nearest-neighbour index over a fixed memory bank."""

    def __init__(self, bank: torch.Tensor, chunk: int = 4096) -> None:
        self.chunk = chunk
        self.dim = bank.shape[1]
        if _HAS_FAISS:
            self._index = faiss.IndexFlatL2(self.dim)
            self._index.add(bank.detach().cpu().numpy().astype("float32"))
            self._bank = None
        else:
            self._index = None
            self._bank = bank.detach()

    def search(self, queries: torch.Tensor, k: int = 1):
        """Return ``(distances, indices)`` of the ``k`` nearest neighbours.

        Distances are Euclidean (not squared). Shapes: ``(Q, k)`` each.
        """
        if self._index is not None:
            sq_dist, idx = self._index.search(
                queries.detach().cpu().numpy().astype("float32"), k
            )
            dist = torch.from_numpy(sq_dist).clamp_min(0.0).sqrt()
            return dist, torch.from_numpy(idx).long()

        # Torch fallback: chunked to bound memory.
        bank = self._bank.to(queries.device)
        all_dist, all_idx = [], []
        for start in range(0, queries.shape[0], self.chunk):
            block = queries[start : start + self.chunk]
            d = torch.cdist(block, bank)  # (chunk, M)
            dist, idx = d.topk(k, dim=1, largest=False)
            all_dist.append(dist)
            all_idx.append(idx)
        return torch.cat(all_dist, 0), torch.cat(all_idx, 0)
