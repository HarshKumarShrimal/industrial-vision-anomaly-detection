"""Greedy k-center coreset subsampling for PatchCore.

PatchCore stores a *coreset* of the training patch features rather than the
full set. The coreset is chosen by greedy furthest-point (k-center) sampling so
that the retained points cover the feature manifold as evenly as possible.

To make the O(N x M) distance computation affordable, the original paper first
projects features to a lower dimension with a sparse Gaussian random matrix
(Johnson-Lindenstrauss). We reproduce that here.
"""
from __future__ import annotations

import torch
from tqdm.auto import tqdm


class KCenterGreedy:
    """Greedy k-center (furthest-point) coreset selection.

    Args:
        projection_dim: Target dimensionality of the random projection used only
            to *measure* distances. ``None`` disables projection.
        seed: RNG seed for the projection and the first centre — keeps coresets
            reproducible across runs.
    """

    def __init__(self, projection_dim: int | None = 128, seed: int = 42) -> None:
        self.projection_dim = projection_dim
        self.seed = seed

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        if self.projection_dim is None or self.projection_dim >= features.shape[1]:
            return features
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        proj = torch.randn(
            features.shape[1], self.projection_dim, generator=generator
        ) / (self.projection_dim ** 0.5)
        return features @ proj.to(features.device, features.dtype)

    @torch.no_grad()
    def sample(self, features: torch.Tensor, ratio: float) -> torch.Tensor:
        """Return indices of the selected coreset.

        Args:
            features: ``(N, D)`` patch feature bank.
            ratio: Fraction of points to keep, in ``(0, 1]``.

        Returns:
            A 1D LongTensor of selected row indices into ``features``.
        """
        n = features.shape[0]
        n_select = max(1, int(round(n * ratio)))
        if n_select >= n:
            return torch.arange(n, device=features.device)

        proj = self._project(features)

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        start = int(torch.randint(0, n, (1,), generator=generator).item())

        selected = torch.empty(n_select, dtype=torch.long, device=features.device)
        selected[0] = start
        # Distance from every point to the nearest selected centre so far.
        min_dist = torch.cdist(proj, proj[start : start + 1]).squeeze(1)

        # The greedy loop is the slow part on CPU; show progress so it never
        # looks like the run has hung.
        iterator = tqdm(
            range(1, n_select),
            desc=f"Coreset ({ratio:.0%}, {n_select}/{n})",
            leave=False,
        )
        for i in iterator:
            new_idx = int(torch.argmax(min_dist).item())
            selected[i] = new_idx
            dist_new = torch.cdist(proj, proj[new_idx : new_idx + 1]).squeeze(1)
            min_dist = torch.minimum(min_dist, dist_new)

        return selected
