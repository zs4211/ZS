"""Deterministic Tucker/HOSVD feasibility diagnostics.

This module contains representation diagnostics only.  It does not implement
a Tucker manifold, an optimiser, or any retrieval objective.  Arrays may have
any number of modes; Step 5 uses the explicit order ``(x, y, z, time,
component)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.mass_continuity import _derivative as _step3_derivative


Array = np.ndarray


def unfold(values: Array, mode: int) -> Array:
    """Return the mode-``mode`` unfolding with that mode as matrix rows."""
    array = np.asarray(values, dtype=np.float64)
    return np.reshape(np.moveaxis(array, mode, 0), (array.shape[mode], -1))


def mode_product(values: Array, matrix: Array, mode: int) -> Array:
    """Multiply a tensor by ``matrix`` along one mode.

    ``matrix`` has shape ``(new_mode_size, old_mode_size)``.
    """
    array = np.asarray(values, dtype=np.float64)
    factor = np.asarray(matrix, dtype=np.float64)
    if factor.ndim != 2 or factor.shape[1] != array.shape[mode]:
        raise ValueError("matrix second dimension must match the selected mode")
    product = np.tensordot(factor, array, axes=(1, mode))
    return np.moveaxis(product, 0, mode)


def mode_spectrum(values: Array, mode: int) -> dict[str, Array | float | int]:
    """Compute singular values, energy curve and entropy effective rank."""
    singular_values = np.linalg.svd(unfold(values, mode), compute_uv=False)
    energy = singular_values * singular_values
    total = float(np.sum(energy))
    if total == 0.0:
        cumulative = np.zeros_like(energy)
        effective_rank = 0.0
        ranks = {threshold: 0 for threshold in (0.90, 0.95, 0.99)}
    else:
        probabilities = energy / total
        positive = probabilities > 0.0
        effective_rank = float(
            np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        )
        cumulative = np.cumsum(probabilities)
        ranks = {
            threshold: int(np.searchsorted(cumulative, threshold) + 1)
            for threshold in (0.90, 0.95, 0.99)
        }
    return {
        "singular_values": singular_values,
        "cumulative_energy": cumulative,
        "effective_rank": effective_rank,
        "rank_90": ranks[0.90],
        "rank_95": ranks[0.95],
        "rank_99": ranks[0.99],
    }


@dataclass(frozen=True)
class TuckerApproximation:
    core: Array
    factors: tuple[Array, ...]
    reconstruction: Array
    ranks: tuple[int, ...]


def st_hosvd(values: Array, ranks: Sequence[int]) -> TuckerApproximation:
    """Sequentially truncated HOSVD with deterministic SVD factors."""
    array = np.asarray(values, dtype=np.float64)
    requested = tuple(int(rank) for rank in ranks)
    if len(requested) != array.ndim:
        raise ValueError("one rank is required for every tensor mode")
    if any(rank < 1 or rank > size for rank, size in zip(requested, array.shape)):
        raise ValueError("each rank must lie between one and its mode size")

    core = array.copy()
    factors: list[Array] = []
    for mode, rank in enumerate(requested):
        left, _, _ = np.linalg.svd(unfold(core, mode), full_matrices=False)
        factor = left[:, :rank]
        factors.append(factor)
        core = mode_product(core, factor.T, mode)
    reconstruction = core
    for mode, factor in enumerate(factors):
        reconstruction = mode_product(reconstruction, factor, mode)
    return TuckerApproximation(core, tuple(factors), reconstruction, requested)


def tucker_parameter_count(shape: Sequence[int], ranks: Sequence[int]) -> int:
    """Return stored core-plus-factor entries (without quotient correction)."""
    sizes = tuple(int(size) for size in shape)
    requested = tuple(int(rank) for rank in ranks)
    if len(sizes) != len(requested):
        raise ValueError("shape and ranks must have equal length")
    return int(np.prod(requested) + sum(n * r for n, r in zip(sizes, requested)))


def masked_st_hosvd(
    values: Array,
    valid_mask: Array,
    ranks: Sequence[int],
    *,
    iterations: int = 8,
) -> TuckerApproximation:
    """Fit a Tucker representation without zero-filling missing samples.

    Missing entries are initialised by the mean at equal trailing-mode index
    (Step 5: component mean), with the global observed mean as fallback.  Each
    iteration resets observed samples exactly and replaces only missing entries
    with the current ST-HOSVD reconstruction.  The returned reconstruction is
    not clamped to the observations and can therefore be evaluated on them.
    This is a representation diagnostic, not evidence about unobserved truth.
    """
    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != array.shape:
        try:
            mask = np.broadcast_to(mask, array.shape)
        except ValueError as error:
            raise ValueError("valid_mask is not broadcastable to values") from error
    mask = mask & np.isfinite(array)
    if not np.any(mask):
        raise ValueError("at least one finite valid sample is required")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    global_mean = float(np.mean(array[mask]))
    filled = np.full(array.shape, global_mean, dtype=np.float64)
    if array.ndim:
        # The final mode is component in Step 5; initialise each component
        # independently so missing vertical velocity is not seeded by u/v.
        for index in range(array.shape[-1]):
            selector = (..., index)
            component_mask = mask[selector]
            mean = (
                float(np.mean(array[selector][component_mask]))
                if np.any(component_mask)
                else global_mean
            )
            filled[selector] = mean
    filled[mask] = array[mask]
    approximation = None
    for _ in range(iterations):
        approximation = st_hosvd(filled, ranks)
        filled[~mask] = approximation.reconstruction[~mask]
        filled[mask] = array[mask]
    assert approximation is not None
    return st_hosvd(filled, ranks)


def _largest_true_rectangle(mask: Array) -> tuple[int, int, int, int, int]:
    """Return ``(area,y0,y1,x0,x1)`` for a largest all-true 2-D rectangle."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    heights = np.zeros(binary.shape[1], dtype=np.int64)
    best = (0, 0, 0, 0, 0)
    for row in range(binary.shape[0]):
        heights = np.where(binary[row], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for column in range(binary.shape[1] + 1):
            height = int(heights[column]) if column < binary.shape[1] else 0
            start = column
            while stack and stack[-1][1] > height:
                left, previous_height = stack.pop()
                area = previous_height * (column - left)
                candidate = (
                    area,
                    row - previous_height + 1,
                    row + 1,
                    left,
                    column,
                )
                if candidate[0] > best[0]:
                    best = candidate
                start = left
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    return best


def largest_true_cuboid(mask_zyx: Array) -> tuple[slice, slice, slice]:
    """Find an exact largest-volume axis-aligned all-true 3-D cuboid."""
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask_zyx must be three-dimensional")
    best_volume = 0
    best = (slice(0, 0), slice(0, 0), slice(0, 0))
    for z0 in range(mask.shape[0]):
        shared = np.ones(mask.shape[1:], dtype=bool)
        for z1 in range(z0, mask.shape[0]):
            shared &= mask[z1]
            area, y0, y1, x0, x1 = _largest_true_rectangle(shared)
            volume = area * (z1 - z0 + 1)
            if volume > best_volume:
                best_volume = volume
                best = (slice(z0, z1 + 1), slice(y0, y1), slice(x0, x1))
    if best_volume == 0:
        raise ValueError("mask contains no valid cell")
    return best


def relative_error(reference: Array, estimate: Array, mask: Array | None = None) -> float:
    """Euclidean relative error on an optional broadcastable mask."""
    truth = np.asarray(reference, dtype=np.float64)
    approximation = np.asarray(estimate, dtype=np.float64)
    if truth.shape != approximation.shape:
        raise ValueError("reference and estimate must have equal shapes")
    valid = np.isfinite(truth) & np.isfinite(approximation)
    if mask is not None:
        valid &= np.broadcast_to(np.asarray(mask, dtype=bool), truth.shape)
    if not np.any(valid):
        return float("nan")
    denominator = float(np.linalg.norm(truth[valid]))
    numerator = float(np.linalg.norm(approximation[valid] - truth[valid]))
    return numerator / denominator if denominator > 0.0 else numerator


def vertical_vorticity_where_valid(
    u_zyx: Array,
    v_zyx: Array,
    x_m: Array,
    y_m: Array,
    physics_mask_zyx: Array,
) -> tuple[Array, Array]:
    """Return ``dv/dx-du/dy`` using the exact Step 3 difference convention.

    Computational zero fill is used only outside ``physics_mask_zyx``.  The
    caller-supplied mask must already guarantee complete stencils; undefined
    outputs are returned as NaN and therefore cannot masquerade as zero
    vorticity.
    """
    u = np.asarray(u_zyx, dtype=np.float64)
    v = np.asarray(v_zyx, dtype=np.float64)
    mask = np.asarray(physics_mask_zyx, dtype=bool)
    if u.shape != v.shape or u.shape != mask.shape or u.ndim != 3:
        raise ValueError("u, v, and physics_mask must have equal (z,y,x) shape")
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    finite_input = np.isfinite(u) & np.isfinite(v)
    if np.any(mask & ~finite_input):
        raise ValueError("physics_mask marks non-finite input as valid")
    u_filled = np.where(np.isfinite(u), u, 0.0)
    v_filled = np.where(np.isfinite(v), v, 0.0)
    # A Step 3 physics mask is stricter than required here: it guarantees the
    # x stencil for u and y stencil for v (as well as the unused z stencil).
    vorticity = (
        _step3_derivative(v_filled, x, axis=2)
        - _step3_derivative(u_filled, y, axis=1)
    )
    return np.where(mask, vorticity, np.nan), mask


def longest_true_run(values: Array) -> int:
    """Length of the longest consecutive True run in a one-dimensional mask."""
    flags = np.asarray(values, dtype=bool)
    if flags.ndim != 1:
        raise ValueError("values must be one-dimensional")
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best
