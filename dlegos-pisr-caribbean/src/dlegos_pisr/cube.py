"""Spatiotemporal cube containers and overlap-aware spatial tiling.

Tensor convention: (batch, variable, time, latitude, longitude).
Coordinates are optional; arrays remain ordinary PyTorch tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch


@dataclass(frozen=True)
class CubeSpec:
    """Coordinate and variable metadata for a regular gridded cube."""

    variables: tuple[str, ...]
    latitudes: tuple[float, ...] | None = None
    longitudes: tuple[float, ...] | None = None
    times: tuple[str, ...] | None = None
    lat_spacing_deg: float | None = None
    lon_spacing_deg: float | None = None

    def validate(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 5:
            raise ValueError("Cube tensors must have shape (batch, variable, time, lat, lon).")
        if tensor.shape[1] != len(self.variables):
            raise ValueError(
                f"Variable metadata has {len(self.variables)} names, "
                f"but tensor has {tensor.shape[1]} channels."
            )
        for name, coords, size in (
            ("latitudes", self.latitudes, tensor.shape[-2]),
            ("longitudes", self.longitudes, tensor.shape[-1]),
            ("times", self.times, tensor.shape[2]),
        ):
            if coords is not None and len(coords) != size:
                raise ValueError(f"{name} has length {len(coords)}; expected {size}.")


@dataclass
class SpacetimeCube:
    """A named wrapper around a 5-D PyTorch tensor."""

    values: torch.Tensor
    spec: CubeSpec

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor.")
        self.spec.validate(self.values)

    @property
    def shape(self) -> torch.Size:
        return self.values.shape

    def variable(self, name: str) -> torch.Tensor:
        """Return one variable as (batch, time, lat, lon)."""
        try:
            index = self.spec.variables.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown variable {name!r}. Available: {self.spec.variables}") from exc
        return self.values[:, index]


@dataclass(frozen=True)
class Tile:
    """A spatial window in the final two tensor dimensions."""

    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def iter_tiles(
    tensor: torch.Tensor,
    tile_size: int | tuple[int, int],
    overlap: int | tuple[int, int] = 0,
) -> Iterator[tuple[Tile, torch.Tensor]]:
    """Yield spatial tiles from any tensor with at least two dimensions.

    If a grid is smaller than a tile, the yielded tile covers the full grid;
    this function does not pad inputs.
    """
    if tensor.ndim < 2:
        raise ValueError("Input must have at least two dimensions.")
    th, tw = (tile_size, tile_size) if isinstance(tile_size, int) else tile_size
    oh, ow = (overlap, overlap) if isinstance(overlap, int) else overlap
    h, w = tensor.shape[-2:]
    if min(th, tw) <= 0 or min(oh, ow) < 0 or oh >= th or ow >= tw:
        raise ValueError("Tile sizes must be positive; overlap must be non-negative and smaller than tile size.")
    th, tw = min(th, h), min(tw, w)
    ys, xs = _starts(h, th, th - oh), _starts(w, tw, tw - ow)
    for y0 in ys:
        for x0 in xs:
            tile = Tile(y0, y0 + th, x0, x0 + tw)
            sy, sx = tile.slices
            yield tile, tensor[..., sy, sx]


def stitch_tiles(
    tiles: Sequence[tuple[Tile, torch.Tensor]],
    output_shape: tuple[int, int],
) -> torch.Tensor:
    """Reconstruct a field by uniform weighted overlap-add.

    The leading dimensions are taken from the first tile. Every output pixel
    must be covered by at least one tile.
    """
    if not tiles:
        raise ValueError("At least one tile is required.")
    h, w = output_shape
    first = tiles[0][1]
    out = first.new_zeros((*first.shape[:-2], h, w))
    weights = first.new_zeros((h, w))
    for window, value in tiles:
        sy, sx = window.slices
        if value.shape[-2:] != (window.y1 - window.y0, window.x1 - window.x0):
            raise ValueError("Tile tensor shape does not match its Tile window.")
        out[..., sy, sx] += value
        weights[sy, sx] += 1
    if torch.any(weights == 0):
        raise ValueError("Tile set leaves uncovered pixels.")
    return out / weights
