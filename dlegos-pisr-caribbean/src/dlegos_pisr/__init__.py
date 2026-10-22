"""D-LEGOS-PISR: modular physics-informed gridded field modeling."""

from .cube import CubeSpec, SpacetimeCube, Tile, iter_tiles, stitch_tiles
from .metrics import (
    critical_success_index,
    fractions_skill_score,
    mae,
    precipitation_scores,
    rmse,
)
from .model import DLegosPISR, MultiVarEDSR
from .physics import interface_consistency, normalized_physical_consistency, physics_terms

__all__ = [
    "CubeSpec", "SpacetimeCube", "Tile", "iter_tiles", "stitch_tiles",
    "DLegosPISR", "MultiVarEDSR", "physics_terms",
    "normalized_physical_consistency", "interface_consistency",
    "mae", "rmse", "critical_success_index", "fractions_skill_score",
    "precipitation_scores",
]
__version__ = "0.1.0"
