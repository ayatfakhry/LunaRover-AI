"""
terrain_generator.py
====================
Synthetic lunar terrain generation using Diamond-Square fractal noise
combined with physically-motivated crater placement.

The output is a normalized heightmap (2D NumPy array) with values in [0, 1],
where 0 = deepest crater floor, 1 = highest ridge peak.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _next_power_of_two(n: int) -> int:
    """Return smallest power-of-two >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Diamond-Square fractal heightmap
# ---------------------------------------------------------------------------

def diamond_square(size: int, roughness: float = 0.6, seed: int = 0) -> np.ndarray:
    """
    Generate a square heightmap via the Diamond-Square algorithm.

    Parameters
    ----------
    size      : Grid side length (will be padded to 2^n + 1).
    roughness : Controls fractal roughness in [0, 1].  Higher = rougher.
    seed      : Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (size, size), values in [0, 1].
    """
    rng = np.random.default_rng(seed)
    n = _next_power_of_two(size - 1)
    grid_size = n + 1
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)

    # Seed corners
    grid[0, 0] = rng.random()
    grid[0, n] = rng.random()
    grid[n, 0] = rng.random()
    grid[n, n] = rng.random()

    step = n
    scale = 1.0

    while step > 1:
        half = step // 2

        # Diamond step
        for y in range(0, n, step):
            for x in range(0, n, step):
                avg = (grid[y, x] + grid[y + step, x] +
                       grid[y, x + step] + grid[y + step, x + step]) / 4.0
                grid[y + half, x + half] = avg + rng.uniform(-scale, scale)

        # Square step
        for y in range(0, grid_size, half):
            for x in range((y + half) % step, grid_size, step):
                neighbors, count = 0.0, 0
                for dy, dx in [(-half, 0), (half, 0), (0, -half), (0, half)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < grid_size and 0 <= nx < grid_size:
                        neighbors += grid[ny, nx]
                        count += 1
                grid[y, x] = neighbors / count + rng.uniform(-scale, scale)

        step = half
        scale *= (2 ** (-roughness))

    # Crop to requested size and normalise
    grid = grid[:size, :size]
    grid -= grid.min()
    grid /= grid.max() + 1e-9
    return grid


# ---------------------------------------------------------------------------
# Crater placement
# ---------------------------------------------------------------------------

def _place_crater(heightmap: np.ndarray, cy: int, cx: int,
                  radius: float, depth: float) -> None:
    """Carve a single bowl-shaped crater into *heightmap* in-place."""
    H, W = heightmap.shape
    y_lo = max(0, int(cy - radius * 1.5))
    y_hi = min(H, int(cy + radius * 1.5) + 1)
    x_lo = max(0, int(cx - radius * 1.5))
    x_hi = min(W, int(cx + radius * 1.5) + 1)

    for y in range(y_lo, y_hi):
        for x in range(x_lo, x_hi):
            dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            if dist < radius:
                # Bowl profile: parabolic floor + raised rim
                t = dist / radius           # 0 at centre, 1 at rim
                profile = -depth * (1.0 - (2 * t - 1) ** 2)
                rim_boost = depth * 0.3 * np.exp(-((t - 1.0) ** 2) / 0.02)
                heightmap[y, x] += profile + rim_boost


def add_craters(heightmap: np.ndarray, n_craters: int = 30,
                seed: int = 0) -> np.ndarray:
    """
    Overlay *n_craters* craters of varying size onto *heightmap*.

    Returns
    -------
    Modified heightmap (copy), values re-normalised to [0, 1].
    """
    rng = np.random.default_rng(seed + 99)
    H, W = heightmap.shape
    hmap = heightmap.copy()

    for _ in range(n_craters):
        cy = rng.integers(0, H)
        cx = rng.integers(0, W)
        radius = rng.uniform(3, max(4, min(H, W) // 8))
        depth = rng.uniform(0.05, 0.25)
        _place_crater(hmap, cy, cx, radius, depth)

    hmap -= hmap.min()
    hmap /= hmap.max() + 1e-9
    return hmap


# ---------------------------------------------------------------------------
# Rock field overlay
# ---------------------------------------------------------------------------

def add_rocks(heightmap: np.ndarray, density: float = 0.02,
              seed: int = 0) -> np.ndarray:
    """
    Scatter small Gaussian rock bumps across *heightmap*.

    Parameters
    ----------
    density : Fraction of cells that seed a rock [0, 1].
    """
    rng = np.random.default_rng(seed + 7)
    H, W = heightmap.shape
    hmap = heightmap.copy()
    n_rocks = int(density * H * W)

    for _ in range(n_rocks):
        ry = rng.integers(1, H - 1)
        rx = rng.integers(1, W - 1)
        height = rng.uniform(0.02, 0.08)
        sigma = rng.uniform(0.5, 1.5)
        bump = np.zeros_like(hmap)
        bump[ry, rx] = height
        bump = gaussian_filter(bump, sigma=sigma)
        hmap += bump

    hmap -= hmap.min()
    hmap /= hmap.max() + 1e-9
    return hmap


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class LunarTerrainGenerator:
    """
    High-level interface for generating a complete synthetic lunar terrain.

    Attributes
    ----------
    heightmap      : 2-D float array [0, 1], shape (grid_size, grid_size).
    slope_map      : Gradient magnitude (radians, approx).
    hazard_map     : Boolean mask — True where terrain is hazardous.
    terrain_labels : Integer class label per cell (see TERRAIN_CLASSES).
    """

    TERRAIN_CLASSES = {
        0: "FLAT",
        1: "SLOPE",
        2: "ROCKY",
        3: "CRATER_RIM",
        4: "CRATER_FLOOR",
    }

    def __init__(self, grid_size: int = 128, roughness: float = 0.55,
                 n_craters: int = 25, rock_density: float = 0.015,
                 seed: int = 42):
        self.grid_size = grid_size
        self.roughness = roughness
        self.n_craters = n_craters
        self.rock_density = rock_density
        self.seed = seed

        self.heightmap: np.ndarray = np.zeros((grid_size, grid_size))
        self.slope_map: np.ndarray = np.zeros((grid_size, grid_size))
        self.hazard_map: np.ndarray = np.zeros((grid_size, grid_size), dtype=bool)
        self.terrain_labels: np.ndarray = np.zeros((grid_size, grid_size), dtype=int)

    # ------------------------------------------------------------------
    def generate(self) -> "LunarTerrainGenerator":
        """Run the full terrain generation pipeline and return self."""
        # 1. Base fractal heightmap
        base = diamond_square(self.grid_size, self.roughness, self.seed)

        # 2. Apply light Gaussian smoothing (lunar regolith effect)
        base = gaussian_filter(base, sigma=1.2)

        # 3. Add craters
        base = add_craters(base, self.n_craters, self.seed)

        # 4. Add rocks
        base = add_rocks(base, self.rock_density, self.seed)

        self.heightmap = base

        # 5. Derived maps
        self._compute_slope_map()
        self._compute_hazard_map()
        self._classify_terrain()

        return self

    # ------------------------------------------------------------------
    def _compute_slope_map(self) -> None:
        """Compute per-cell slope magnitude (gradient) in normalised units."""
        gy, gx = np.gradient(self.heightmap)
        self.slope_map = np.sqrt(gx ** 2 + gy ** 2)
        # Normalise to [0, 1]
        self.slope_map /= self.slope_map.max() + 1e-9

    # ------------------------------------------------------------------
    def _compute_hazard_map(self,
                            slope_thresh: float = 0.35,
                            height_floor: float = 0.15) -> None:
        """
        Mark cells as hazardous if slope exceeds threshold or elevation
        sits in the deepest crater-floor tier.
        """
        self.hazard_map = (
            (self.slope_map > slope_thresh) |
            (self.heightmap < height_floor)
        )

    # ------------------------------------------------------------------
    def _classify_terrain(self) -> None:
        """Assign integer terrain class labels to each cell."""
        H = self.heightmap
        S = self.slope_map
        labels = np.zeros(H.shape, dtype=int)

        labels[S > 0.35] = 1                           # SLOPE
        labels[(S > 0.25) & (H > 0.55)] = 2            # ROCKY
        labels[(S > 0.30) & (H > 0.60)] = 3            # CRATER_RIM
        labels[H < 0.15] = 4                            # CRATER_FLOOR

        self.terrain_labels = labels

    # ------------------------------------------------------------------
    def get_traversal_cost(self, slope_weight: float = 2.0,
                           hazard_penalty: float = 1e6) -> np.ndarray:
        """
        Build a per-cell traversal cost array for path planners.

        Cost = 1 + slope_weight * slope  (+ hazard_penalty if hazardous)
        """
        cost = 1.0 + slope_weight * self.slope_map
        cost[self.hazard_map] += hazard_penalty
        return cost

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a dict of terrain statistics."""
        return {
            "grid_size": self.grid_size,
            "elevation_min": float(self.heightmap.min()),
            "elevation_max": float(self.heightmap.max()),
            "elevation_mean": float(self.heightmap.mean()),
            "slope_mean": float(self.slope_map.mean()),
            "hazard_fraction": float(self.hazard_map.mean()),
            "class_counts": {
                self.TERRAIN_CLASSES[k]: int((self.terrain_labels == k).sum())
                for k in self.TERRAIN_CLASSES
            },
        }
