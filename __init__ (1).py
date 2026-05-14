"""
mapping.py
==========
SLAM-inspired occupancy grid mapping for the lunar rover.

The mapper maintains a probabilistic occupancy grid updated via log-odds,
simulates a virtual LiDAR sensor (ray-casting), and tracks frontier cells
for exploration planning.

Occupancy values:
  < 0   → free space (log-odds negative)
  = 0   → unknown
  > 0   → occupied (log-odds positive)
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Set


# ---------------------------------------------------------------------------
# Log-odds update constants (Thrun et al. probabilistic robotics)
# ---------------------------------------------------------------------------

LOG_ODD_FREE      = -0.4     # Update for a free-space observation
LOG_ODD_OCCUPIED  =  0.9     # Update for an occupied observation
LOG_ODD_PRIOR     =  0.0     # Prior (unknown)
LOG_ODD_MIN       = -5.0     # Clamp floor
LOG_ODD_MAX       =  5.0     # Clamp ceiling

PROB_OCCUPIED_THRESH = 0.65  # log_odds > log(0.65/0.35) → occupied


def log_odds_to_prob(lo: float) -> float:
    return 1.0 - 1.0 / (1.0 + math.exp(lo))


def prob_to_log_odds(p: float) -> float:
    return math.log(p / (1.0 - p + 1e-9) + 1e-9)


# ---------------------------------------------------------------------------
# Virtual LiDAR sensor
# ---------------------------------------------------------------------------

class LidarSensor:
    """
    Ray-casting LiDAR sensor for occupancy grid updates.

    Casts rays from the rover's position and returns (free, hit) cell lists.
    """

    def __init__(self,
                 max_range: int = 12,
                 fov_deg: float = 360.0,
                 n_rays: int = 72,
                 noise_sigma: float = 0.3):
        self.max_range = max_range
        self.fov_deg = fov_deg
        self.n_rays = n_rays
        self.noise_sigma = noise_sigma          # Range noise (cells)

    def scan(self, position: Tuple[int, int],
             obstacle_map: np.ndarray,
             heading_rad: float = 0.0) -> Tuple[List[Tuple[int, int]],
                                                 List[Tuple[int, int]]]:
        """
        Perform a LiDAR scan from *position*.

        Parameters
        ----------
        position     : (row, col) rover position.
        obstacle_map : Binary 2-D array (True = obstacle).
        heading_rad  : Rover heading (for partial FOV scans).

        Returns
        -------
        (free_cells, hit_cells) — lists of (row, col) grid cells.
        """
        H, W = obstacle_map.shape
        row, col = position
        free_cells: List[Tuple[int, int]] = []
        hit_cells:  List[Tuple[int, int]] = []

        half_fov = math.radians(self.fov_deg / 2.0)
        angle_start = heading_rad - half_fov
        angle_step  = math.radians(self.fov_deg) / self.n_rays

        for i in range(self.n_rays):
            angle = angle_start + i * angle_step
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)

            # Add noise to simulated range
            max_r = self.max_range + np.random.normal(0, self.noise_sigma)
            max_r = max(1, min(int(max_r), self.max_range + 2))

            hit = False
            for r in range(1, max_r + 1):
                nr = int(round(row + r * (-sin_a)))   # row increases downward
                nc = int(round(col + r * cos_a))

                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break

                cell = (nr, nc)
                if obstacle_map[nr, nc]:
                    hit_cells.append(cell)
                    hit = True
                    break
                else:
                    free_cells.append(cell)

        return free_cells, hit_cells


# ---------------------------------------------------------------------------
# Occupancy grid mapper
# ---------------------------------------------------------------------------

class OccupancyGridMapper:
    """
    Probabilistic occupancy grid map maintained via log-odds updates.

    Attributes
    ----------
    log_odds_grid   : 2-D float array of log-odds values.
    known_mask      : Boolean mask — True for cells that have been observed.
    frontier_cells  : Set of (row, col) cells on the exploration frontier.
    """

    def __init__(self, grid_size: int, sensor: Optional[LidarSensor] = None):
        self.grid_size = grid_size
        self.sensor = sensor or LidarSensor()

        self.log_odds_grid: np.ndarray = np.full(
            (grid_size, grid_size), LOG_ODD_PRIOR, dtype=np.float32
        )
        self.known_mask: np.ndarray = np.zeros(
            (grid_size, grid_size), dtype=bool
        )
        self.frontier_cells: Set[Tuple[int, int]] = set()
        self._scan_count: int = 0

    # ------------------------------------------------------------------
    # Grid update
    # ------------------------------------------------------------------

    def update(self, position: Tuple[int, int],
               obstacle_map: np.ndarray,
               heading_rad: float = 0.0) -> None:
        """
        Perform a LiDAR scan from *position* and update the occupancy grid.
        """
        free_cells, hit_cells = self.sensor.scan(position, obstacle_map,
                                                  heading_rad)

        for (r, c) in free_cells:
            self.log_odds_grid[r, c] = np.clip(
                self.log_odds_grid[r, c] + LOG_ODD_FREE,
                LOG_ODD_MIN, LOG_ODD_MAX
            )
            self.known_mask[r, c] = True

        for (r, c) in hit_cells:
            self.log_odds_grid[r, c] = np.clip(
                self.log_odds_grid[r, c] + LOG_ODD_OCCUPIED,
                LOG_ODD_MIN, LOG_ODD_MAX
            )
            self.known_mask[r, c] = True

        self._scan_count += 1
        self._update_frontiers()

    # ------------------------------------------------------------------
    # Frontier detection
    # ------------------------------------------------------------------

    def _update_frontiers(self) -> None:
        """
        A frontier cell is known-free adjacent to at least one unknown cell.
        """
        H, W = self.log_odds_grid.shape
        free_mask = self.log_odds_grid < 0
        unknown_mask = ~self.known_mask
        frontiers: Set[Tuple[int, int]] = set()

        rows, cols = np.where(free_mask)
        for r, c in zip(rows, cols):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and unknown_mask[nr, nc]:
                        frontiers.add((r, c))
                        break

        self.frontier_cells = frontiers

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def probability_map(self) -> np.ndarray:
        """Return per-cell occupancy probability in [0, 1]."""
        lo = self.log_odds_grid
        return 1.0 - 1.0 / (1.0 + np.exp(lo))

    @property
    def binary_map(self) -> np.ndarray:
        """Return binary occupancy map (True = occupied)."""
        log_thresh = prob_to_log_odds(PROB_OCCUPIED_THRESH)
        return self.log_odds_grid > log_thresh

    @property
    def exploration_coverage(self) -> float:
        """Fraction of grid cells that have been observed."""
        return float(self.known_mask.mean())

    def best_frontier(self, rover_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Return the nearest frontier cell to *rover_pos*.

        Returns None if no frontiers exist.
        """
        if not self.frontier_cells:
            return None
        r0, c0 = rover_pos
        best = min(
            self.frontier_cells,
            key=lambda f: (f[0] - r0) ** 2 + (f[1] - c0) ** 2,
        )
        return best

    def get_frontier_list(self) -> List[Tuple[int, int]]:
        return sorted(self.frontier_cells)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        H, W = self.log_odds_grid.shape
        return {
            "grid_size": (H, W),
            "scans_performed": self._scan_count,
            "cells_known": int(self.known_mask.sum()),
            "cells_free": int((self.log_odds_grid < 0).sum()),
            "cells_occupied": int(self.binary_map.sum()),
            "frontier_count": len(self.frontier_cells),
            "exploration_coverage_pct": round(100 * self.exploration_coverage, 1),
        }


# ---------------------------------------------------------------------------
# Exploration trajectory planner (frontier-based)
# ---------------------------------------------------------------------------

class FrontierExplorer:
    """
    Generates exploration goals by targeting the nearest unvisited frontier.
    """

    def __init__(self, mapper: OccupancyGridMapper):
        self.mapper = mapper
        self.visited_frontiers: List[Tuple[int, int]] = []

    def next_goal(self, rover_pos: Tuple[int, int],
                  min_distance: int = 3) -> Optional[Tuple[int, int]]:
        """
        Return the next exploration goal (best frontier not recently visited).
        """
        r0, c0 = rover_pos
        candidates = [
            f for f in self.mapper.frontier_cells
            if f not in self.visited_frontiers
            and math.sqrt((f[0] - r0) ** 2 + (f[1] - c0) ** 2) >= min_distance
        ]
        if not candidates:
            return None

        goal = min(
            candidates,
            key=lambda f: (f[0] - r0) ** 2 + (f[1] - c0) ** 2,
        )
        self.visited_frontiers.append(goal)
        return goal
