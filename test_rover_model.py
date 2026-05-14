"""
path_planner.py
===============
Terrain-aware path planning for the lunar rover.

Implements:
  - A*       : Heuristic-guided optimal search (primary planner)
  - Dijkstra : Exhaustive cost-minimising search (baseline)

Both planners consume a 2-D cost array where each cell's value represents
the traversal cost.  Impassable cells should carry a very large cost (1e6+).

8-directional movement is supported (including diagonals).
"""

import heapq
import math
import numpy as np
from typing import List, Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Cell = Tuple[int, int]          # (row, col)
Path = List[Cell]


# ---------------------------------------------------------------------------
# Movement neighbourhood
# ---------------------------------------------------------------------------

_NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),           ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

def _move_cost(dr: int, dc: int) -> float:
    """Return the geometric cost of a single move (1.0 cardinal, √2 diagonal)."""
    return math.sqrt(2) if (dr != 0 and dc != 0) else 1.0


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def _heuristic_euclidean(a: Cell, b: Cell) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _heuristic_manhattan(a: Cell, b: Cell) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _heuristic_octile(a: Cell, b: Cell) -> float:
    """Octile distance — tight admissible heuristic for 8-directional grids."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_path(came_from: Dict[Cell, Optional[Cell]],
                      current: Cell) -> Path:
    path: Path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# A* planner
# ---------------------------------------------------------------------------

def astar(cost_map: np.ndarray,
          start: Cell,
          goal: Cell,
          heuristic: str = "octile",
          obstacle_threshold: float = 1e5) -> Tuple[Optional[Path], float]:
    """
    Find the lowest-cost path from *start* to *goal* using A*.

    Parameters
    ----------
    cost_map           : 2-D array, cost per cell.
    start              : (row, col) source cell.
    goal               : (row, col) destination cell.
    heuristic          : "euclidean" | "manhattan" | "octile"
    obstacle_threshold : Cells with cost >= this value are impassable.

    Returns
    -------
    (path, total_cost) or (None, inf) if no path found.
    """
    H, W = cost_map.shape

    if not (_in_bounds(start, H, W) and _in_bounds(goal, H, W)):
        return None, math.inf

    h_fn = {
        "euclidean": _heuristic_euclidean,
        "manhattan": _heuristic_manhattan,
        "octile":    _heuristic_octile,
    }.get(heuristic, _heuristic_octile)

    # g_score[cell] = best known cost from start to cell
    g_score: Dict[Cell, float] = {start: 0.0}
    came_from: Dict[Cell, Optional[Cell]] = {start: None}

    # Priority queue: (f_score, cell)
    open_heap: List[Tuple[float, Cell]] = []
    heapq.heappush(open_heap, (h_fn(start, goal), start))

    closed: set = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            path = _reconstruct_path(came_from, current)
            return path, g_score[current]

        if current in closed:
            continue
        closed.add(current)

        r, c = current
        for dr, dc in _NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)

            if not _in_bounds(nb, H, W):
                continue
            if nb in closed:
                continue
            if cost_map[nr, nc] >= obstacle_threshold:
                continue

            move_dist = _move_cost(dr, dc)
            # Terrain cost is the average of current and neighbour cells
            terrain_cost = (cost_map[r, c] + cost_map[nr, nc]) / 2.0
            tentative_g = g_score[current] + move_dist * terrain_cost

            if tentative_g < g_score.get(nb, math.inf):
                g_score[nb] = tentative_g
                came_from[nb] = current
                f = tentative_g + h_fn(nb, goal)
                heapq.heappush(open_heap, (f, nb))

    return None, math.inf      # No path found


# ---------------------------------------------------------------------------
# Dijkstra planner
# ---------------------------------------------------------------------------

def dijkstra(cost_map: np.ndarray,
             start: Cell,
             goal: Cell,
             obstacle_threshold: float = 1e5) -> Tuple[Optional[Path], float]:
    """
    Find the lowest-cost path from *start* to *goal* using Dijkstra's algorithm.

    Returns
    -------
    (path, total_cost) or (None, inf) if no path found.
    """
    H, W = cost_map.shape

    if not (_in_bounds(start, H, W) and _in_bounds(goal, H, W)):
        return None, math.inf

    dist: Dict[Cell, float] = {start: 0.0}
    came_from: Dict[Cell, Optional[Cell]] = {start: None}
    open_heap: List[Tuple[float, Cell]] = [(0.0, start)]
    closed: set = set()

    while open_heap:
        d, current = heapq.heappop(open_heap)

        if current == goal:
            path = _reconstruct_path(came_from, current)
            return path, dist[current]

        if current in closed:
            continue
        closed.add(current)

        r, c = current
        for dr, dc in _NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)

            if not _in_bounds(nb, H, W):
                continue
            if nb in closed:
                continue
            if cost_map[nr, nc] >= obstacle_threshold:
                continue

            move_dist = _move_cost(dr, dc)
            terrain_cost = (cost_map[r, c] + cost_map[nr, nc]) / 2.0
            new_d = dist[current] + move_dist * terrain_cost

            if new_d < dist.get(nb, math.inf):
                dist[nb] = new_d
                came_from[nb] = current
                heapq.heappush(open_heap, (new_d, nb))

    return None, math.inf


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _in_bounds(cell: Cell, H: int, W: int) -> bool:
    return 0 <= cell[0] < H and 0 <= cell[1] < W


def smooth_path(path: Path, iterations: int = 2) -> Path:
    """
    Apply simple path-smoothing (average of neighbours) to reduce zig-zags.
    Endpoints are kept fixed.
    """
    if len(path) <= 2:
        return path

    arr = np.array(path, dtype=float)
    for _ in range(iterations):
        smoothed = arr.copy()
        for i in range(1, len(arr) - 1):
            smoothed[i] = (arr[i - 1] + arr[i] + arr[i + 1]) / 3.0
        arr = smoothed

    return [(int(round(r)), int(round(c))) for r, c in arr]


# ---------------------------------------------------------------------------
# PathPlanner facade
# ---------------------------------------------------------------------------

class PathPlanner:
    """
    High-level path planner that wraps A* and Dijkstra with pre-processing.

    Accepts a terrain generator's cost map and optionally inflates obstacles
    before planning.
    """

    def __init__(self,
                 cost_map: np.ndarray,
                 algorithm: str = "astar",
                 obstacle_inflation: int = 2,
                 obstacle_threshold: float = 1e5):
        """
        Parameters
        ----------
        cost_map             : Raw per-cell traversal cost.
        algorithm            : "astar" | "dijkstra"
        obstacle_inflation   : Dilation radius to widen obstacle margins (cells).
        obstacle_threshold   : Cost value above which a cell is impassable.
        """
        import cv2 as _cv2
        self.algorithm = algorithm.lower()
        self.obstacle_threshold = obstacle_threshold
        self.raw_cost_map = cost_map.copy()

        # Inflate obstacles using morphological dilation of the obstacle mask
        obs_mask = (cost_map >= obstacle_threshold).astype(np.uint8)
        if obstacle_inflation > 0:
            kernel = _cv2.getStructuringElement(
                _cv2.MORPH_ELLIPSE,
                (2 * obstacle_inflation + 1, 2 * obstacle_inflation + 1),
            )
            inflated_mask = _cv2.dilate(obs_mask, kernel)
        else:
            inflated_mask = obs_mask

        self.cost_map = cost_map.copy()
        self.cost_map[inflated_mask > 0] = obstacle_threshold

        self.last_path: Optional[Path] = None
        self.last_cost: float = math.inf

    # ------------------------------------------------------------------
    def plan(self, start: Cell, goal: Cell,
             smooth: bool = True) -> Tuple[Optional[Path], float]:
        """
        Plan a path from *start* to *goal*.

        Returns
        -------
        (path, cost) — path is None if unreachable.
        """
        if self.algorithm == "astar":
            path, cost = astar(self.cost_map, start, goal,
                               obstacle_threshold=self.obstacle_threshold)
        elif self.algorithm == "dijkstra":
            path, cost = dijkstra(self.cost_map, start, goal,
                                  obstacle_threshold=self.obstacle_threshold)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm!r}")

        if path and smooth:
            path = smooth_path(path)

        self.last_path = path
        self.last_cost = cost
        return path, cost

    # ------------------------------------------------------------------
    def plan_multi_waypoint(self, waypoints: List[Cell],
                            smooth: bool = True) -> Tuple[Path, float]:
        """
        Plan a path through a sequence of waypoints (start → wp1 → wp2 → …).

        Returns
        -------
        (full_path, total_cost)
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints.")

        full_path: Path = []
        total_cost = 0.0

        for i in range(len(waypoints) - 1):
            seg_path, seg_cost = self.plan(waypoints[i], waypoints[i + 1],
                                           smooth=smooth)
            if seg_path is None:
                print(f"[PathPlanner] WARNING: No path from {waypoints[i]} "
                      f"to {waypoints[i + 1]}")
                continue
            # Avoid duplicate junction points
            if full_path and seg_path:
                seg_path = seg_path[1:]
            full_path.extend(seg_path)
            total_cost += seg_cost

        self.last_path = full_path
        self.last_cost = total_cost
        return full_path, total_cost

    # ------------------------------------------------------------------
    def path_statistics(self, path: Path) -> dict:
        """Return length, cost, and straightness metrics for a path."""
        if not path:
            return {}
        n = len(path)
        # Geometric length
        length = sum(
            _heuristic_euclidean(path[i], path[i + 1])
            for i in range(n - 1)
        )
        # Straight-line distance
        straight = _heuristic_euclidean(path[0], path[-1])
        return {
            "num_cells": n,
            "geometric_length": round(length, 2),
            "straight_line_distance": round(straight, 2),
            "tortuosity": round(length / (straight + 1e-9), 3),
            "plan_cost": round(self.last_cost, 3),
        }
