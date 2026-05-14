"""
mission_planner.py
==================
Mission planning and waypoint scheduling for the lunar rover.

Supports:
  - Priority-based waypoint ordering
  - Science target classification
  - Dynamic re-routing on obstacle discovery
  - Mission status tracking and reporting
"""

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class WaypointType(Enum):
    START         = "START"
    SCIENCE       = "SCIENCE"
    CHECKPOINT    = "CHECKPOINT"
    RETURN_BASE   = "RETURN_BASE"
    EMERGENCY     = "EMERGENCY"


class MissionStatus(Enum):
    IDLE       = "IDLE"
    ACTIVE     = "ACTIVE"
    PAUSED     = "PAUSED"
    COMPLETED  = "COMPLETED"
    ABORTED    = "ABORTED"


# ---------------------------------------------------------------------------
# Waypoint
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    row: int
    col: int
    name: str = "WP"
    wp_type: WaypointType = WaypointType.CHECKPOINT
    priority: int = 5               # 1 = highest, 10 = lowest
    science_value: float = 0.0      # Science yield (arbitrary units)
    min_battery_pct: float = 10.0   # Min battery % required to visit
    visited: bool = False
    visit_time: Optional[float] = None

    @property
    def position(self) -> Tuple[int, int]:
        return (self.row, self.col)

    def mark_visited(self) -> None:
        self.visited = True
        self.visit_time = time.time()

    def distance_to(self, other: "Waypoint") -> float:
        return math.sqrt((self.row - other.row) ** 2 +
                         (self.col - other.col) ** 2)


# ---------------------------------------------------------------------------
# Mission Planner
# ---------------------------------------------------------------------------

class MissionPlanner:
    """
    Schedules and manages a multi-waypoint rover mission.

    The planner:
      1. Accepts a list of Waypoint objects.
      2. Orders them by priority (then by science value).
      3. Checks battery feasibility before committing to each leg.
      4. Supports dynamic re-insertion of skipped waypoints.
      5. Generates a mission report at completion.
    """

    def __init__(self,
                 rover_start: Tuple[int, int],
                 battery_capacity_wh: float = 1500.0,
                 energy_per_cell: float = 0.5,
                 grid_size: int = 128):
        self.start_pos = rover_start
        self.battery_capacity_wh = battery_capacity_wh
        self.energy_per_cell = energy_per_cell
        self.grid_size = grid_size

        self.waypoints: List[Waypoint] = []
        self.queue: List[Waypoint] = []
        self.completed: List[Waypoint] = []
        self.skipped: List[Waypoint] = []

        self.status: MissionStatus = MissionStatus.IDLE
        self.current_waypoint_idx: int = 0
        self.mission_start_time: Optional[float] = None
        self.mission_end_time: Optional[float] = None
        self.total_distance_m: float = 0.0
        self.total_energy_used_wh: float = 0.0

        # Add implicit start waypoint
        self._start_wp = Waypoint(
            row=rover_start[0], col=rover_start[1],
            name="BASE", wp_type=WaypointType.START, priority=0
        )

    # ------------------------------------------------------------------
    # Waypoint management
    # ------------------------------------------------------------------

    def add_waypoint(self, waypoint: Waypoint) -> None:
        self.waypoints.append(waypoint)

    def add_science_target(self, row: int, col: int, name: str,
                           priority: int = 3,
                           science_value: float = 10.0) -> Waypoint:
        wp = Waypoint(row=row, col=col, name=name,
                      wp_type=WaypointType.SCIENCE,
                      priority=priority,
                      science_value=science_value)
        self.add_waypoint(wp)
        return wp

    def add_checkpoint(self, row: int, col: int, name: str,
                       priority: int = 5) -> Waypoint:
        wp = Waypoint(row=row, col=col, name=name,
                      wp_type=WaypointType.CHECKPOINT,
                      priority=priority)
        self.add_waypoint(wp)
        return wp

    def set_return_base(self) -> Waypoint:
        wp = Waypoint(
            row=self.start_pos[0], col=self.start_pos[1],
            name="RETURN_BASE", wp_type=WaypointType.RETURN_BASE,
            priority=9
        )
        self.add_waypoint(wp)
        return wp

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def build_queue(self, strategy: str = "priority") -> List[Waypoint]:
        """
        Order waypoints into a mission queue.

        Parameters
        ----------
        strategy : "priority"  — sort by priority then science value
                   "nearest"   — greedy nearest-neighbour from start
                   "science"   — maximise science value first
        """
        wps = [w for w in self.waypoints if not w.visited]

        if strategy == "priority":
            self.queue = sorted(wps, key=lambda w: (w.priority, -w.science_value))

        elif strategy == "nearest":
            remaining = wps.copy()
            ordered: List[Waypoint] = []
            cur = self._start_wp
            while remaining:
                nearest = min(remaining, key=lambda w: cur.distance_to(w))
                ordered.append(nearest)
                remaining.remove(nearest)
                cur = nearest
            self.queue = ordered

        elif strategy == "science":
            self.queue = sorted(wps, key=lambda w: -w.science_value)

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        return self.queue

    # ------------------------------------------------------------------
    # Mission execution interface
    # ------------------------------------------------------------------

    def start_mission(self) -> None:
        if not self.queue:
            self.build_queue()
        self.status = MissionStatus.ACTIVE
        self.mission_start_time = time.time()
        self.current_waypoint_idx = 0

    def next_waypoint(self, current_battery_wh: float) -> Optional[Waypoint]:
        """
        Return the next waypoint the rover should head to.

        Skips waypoints that cannot be reached given current battery level.
        Returns None when the queue is exhausted.
        """
        if self.status != MissionStatus.ACTIVE:
            return None

        while self.current_waypoint_idx < len(self.queue):
            wp = self.queue[self.current_waypoint_idx]
            self.current_waypoint_idx += 1

            battery_pct = 100.0 * current_battery_wh / self.battery_capacity_wh
            if battery_pct >= wp.min_battery_pct:
                return wp
            else:
                self.skipped.append(wp)
                print(f"[MissionPlanner] Skipping {wp.name}: "
                      f"battery {battery_pct:.1f}% < required {wp.min_battery_pct}%")

        # Queue exhausted
        self.complete_mission()
        return None

    def mark_waypoint_reached(self, waypoint: Waypoint,
                               distance_m: float,
                               energy_used_wh: float) -> None:
        waypoint.mark_visited()
        self.completed.append(waypoint)
        self.total_distance_m += distance_m
        self.total_energy_used_wh += energy_used_wh
        print(f"[MissionPlanner] ✓ Reached {waypoint.name} "
              f"({waypoint.wp_type.value}) | "
              f"dist={distance_m:.1f}m | energy={energy_used_wh:.2f}Wh")

    def abort_mission(self, reason: str = "Unknown") -> None:
        self.status = MissionStatus.ABORTED
        self.mission_end_time = time.time()
        print(f"[MissionPlanner] ✗ Mission ABORTED: {reason}")

    def complete_mission(self) -> None:
        self.status = MissionStatus.COMPLETED
        self.mission_end_time = time.time()
        print(f"[MissionPlanner] ✓ Mission COMPLETED — "
              f"{len(self.completed)} waypoints reached.")

    # ------------------------------------------------------------------
    # Feasibility check
    # ------------------------------------------------------------------

    def estimate_energy_to_waypoint(self,
                                     from_pos: Tuple[int, int],
                                     waypoint: Waypoint,
                                     slope_map: Optional[np.ndarray] = None) -> float:
        """
        Rough energy estimate for travelling from *from_pos* to *waypoint*.
        Uses straight-line distance and average slope penalty.
        """
        dist_cells = math.sqrt(
            (waypoint.row - from_pos[0]) ** 2 +
            (waypoint.col - from_pos[1]) ** 2
        )
        slope_factor = 1.0
        if slope_map is not None:
            # Sample slope along direct line
            n_samples = max(2, int(dist_cells))
            rows = np.linspace(from_pos[0], waypoint.row, n_samples, dtype=int)
            cols = np.linspace(from_pos[1], waypoint.col, n_samples, dtype=int)
            rows = np.clip(rows, 0, slope_map.shape[0] - 1)
            cols = np.clip(cols, 0, slope_map.shape[1] - 1)
            avg_slope = float(slope_map[rows, cols].mean())
            slope_factor = 1.0 + 2.0 * avg_slope

        return dist_cells * self.energy_per_cell * slope_factor

    def is_mission_feasible(self, current_battery_wh: float,
                             current_pos: Tuple[int, int],
                             slope_map: Optional[np.ndarray] = None) -> bool:
        """
        Return True if there is enough battery to complete the remaining queue.
        """
        remaining_energy = current_battery_wh
        pos = current_pos

        for wp in self.queue[self.current_waypoint_idx:]:
            needed = self.estimate_energy_to_waypoint(pos, wp, slope_map)
            remaining_energy -= needed
            if remaining_energy < 0:
                return False
            pos = wp.position

        return True

    # ------------------------------------------------------------------
    # Auto-generate waypoints from terrain
    # ------------------------------------------------------------------

    @classmethod
    def generate_science_targets(cls,
                                  heightmap: np.ndarray,
                                  slope_map: np.ndarray,
                                  n_targets: int = 6,
                                  seed: int = 42) -> List[Waypoint]:
        """
        Automatically select science targets from interesting terrain features
        (crater rims, moderate slope areas, flat plains).
        """
        rng = np.random.default_rng(seed)
        H, W = heightmap.shape
        targets: List[Waypoint] = []

        # Feature: crater rim candidates (high elevation + moderate slope)
        rim_mask = (heightmap > 0.65) & (slope_map > 0.20) & (slope_map < 0.38)
        # Feature: flat plain candidates (low slope, mid elevation)
        plain_mask = (slope_map < 0.10) & (heightmap > 0.3) & (heightmap < 0.6)

        def _sample_from_mask(mask: np.ndarray, n: int,
                               wp_type: str, base_priority: int,
                               science_val: float) -> List[Waypoint]:
            rows, cols = np.where(mask)
            if len(rows) == 0:
                return []
            idxs = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
            wps = []
            for k, i in enumerate(idxs):
                wps.append(Waypoint(
                    row=int(rows[i]), col=int(cols[i]),
                    name=f"{wp_type}-{k+1}",
                    wp_type=WaypointType.SCIENCE,
                    priority=base_priority,
                    science_value=science_val + rng.uniform(-2, 2),
                ))
            return wps

        n_rim   = n_targets // 2
        n_plain = n_targets - n_rim
        targets += _sample_from_mask(rim_mask,   n_rim,   "RIM",   2, 15.0)
        targets += _sample_from_mask(plain_mask, n_plain, "PLAIN", 4, 8.0)
        return targets

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def mission_report(self) -> Dict:
        duration = None
        if self.mission_start_time and self.mission_end_time:
            duration = round(self.mission_end_time - self.mission_start_time, 2)

        total_science = sum(w.science_value for w in self.completed)

        return {
            "status": self.status.value,
            "waypoints_total": len(self.waypoints),
            "waypoints_completed": len(self.completed),
            "waypoints_skipped": len(self.skipped),
            "total_distance_m": round(self.total_distance_m, 2),
            "total_energy_wh": round(self.total_energy_used_wh, 2),
            "total_science_value": round(total_science, 2),
            "mission_duration_s": duration,
            "completed_names": [w.name for w in self.completed],
            "skipped_names": [w.name for w in self.skipped],
        }

    def print_report(self) -> None:
        r = self.mission_report()
        print("\n" + "=" * 55)
        print("          LUNAROVER MISSION REPORT")
        print("=" * 55)
        print(f"  Status            : {r['status']}")
        print(f"  Waypoints Total   : {r['waypoints_total']}")
        print(f"  Completed         : {r['waypoints_completed']}")
        print(f"  Skipped           : {r['waypoints_skipped']}")
        print(f"  Total Distance    : {r['total_distance_m']} m")
        print(f"  Energy Used       : {r['total_energy_wh']} Wh")
        print(f"  Science Value     : {r['total_science_value']} pts")
        if r['mission_duration_s']:
            print(f"  Duration          : {r['mission_duration_s']} s")
        print(f"  Completed WPs     : {', '.join(r['completed_names']) or 'None'}")
        print(f"  Skipped WPs       : {', '.join(r['skipped_names']) or 'None'}")
        print("=" * 55 + "\n")
