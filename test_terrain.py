"""
rover_model.py
==============
Kinematic and state model for a differential-drive lunar rover.

Tracks position, heading, velocity, odometry, and per-step diagnostics.
Supports both continuous (physics-step) and discrete (grid-cell) movement.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Rover configuration
# ---------------------------------------------------------------------------

@dataclass
class RoverConfig:
    """Physical and operational parameters of the rover."""
    name: str = "LunaRover-1"
    mass_kg: float = 180.0          # Rover mass (kg)
    wheel_radius_m: float = 0.27    # Wheel radius (m)
    track_width_m: float = 1.4      # Distance between left/right wheels (m)
    max_speed_ms: float = 0.3       # Maximum linear speed (m/s)
    max_slope_deg: float = 30.0     # Maximum traversable slope (degrees)
    battery_capacity_wh: float = 1500.0   # Battery capacity (Wh)
    sensor_range_cells: int = 12    # Perception range in grid cells
    cell_size_m: float = 1.0        # Real-world size of one grid cell (m)


# ---------------------------------------------------------------------------
# Per-step state snapshot
# ---------------------------------------------------------------------------

@dataclass
class RoverState:
    """Complete rover state at a single timestep."""
    x: float = 0.0          # Column (grid cell, float)
    y: float = 0.0          # Row (grid cell, float)
    heading_rad: float = 0.0  # Heading in radians (0 = East, CCW positive)
    speed_ms: float = 0.0   # Current linear speed (m/s)
    battery_wh: float = 1500.0  # Remaining battery (Wh)
    odometry_m: float = 0.0    # Cumulative distance travelled (m)
    step: int = 0

    @property
    def position(self) -> Tuple[int, int]:
        """Return (row, col) as integer grid indices."""
        return (int(round(self.y)), int(round(self.x)))

    @property
    def heading_deg(self) -> float:
        return math.degrees(self.heading_rad) % 360.0


# ---------------------------------------------------------------------------
# History record for a full mission
# ---------------------------------------------------------------------------

@dataclass
class MissionLog:
    states: List[RoverState] = field(default_factory=list)
    waypoints_reached: List[Tuple[int, int]] = field(default_factory=list)
    events: List[str] = field(default_factory=list)

    def record(self, state: RoverState) -> None:
        self.states.append(RoverState(**state.__dict__))

    def log_event(self, msg: str) -> None:
        self.events.append(msg)

    @property
    def path_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        xs = np.array([s.x for s in self.states])
        ys = np.array([s.y for s in self.states])
        return xs, ys

    @property
    def battery_history(self) -> np.ndarray:
        return np.array([s.battery_wh for s in self.states])

    @property
    def odometry_history(self) -> np.ndarray:
        return np.array([s.odometry_m for s in self.states])


# ---------------------------------------------------------------------------
# Rover model
# ---------------------------------------------------------------------------

class RoverModel:
    """
    Differential-drive rover kinematics on a discrete grid.

    The grid is treated as a 2-D array [row, col] = [y, x].
    Movement along a pre-computed path is performed step-by-step,
    with physics noise and slope-dependent speed scaling applied.
    """

    def __init__(self, config: Optional[RoverConfig] = None,
                 start: Tuple[int, int] = (5, 5),
                 heightmap: Optional[np.ndarray] = None,
                 slope_map: Optional[np.ndarray] = None,
                 noise_sigma: float = 0.0):
        """
        Parameters
        ----------
        config       : RoverConfig instance (defaults created if None).
        start        : (row, col) starting position on the grid.
        heightmap    : 2-D elevation array for slope-speed scaling.
        slope_map    : Pre-computed slope magnitudes [0,1].
        noise_sigma  : Gaussian position noise per step (cells). 0 = perfect.
        """
        self.config = config or RoverConfig()
        self.heightmap = heightmap
        self.slope_map = slope_map
        self.noise_sigma = noise_sigma

        self.state = RoverState(
            x=float(start[1]),
            y=float(start[0]),
            battery_wh=self.config.battery_capacity_wh,
        )
        self.log = MissionLog()
        self.log.record(self.state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _slope_at(self, row: int, col: int) -> float:
        """Return slope magnitude [0,1] at (row, col), or 0 if unavailable."""
        if self.slope_map is None:
            return 0.0
        r = np.clip(row, 0, self.slope_map.shape[0] - 1)
        c = np.clip(col, 0, self.slope_map.shape[1] - 1)
        return float(self.slope_map[r, c])

    def _elevation_delta(self, r0: int, c0: int, r1: int, c1: int) -> float:
        """Signed elevation change from (r0,c0) to (r1,c1)."""
        if self.heightmap is None:
            return 0.0
        e0 = self.heightmap[
            np.clip(r0, 0, self.heightmap.shape[0] - 1),
            np.clip(c0, 0, self.heightmap.shape[1] - 1),
        ]
        e1 = self.heightmap[
            np.clip(r1, 0, self.heightmap.shape[0] - 1),
            np.clip(c1, 0, self.heightmap.shape[1] - 1),
        ]
        return float(e1 - e0)

    # ------------------------------------------------------------------
    # Single step movement
    # ------------------------------------------------------------------

    def step_to(self, target_row: int, target_col: int,
                energy_cost: float = 0.0) -> bool:
        """
        Move rover one grid step toward *target* cell.

        Parameters
        ----------
        target_row, target_col : Destination cell indices.
        energy_cost            : Pre-computed energy to deduct (Wh).

        Returns
        -------
        True if movement succeeded, False if battery depleted.
        """
        if self.state.battery_wh <= 0:
            self.log.log_event(f"[Step {self.state.step}] Battery depleted — rover halted.")
            return False

        prev_r, prev_c = self.state.position
        dy = target_row - prev_r
        dx = target_col - prev_c

        # Heading update
        if dx != 0 or dy != 0:
            self.state.heading_rad = math.atan2(-dy, dx)  # screen coords (y down)

        # Distance (Euclidean, in cells → metres)
        dist_cells = math.sqrt(dx ** 2 + dy ** 2)
        dist_m = dist_cells * self.config.cell_size_m

        # Speed: reduced on steep slopes
        slope = self._slope_at(target_row, target_col)
        speed_factor = max(0.2, 1.0 - slope * 1.5)
        self.state.speed_ms = self.config.max_speed_ms * speed_factor

        # Add odometry noise
        if self.noise_sigma > 0:
            noise = np.random.normal(0, self.noise_sigma, 2)
            self.state.x = target_col + noise[0]
            self.state.y = target_row + noise[1]
        else:
            self.state.x = float(target_col)
            self.state.y = float(target_row)

        # Update state
        self.state.odometry_m += dist_m
        self.state.battery_wh = max(0.0, self.state.battery_wh - energy_cost)
        self.state.step += 1

        self.log.record(self.state)
        return True

    # ------------------------------------------------------------------
    # Follow a full path
    # ------------------------------------------------------------------

    def follow_path(self, path: List[Tuple[int, int]],
                    energy_per_step: Optional[List[float]] = None) -> bool:
        """
        Execute a sequence of grid cells.

        Parameters
        ----------
        path           : List of (row, col) cells.
        energy_per_step: Optional per-step energy values (Wh). Defaults to 0.5.

        Returns
        -------
        True if rover reached the final waypoint, False if stopped early.
        """
        if not path:
            return False

        for i, (row, col) in enumerate(path):
            ecost = energy_per_step[i] if energy_per_step else 0.5
            ok = self.step_to(row, col, energy_cost=ecost)
            if not ok:
                return False

        self.log.waypoints_reached.append(path[-1])
        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> dict:
        s = self.state
        return {
            "position": s.position,
            "heading_deg": round(s.heading_deg, 1),
            "speed_ms": round(s.speed_ms, 3),
            "battery_wh": round(s.battery_wh, 2),
            "battery_pct": round(100 * s.battery_wh / self.config.battery_capacity_wh, 1),
            "odometry_m": round(s.odometry_m, 2),
            "total_steps": s.step,
        }

    def reset(self, start: Tuple[int, int]) -> None:
        """Reset rover to *start* with full battery."""
        self.state = RoverState(
            x=float(start[1]),
            y=float(start[0]),
            battery_wh=self.config.battery_capacity_wh,
        )
        self.log = MissionLog()
        self.log.record(self.state)
