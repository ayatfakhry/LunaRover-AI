"""
energy_model.py
===============
Physics-based energy consumption model for a lunar rover.

Models:
  - Gravitational climbing cost (slope-dependent)
  - Rolling resistance on regolith
  - Wheel slip energy loss
  - Base electrical draw (sensors, comms, compute)
  - Solar power regeneration (optional)

All values in SI units (Joules, Watts, metres, kg).
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

LUNAR_GRAVITY     = 1.62   # m/s²   (Moon surface gravity)
REGOLITH_ROLLING  = 0.15   # Rolling resistance coefficient (lunar regolith)
WHEEL_EFFICIENCY  = 0.85   # Drive-train efficiency


# ---------------------------------------------------------------------------
# Rover power budget dataclass
# ---------------------------------------------------------------------------

@dataclass
class PowerBudget:
    """Breakdown of power consumption for a single time step."""
    propulsion_w: float = 0.0       # Drive motors
    climbing_w: float   = 0.0       # Gravity climbing
    slip_loss_w: float  = 0.0       # Wheel slip dissipation
    base_draw_w: float  = 0.0       # Avionics / sensors / comms
    solar_gain_w: float = 0.0       # Solar panel recharge (negative cost)

    @property
    def net_power_w(self) -> float:
        return (self.propulsion_w + self.climbing_w +
                self.slip_loss_w + self.base_draw_w - self.solar_gain_w)

    @property
    def total_draw_w(self) -> float:
        return self.propulsion_w + self.climbing_w + self.slip_loss_w + self.base_draw_w


# ---------------------------------------------------------------------------
# Energy model
# ---------------------------------------------------------------------------

class EnergyModel:
    """
    Estimates energy consumption for each rover traversal step.

    Parameters
    ----------
    mass_kg        : Rover mass.
    wheel_radius_m : Drive wheel radius.
    base_power_w   : Constant electrical draw (avionics, sensors).
    solar_power_w  : Peak solar panel output (0 = no solar).
    cell_size_m    : Physical size of one grid cell (metres).
    """

    def __init__(self,
                 mass_kg: float = 180.0,
                 wheel_radius_m: float = 0.27,
                 base_power_w: float = 45.0,
                 solar_power_w: float = 120.0,
                 cell_size_m: float = 1.0):
        self.mass_kg = mass_kg
        self.wheel_radius_m = wheel_radius_m
        self.base_power_w = base_power_w
        self.solar_power_w = solar_power_w
        self.cell_size_m = cell_size_m

        # Accumulated history
        self._history: List[PowerBudget] = []
        self._total_energy_j: float = 0.0
        self._total_solar_j: float = 0.0

    # ------------------------------------------------------------------
    # Per-step energy calculation
    # ------------------------------------------------------------------

    def compute_step_energy(self,
                             from_cell: Tuple[int, int],
                             to_cell: Tuple[int, int],
                             slope_magnitude: float,
                             speed_ms: float,
                             elevation_delta: float = 0.0,
                             illuminated: bool = True) -> Tuple[float, PowerBudget]:
        """
        Compute energy consumed (Wh) for one traversal step.

        Parameters
        ----------
        from_cell        : (row, col) source cell.
        to_cell          : (row, col) destination cell.
        slope_magnitude  : Normalised slope [0, 1].
        speed_ms         : Rover speed (m/s).
        elevation_delta  : Signed elevation change (normalised units).
        illuminated      : Whether the rover is in sunlight (for solar).

        Returns
        -------
        (energy_wh, PowerBudget)
        """
        dr = to_cell[0] - from_cell[0]
        dc = to_cell[1] - from_cell[1]
        dist_cells = math.sqrt(dr ** 2 + dc ** 2)
        dist_m = dist_cells * self.cell_size_m

        if dist_m < 1e-9 or speed_ms < 1e-9:
            budget = PowerBudget(base_draw_w=self.base_power_w)
            return 0.0, budget

        # Time for this step (s)
        dt_s = dist_m / speed_ms

        # --- Propulsion (rolling resistance) ---
        F_roll = REGOLITH_ROLLING * self.mass_kg * LUNAR_GRAVITY * math.cos(
            math.asin(min(slope_magnitude, 0.99))
        )
        P_roll = F_roll * speed_ms / WHEEL_EFFICIENCY

        # --- Climbing (potential energy) ---
        # Convert normalised elevation delta to approx metres
        delta_h_m = elevation_delta * 50.0   # 50 m ~ full normalised range
        if delta_h_m > 0:
            F_climb = self.mass_kg * LUNAR_GRAVITY * math.sin(
                math.atan2(delta_h_m, dist_m)
            )
            P_climb = F_climb * speed_ms / WHEEL_EFFICIENCY
        else:
            P_climb = 0.0

        # --- Wheel slip loss ---
        slip_factor = slope_magnitude ** 1.5   # Higher on steep terrain
        P_slip = P_roll * slip_factor * 0.20

        # --- Solar gain ---
        P_solar = self.solar_power_w if illuminated else 0.0

        budget = PowerBudget(
            propulsion_w=P_roll,
            climbing_w=P_climb,
            slip_loss_w=P_slip,
            base_draw_w=self.base_power_w,
            solar_gain_w=P_solar,
        )

        # Energy in Wh
        energy_j = budget.net_power_w * dt_s
        energy_wh = max(0.0, energy_j / 3600.0)

        self._history.append(budget)
        self._total_energy_j += max(0.0, energy_j)
        self._total_solar_j  += P_solar * dt_s

        return energy_wh, budget

    # ------------------------------------------------------------------
    # Path-level energy estimation
    # ------------------------------------------------------------------

    def estimate_path_energy(self,
                              path: List[Tuple[int, int]],
                              slope_map: np.ndarray,
                              heightmap: np.ndarray,
                              speed_ms: float = 0.2,
                              illumination_map: Optional[np.ndarray] = None
                              ) -> Tuple[float, List[float]]:
        """
        Estimate total energy for an entire path.

        Returns
        -------
        (total_energy_wh, per_step_energies)
        """
        if len(path) < 2:
            return 0.0, []

        H, W = slope_map.shape
        per_step: List[float] = []

        for i in range(len(path) - 1):
            r0, c0 = path[i]
            r1, c1 = path[i + 1]
            r0c = np.clip(r0, 0, H - 1); c0c = np.clip(c0, 0, W - 1)
            r1c = np.clip(r1, 0, H - 1); c1c = np.clip(c1, 0, W - 1)

            slope = float(slope_map[r1c, c1c])
            elev_delta = float(heightmap[r1c, c1c] - heightmap[r0c, c0c])

            illuminated = True
            if illumination_map is not None:
                illuminated = bool(not illumination_map[r1c, c1c])

            # Slope-adjusted speed
            speed_factor = max(0.2, 1.0 - slope * 1.5)
            step_speed = speed_ms * speed_factor

            e_wh, _ = self.compute_step_energy(
                (r0, c0), (r1, c1),
                slope_magnitude=slope,
                speed_ms=step_speed,
                elevation_delta=elev_delta,
                illuminated=illuminated,
            )
            per_step.append(e_wh)

        return sum(per_step), per_step

    # ------------------------------------------------------------------
    # Battery range estimator
    # ------------------------------------------------------------------

    def max_range_cells(self,
                        battery_wh: float,
                        avg_slope: float = 0.15,
                        speed_ms: float = 0.2) -> float:
        """
        Estimate maximum travel distance (in cells) given current battery.
        Uses average flat-terrain energy per cell as baseline.
        """
        dummy_from = (0, 0)
        dummy_to   = (1, 0)
        e_per_cell, _ = self.compute_step_energy(
            dummy_from, dummy_to,
            slope_magnitude=avg_slope,
            speed_ms=speed_ms,
        )
        if e_per_cell < 1e-9:
            return float("inf")
        return battery_wh / e_per_cell

    # ------------------------------------------------------------------
    # History & diagnostics
    # ------------------------------------------------------------------

    @property
    def total_energy_wh(self) -> float:
        return self._total_energy_j / 3600.0

    @property
    def total_solar_wh(self) -> float:
        return self._total_solar_j / 3600.0

    def power_history(self) -> np.ndarray:
        """Return array of net power draw (W) per recorded step."""
        return np.array([b.net_power_w for b in self._history])

    def energy_breakdown(self) -> dict:
        """Aggregate power budget across all steps."""
        if not self._history:
            return {}
        prop  = sum(b.propulsion_w for b in self._history)
        climb = sum(b.climbing_w   for b in self._history)
        slip  = sum(b.slip_loss_w  for b in self._history)
        base  = sum(b.base_draw_w  for b in self._history)
        solar = sum(b.solar_gain_w for b in self._history)
        total = prop + climb + slip + base
        return {
            "propulsion_pct" : round(100 * prop  / (total + 1e-9), 1),
            "climbing_pct"   : round(100 * climb / (total + 1e-9), 1),
            "slip_loss_pct"  : round(100 * slip  / (total + 1e-9), 1),
            "base_draw_pct"  : round(100 * base  / (total + 1e-9), 1),
            "solar_offset_wh": round(self.total_solar_wh, 2),
            "net_energy_wh"  : round(self.total_energy_wh, 2),
        }

    def reset(self) -> None:
        self._history.clear()
        self._total_energy_j = 0.0
        self._total_solar_j  = 0.0
