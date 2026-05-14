"""
visualization.py
================
Publication-quality visualization suite for LunaRover AI.

Renders:
  - 3D terrain surface with rover path overlay
  - 2D heightmap with obstacle and path overlays
  - SLAM occupancy map with frontier markers
  - Energy consumption timeline
  - Terrain classification heatmap
  - Mission summary dashboard (multi-panel)
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401 (registers 3d projection)
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

TERRAIN_CMAP  = "gist_earth"
OBSTACLE_CMAP = "Reds"
PATH_COLOUR   = "#00E5FF"
WAYPOINT_COLOURS = {
    "START":       "#00FF88",
    "SCIENCE":     "#FFD700",
    "CHECKPOINT":  "#FF6B35",
    "RETURN_BASE": "#B388FF",
    "EMERGENCY":   "#FF1744",
}

DARK_BG  = "#0A0E1A"
PANEL_BG = "#111827"
GRID_COL = "#1F2937"
TEXT_COL = "#E5E7EB"


def _apply_dark_theme(fig: plt.Figure, axes) -> None:
    """Apply the LunaRover dark space theme to a figure and its axes."""
    fig.patch.set_facecolor(DARK_BG)
    ax_list = axes if hasattr(axes, "__iter__") else [axes]
    for ax in ax_list:
        if ax is None:
            continue
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        if hasattr(ax, "zaxis"):
            ax.zaxis.label.set_color(TEXT_COL)
            ax.tick_params(axis="z", colors=TEXT_COL, labelsize=7)
        if hasattr(ax, "title"):
            ax.title.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(color=GRID_COL, linewidth=0.4, linestyle="--", alpha=0.5)


# ---------------------------------------------------------------------------
# 1. Terrain heightmap (2-D)
# ---------------------------------------------------------------------------

def plot_terrain(heightmap: np.ndarray,
                 title: str = "Lunar Terrain — Heightmap",
                 save_path: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7))
    _apply_dark_theme(fig, ax)
    im = ax.imshow(heightmap, cmap=TERRAIN_CMAP, origin="upper", interpolation="bilinear")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Normalised Elevation", color=TEXT_COL, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL, fontsize=8)
    ax.set_title(title, color=TEXT_COL, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Column (cells)", fontsize=9)
    ax.set_ylabel("Row (cells)", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# 2. Obstacle map
# ---------------------------------------------------------------------------

def plot_obstacle_map(heightmap: np.ndarray,
                      obstacle_map: np.ndarray,
                      save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _apply_dark_theme(fig, axes)

    axes[0].imshow(heightmap, cmap=TERRAIN_CMAP, origin="upper", interpolation="bilinear")
    axes[0].set_title("Heightmap", color=TEXT_COL, fontsize=11, fontweight="bold")

    axes[1].imshow(heightmap, cmap=TERRAIN_CMAP, origin="upper",
                   interpolation="bilinear", alpha=0.6)
    masked = np.ma.masked_where(~obstacle_map, obstacle_map.astype(float))
    axes[1].imshow(masked, cmap=OBSTACLE_CMAP, origin="upper", alpha=0.75, vmin=0, vmax=1)
    axes[1].set_title("Obstacle Map (red = hazard)", color=TEXT_COL,
                       fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Column (cells)", fontsize=9)
        ax.set_ylabel("Row (cells)", fontsize=9)

    plt.suptitle("Obstacle Detection", color=TEXT_COL, fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# 3. Planned path overlay
# ---------------------------------------------------------------------------

def plot_path(heightmap: np.ndarray,
              cost_map: np.ndarray,
              path: List[Tuple[int, int]],
              waypoints: Optional[list] = None,
              title: str = "A* Planned Path",
              save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _apply_dark_theme(fig, axes)

    # Left: cost map
    cm_disp = np.clip(cost_map, 0, 10)
    axes[0].imshow(cm_disp, cmap="hot", origin="upper", interpolation="nearest")
    axes[0].set_title("Traversal Cost Map", color=TEXT_COL, fontsize=11, fontweight="bold")

    # Right: terrain + path
    axes[1].imshow(heightmap, cmap=TERRAIN_CMAP, origin="upper", interpolation="bilinear")

    if path:
        path_arr = np.array(path)
        axes[1].plot(path_arr[:, 1], path_arr[:, 0],
                     color=PATH_COLOUR, linewidth=1.8, alpha=0.90, zorder=5, label="Path")
        # Start / end markers
        axes[1].scatter(path_arr[0, 1], path_arr[0, 0],
                        c="#00FF88", s=80, zorder=6, marker="^", label="Start")
        axes[1].scatter(path_arr[-1, 1], path_arr[-1, 0],
                        c="#FF4444", s=80, zorder=6, marker="*", label="Goal")

    if waypoints:
        for wp in waypoints:
            col_str = WAYPOINT_COLOURS.get(wp.wp_type.value, "#FFFFFF")
            axes[1].scatter(wp.col, wp.row, c=col_str, s=60, zorder=7,
                            marker="D", edgecolors="white", linewidths=0.5)
            axes[1].annotate(wp.name, (wp.col, wp.row),
                             textcoords="offset points", xytext=(4, 4),
                             fontsize=6, color=TEXT_COL)

    axes[1].legend(loc="upper right", fontsize=8,
                   facecolor=PANEL_BG, edgecolor=GRID_COL,
                   labelcolor=TEXT_COL)
    axes[1].set_title(title, color=TEXT_COL, fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Column (cells)", fontsize=9)
        ax.set_ylabel("Row (cells)", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# 4. SLAM occupancy map
# ---------------------------------------------------------------------------

def plot_slam_map(occupancy_prob: np.ndarray,
                  known_mask: np.ndarray,
                  frontier_cells: Optional[list] = None,
                  rover_path: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                  save_path: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7))
    _apply_dark_theme(fig, ax)

    # Unknown = 0.5 (grey), free < 0.5, occupied > 0.5
    disp = np.full(occupancy_prob.shape, 0.5)
    disp[known_mask] = occupancy_prob[known_mask]

    ax.imshow(disp, cmap="RdYlGn_r", origin="upper",
              vmin=0, vmax=1, interpolation="nearest")

    if frontier_cells:
        fc = np.array(list(frontier_cells))
        if len(fc):
            ax.scatter(fc[:, 1], fc[:, 0], c="#00BFFF",
                       s=6, alpha=0.7, zorder=4, label="Frontier")

    if rover_path:
        xs, ys = rover_path
        ax.plot(xs, ys, color=PATH_COLOUR, linewidth=1.5,
                alpha=0.85, zorder=5, label="Rover path")
        ax.scatter(xs[0], ys[0], c="#00FF88", s=80, zorder=6, marker="^")
        ax.scatter(xs[-1], ys[-1], c="#FF4444", s=80, zorder=6, marker="o")

    ax.legend(loc="upper right", fontsize=8,
              facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    ax.set_title("SLAM Occupancy Grid Map", color=TEXT_COL,
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Column (cells)", fontsize=9)
    ax.set_ylabel("Row (cells)", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# 5. Energy timeline
# ---------------------------------------------------------------------------

def plot_energy_profile(battery_history: np.ndarray,
                         per_step_energy: Optional[List[float]] = None,
                         capacity_wh: float = 1500.0,
                         save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    _apply_dark_theme(fig, axes)

    steps = np.arange(len(battery_history))
    pct   = 100.0 * battery_history / capacity_wh

    # Top: battery level
    axes[0].fill_between(steps, pct, alpha=0.3, color="#00E5FF")
    axes[0].plot(steps, pct, color="#00E5FF", linewidth=1.5)
    axes[0].axhline(20, color="#FF4444", linewidth=1.0, linestyle="--",
                    label="20% warning")
    axes[0].axhline(10, color="#FF1744", linewidth=1.0, linestyle=":",
                    label="10% critical")
    axes[0].set_ylabel("Battery (%)", fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Battery Level Over Mission", color=TEXT_COL,
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8, facecolor=PANEL_BG,
                   edgecolor=GRID_COL, labelcolor=TEXT_COL)

    # Bottom: per-step energy cost
    if per_step_energy and len(per_step_energy) > 0:
        e_steps = np.arange(len(per_step_energy))
        axes[1].bar(e_steps, per_step_energy, color="#FFD700", alpha=0.7, width=1.0)
        axes[1].plot(e_steps,
                     np.convolve(per_step_energy,
                                 np.ones(max(1, len(per_step_energy)//20)) /
                                 max(1, len(per_step_energy)//20),
                                 mode="same"),
                     color="#FF6B35", linewidth=1.5, label="Rolling avg")
        axes[1].set_ylabel("Step Energy (Wh)", fontsize=9)
        axes[1].set_xlabel("Step", fontsize=9)
        axes[1].set_title("Per-Step Energy Consumption", color=TEXT_COL,
                          fontsize=11, fontweight="bold")
        axes[1].legend(fontsize=8, facecolor=PANEL_BG,
                       edgecolor=GRID_COL, labelcolor=TEXT_COL)
    else:
        axes[1].set_xlabel("Step", fontsize=9)
        axes[1].set_title("Per-Step Energy (no data)", color=TEXT_COL, fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# 6. Terrain classification map
# ---------------------------------------------------------------------------

def plot_terrain_classification(terrain_labels: np.ndarray,
                                 save_path: Optional[str] = None) -> plt.Figure:
    CLASS_COLOURS = {
        0: "#2ECC71",   # FLAT — green
        1: "#F39C12",   # SLOPE — orange
        2: "#E74C3C",   # ROCKY — red
        3: "#9B59B6",   # CRATER_RIM — purple
        4: "#1A252F",   # CRATER_FLOOR — near-black
    }
    CLASS_NAMES = {
        0: "Flat", 1: "Slope", 2: "Rocky",
        3: "Crater Rim", 4: "Crater Floor"
    }

    colour_arr = np.zeros((*terrain_labels.shape, 3))
    for cls, hex_col in CLASS_COLOURS.items():
        rgb = mcolors.to_rgb(hex_col)
        mask = terrain_labels == cls
        colour_arr[mask] = rgb

    fig, ax = plt.subplots(figsize=(8, 7))
    _apply_dark_theme(fig, ax)
    ax.imshow(colour_arr, origin="upper")

    patches = [
        mpatches.Patch(color=CLASS_COLOURS[k], label=CLASS_NAMES[k])
        for k in CLASS_COLOURS
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9,
              facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    ax.set_title("AI Terrain Classification", color=TEXT_COL,
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Column (cells)", fontsize=9)
    ax.set_ylabel("Row (cells)", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# 7. 3-D terrain surface
# ---------------------------------------------------------------------------

def plot_3d_terrain(heightmap: np.ndarray,
                    path: Optional[List[Tuple[int, int]]] = None,
                    downsample: int = 2,
                    save_path: Optional[str] = None) -> plt.Figure:
    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PANEL_BG)

    H, W = heightmap.shape
    hmap = heightmap[::downsample, ::downsample]
    rows = np.arange(0, H, downsample)
    cols = np.arange(0, W, downsample)
    C, R = np.meshgrid(cols, rows)

    surf = ax.plot_surface(C, R, hmap, cmap=TERRAIN_CMAP,
                           linewidth=0, antialiased=True, alpha=0.88)
    fig.colorbar(surf, ax=ax, shrink=0.4, aspect=12, pad=0.1,
                 label="Elevation").ax.yaxis.set_tick_params(color=TEXT_COL)

    if path and len(path) > 1:
        pr = np.array([p[0] for p in path])
        pc = np.array([p[1] for p in path])
        # Sample heightmap at path cells
        pr_c = np.clip(pr, 0, H - 1)
        pc_c = np.clip(pc, 0, W - 1)
        ph = heightmap[pr_c, pc_c] + 0.02
        ax.plot(pc, pr, ph, color=PATH_COLOUR, linewidth=2.0, zorder=10)
        ax.scatter([pc[0]], [pr[0]], [ph[0]], c="#00FF88", s=60, zorder=11, marker="^")
        ax.scatter([pc[-1]], [pr[-1]], [ph[-1]], c="#FF4444", s=60, zorder=11, marker="*")

    ax.set_title("3D Lunar Terrain Surface", color=TEXT_COL,
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Column", color=TEXT_COL, fontsize=9)
    ax.set_ylabel("Row", color=TEXT_COL, fontsize=9)
    ax.set_zlabel("Elevation", color=TEXT_COL, fontsize=9)
    ax.tick_params(colors=TEXT_COL, labelsize=7)
    ax.grid(color=GRID_COL, linewidth=0.3, linestyle="--", alpha=0.4)
    ax.view_init(elev=35, azim=-55)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# 8. Mission summary dashboard
# ---------------------------------------------------------------------------

def plot_mission_dashboard(heightmap: np.ndarray,
                            obstacle_map: np.ndarray,
                            path: List[Tuple[int, int]],
                            battery_history: np.ndarray,
                            per_step_energy: List[float],
                            slam_prob: np.ndarray,
                            known_mask: np.ndarray,
                            terrain_labels: np.ndarray,
                            waypoints: Optional[list] = None,
                            mission_report: Optional[dict] = None,
                            capacity_wh: float = 1500.0,
                            save_path: Optional[str] = None) -> plt.Figure:
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

    axes = {
        "terrain":   fig.add_subplot(gs[0, 0]),
        "obstacle":  fig.add_subplot(gs[0, 1]),
        "path":      fig.add_subplot(gs[0, 2:]),
        "slam":      fig.add_subplot(gs[1, 0:2]),
        "classify":  fig.add_subplot(gs[1, 2:]),
        "battery":   fig.add_subplot(gs[2, 0:2]),
        "energy":    fig.add_subplot(gs[2, 2:]),
    }

    _apply_dark_theme(fig, list(axes.values()))

    # — Terrain —
    axes["terrain"].imshow(heightmap, cmap=TERRAIN_CMAP, origin="upper")
    axes["terrain"].set_title("Heightmap", color=TEXT_COL, fontsize=9, fontweight="bold")

    # — Obstacle —
    axes["obstacle"].imshow(heightmap, cmap=TERRAIN_CMAP, origin="upper", alpha=0.6)
    masked = np.ma.masked_where(~obstacle_map, np.ones_like(obstacle_map, float))
    axes["obstacle"].imshow(masked, cmap=OBSTACLE_CMAP, origin="upper", alpha=0.75)
    axes["obstacle"].set_title("Obstacles", color=TEXT_COL, fontsize=9, fontweight="bold")

    # — Path —
    axes["path"].imshow(heightmap, cmap=TERRAIN_CMAP, origin="upper", interpolation="bilinear")
    if path:
        pa = np.array(path)
        axes["path"].plot(pa[:, 1], pa[:, 0],
                          color=PATH_COLOUR, linewidth=1.6, alpha=0.9, zorder=5)
        axes["path"].scatter(pa[0, 1], pa[0, 0], c="#00FF88", s=60, zorder=6, marker="^")
        axes["path"].scatter(pa[-1, 1], pa[-1, 0], c="#FF4444", s=60, zorder=6, marker="*")
    if waypoints:
        for wp in waypoints:
            c = WAYPOINT_COLOURS.get(wp.wp_type.value, "#FFFFFF")
            axes["path"].scatter(wp.col, wp.row, c=c, s=40, zorder=7,
                                  marker="D", edgecolors="white", linewidths=0.4)
    axes["path"].set_title("Planned Mission Path", color=TEXT_COL,
                            fontsize=9, fontweight="bold")

    # — SLAM —
    disp = np.full(slam_prob.shape, 0.5)
    disp[known_mask] = slam_prob[known_mask]
    axes["slam"].imshow(disp, cmap="RdYlGn_r", origin="upper", vmin=0, vmax=1)
    if path:
        pa = np.array(path)
        axes["slam"].plot(pa[:, 1], pa[:, 0], color=PATH_COLOUR,
                          linewidth=1.2, alpha=0.7, zorder=5)
    axes["slam"].set_title("SLAM Occupancy Map", color=TEXT_COL, fontsize=9, fontweight="bold")

    # — Classification —
    CLASS_COLOURS = {0:"#2ECC71", 1:"#F39C12", 2:"#E74C3C", 3:"#9B59B6", 4:"#1A252F"}
    colour_arr = np.zeros((*terrain_labels.shape, 3))
    for cls, hx in CLASS_COLOURS.items():
        colour_arr[terrain_labels == cls] = mcolors.to_rgb(hx)
    axes["classify"].imshow(colour_arr, origin="upper")
    axes["classify"].set_title("Terrain Classification", color=TEXT_COL,
                                fontsize=9, fontweight="bold")

    # — Battery —
    steps = np.arange(len(battery_history))
    pct = 100.0 * battery_history / capacity_wh
    axes["battery"].fill_between(steps, pct, alpha=0.3, color="#00E5FF")
    axes["battery"].plot(steps, pct, color="#00E5FF", linewidth=1.4)
    axes["battery"].axhline(20, color="#FF4444", linewidth=0.8, linestyle="--")
    axes["battery"].set_ylabel("Battery (%)", fontsize=8)
    axes["battery"].set_xlabel("Step", fontsize=8)
    axes["battery"].set_ylim(0, 105)
    axes["battery"].set_title("Battery Level", color=TEXT_COL, fontsize=9, fontweight="bold")

    # — Per-step energy —
    if per_step_energy:
        e_steps = np.arange(len(per_step_energy))
        axes["energy"].bar(e_steps, per_step_energy, color="#FFD700",
                           alpha=0.7, width=1.0)
        axes["energy"].set_xlabel("Step", fontsize=8)
        axes["energy"].set_ylabel("Energy (Wh)", fontsize=8)
    axes["energy"].set_title("Step Energy Cost", color=TEXT_COL, fontsize=9, fontweight="bold")

    # Title
    title_str = "🌕  LunaRover AI — Mission Dashboard"
    if mission_report:
        s = mission_report
        title_str += (f"   |   Status: {s.get('status','?')}   "
                      f"|   WPs: {s.get('waypoints_completed',0)}/{s.get('waypoints_total',0)}   "
                      f"|   Dist: {s.get('total_distance_m',0)} m   "
                      f"|   Science: {s.get('total_science_value',0)} pts")

    fig.suptitle(title_str, color=TEXT_COL, fontsize=13,
                 fontweight="bold", y=0.995)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    return fig


# ---------------------------------------------------------------------------
# 9. Performance metrics bar chart
# ---------------------------------------------------------------------------

def plot_performance_metrics(metrics: dict,
                              save_path: Optional[str] = None) -> plt.Figure:
    labels = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_theme(fig, ax)

    colours = plt.cm.plasma(np.linspace(0.2, 0.9, len(labels)))
    bars = ax.barh(labels, values, color=colours, edgecolor=GRID_COL, height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01 * max(values),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left",
                color=TEXT_COL, fontsize=8)

    ax.set_title("Performance Evaluation Metrics", color=TEXT_COL,
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Value", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    return fig
