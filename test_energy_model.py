"""
obstacle_detection.py
=====================
Multi-modal obstacle detection for lunar terrain.

Detection methods:
  1. Slope thresholding        — steep gradients flag impassable terrain
  2. OpenCV morphology         — rock cluster detection via dilation/erosion
  3. Shadow masking            — simulated low-angle solar illumination
  4. AI terrain classification — Random Forest trained on local features
"""

import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.ndimage import generic_filter
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Slope-based detection
# ---------------------------------------------------------------------------

def detect_by_slope(slope_map: np.ndarray,
                    threshold: float = 0.30) -> np.ndarray:
    """
    Return a boolean mask of cells whose slope exceeds *threshold*.

    Parameters
    ----------
    slope_map : 2-D array of normalised slope values [0, 1].
    threshold : Fraction of max slope above which cells are obstacles.

    Returns
    -------
    Boolean obstacle mask (True = obstacle).
    """
    return slope_map > threshold


# ---------------------------------------------------------------------------
# 2. Rock cluster detection via OpenCV morphology
# ---------------------------------------------------------------------------

def detect_rocks_morphology(heightmap: np.ndarray,
                            high_elev_thresh: float = 0.70,
                            kernel_size: int = 3) -> np.ndarray:
    """
    Identify rock clusters using morphological operations on the heightmap.

    High-elevation peaks above *high_elev_thresh* are dilated to represent
    the effective obstacle footprint of protruding boulders.

    Returns
    -------
    Boolean mask — True where rocks detected.
    """
    # Convert to 8-bit image for OpenCV
    img = (heightmap * 255).astype(np.uint8)
    _, peaks = cv2.threshold(img, int(high_elev_thresh * 255), 255,
                             cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    # Dilate to grow obstacle footprint
    dilated = cv2.dilate(peaks, kernel, iterations=2)
    # Erode slightly to remove single-pixel noise
    result = cv2.erode(dilated, kernel, iterations=1)
    return result > 0


# ---------------------------------------------------------------------------
# 3. Shadow masking (simulated illumination)
# ---------------------------------------------------------------------------

def compute_shadow_mask(heightmap: np.ndarray,
                        sun_azimuth_deg: float = 45.0,
                        sun_elevation_deg: float = 10.0,
                        shadow_threshold: float = 0.05) -> np.ndarray:
    """
    Simulate low-angle solar illumination and return a shadow mask.

    A simplified ray-marching approach: for each cell, we march backward
    along the sun direction and check if any upstream cell casts a shadow.

    Parameters
    ----------
    heightmap           : 2-D elevation array [0, 1].
    sun_azimuth_deg     : Sun azimuth (0 = East, 90 = North).
    sun_elevation_deg   : Sun angle above horizon.
    shadow_threshold    : Cells below this illumination are "in shadow".

    Returns
    -------
    Boolean mask — True where in shadow (potential navigation hazard).
    """
    H, W = heightmap.shape
    az_rad = np.radians(sun_azimuth_deg)
    el_rad = np.radians(sun_elevation_deg)

    # Sun direction vector (grid coords)
    sun_dy = -np.sin(az_rad)
    sun_dx = np.cos(az_rad)
    tan_el = np.tan(el_rad)

    illumination = np.ones((H, W), dtype=np.float32)
    step_size = 1.0
    max_steps = max(H, W)

    for row in range(H):
        for col in range(W):
            curr_h = heightmap[row, col]
            # March backward along sun ray
            for k in range(1, max_steps):
                src_row = int(row + k * sun_dy * step_size)
                src_col = int(col - k * sun_dx * step_size)
                if src_row < 0 or src_row >= H or src_col < 0 or src_col >= W:
                    break
                blocker_h = heightmap[src_row, src_col]
                # If the upstream cell is higher, accounting for sun angle
                if blocker_h > curr_h + k * step_size * tan_el * 0.05:
                    illumination[row, col] = 0.0
                    break

    return illumination < shadow_threshold


# ---------------------------------------------------------------------------
# 4. AI terrain classifier
# ---------------------------------------------------------------------------

def _extract_features(heightmap: np.ndarray,
                      slope_map: np.ndarray) -> np.ndarray:
    """
    Build a per-cell feature matrix for the terrain classifier.

    Features per cell (9 total):
      - local elevation
      - slope magnitude
      - local elevation std (3x3 window)
      - local elevation mean (3x3 window)
      - local max elevation (3x3 window)
      - gradient x, gradient y
      - laplacian (curvature proxy)
      - roughness = std / (mean + eps)
    """
    H, W = heightmap.shape

    gy, gx = np.gradient(heightmap)
    laplacian = cv2.Laplacian(heightmap.astype(np.float32), cv2.CV_32F)

    # Window statistics using generic_filter
    local_std = generic_filter(heightmap, np.std, size=3)
    local_mean = generic_filter(heightmap, np.mean, size=3)
    local_max = generic_filter(heightmap, np.max, size=3)
    roughness = local_std / (local_mean + 1e-9)

    features = np.stack([
        heightmap,
        slope_map,
        local_std,
        local_mean,
        local_max,
        gx,
        gy,
        laplacian,
        roughness,
    ], axis=-1)  # (H, W, 9)

    return features.reshape(-1, 9)


class TerrainClassifier:
    """
    Random Forest terrain classifier.

    Classes:
        0 = FLAT          (safe, fast travel)
        1 = SLOPE         (traversable but costly)
        2 = ROCKY         (slow, risky)
        3 = CRATER_RIM    (very risky)
        4 = CRATER_FLOOR  (avoid)
    """

    CLASS_NAMES = {0: "FLAT", 1: "SLOPE", 2: "ROCKY",
                   3: "CRATER_RIM", 4: "CRATER_FLOOR"}

    def __init__(self, n_estimators: int = 100, seed: int = 42):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=12,
                random_state=seed,
                n_jobs=-1,
            )),
        ])
        self.fitted = False

    def _make_labels(self, heightmap: np.ndarray,
                     slope_map: np.ndarray) -> np.ndarray:
        """Generate ground-truth labels from terrain rules."""
        labels = np.zeros(heightmap.shape, dtype=int)
        labels[slope_map > 0.35] = 1
        labels[(slope_map > 0.25) & (heightmap > 0.55)] = 2
        labels[(slope_map > 0.30) & (heightmap > 0.60)] = 3
        labels[heightmap < 0.15] = 4
        return labels.ravel()

    def fit(self, heightmap: np.ndarray, slope_map: np.ndarray) -> "TerrainClassifier":
        """Train the classifier on the given terrain arrays."""
        X = _extract_features(heightmap, slope_map)
        y = self._make_labels(heightmap, slope_map)
        self.pipeline.fit(X, y)
        self.fitted = True
        return self

    def predict(self, heightmap: np.ndarray,
                slope_map: np.ndarray) -> np.ndarray:
        """
        Predict terrain class for every cell.

        Returns
        -------
        2-D integer array of class labels, shape = heightmap.shape.
        """
        if not self.fitted:
            raise RuntimeError("TerrainClassifier must be fitted before predicting.")
        X = _extract_features(heightmap, slope_map)
        preds = self.pipeline.predict(X)
        return preds.reshape(heightmap.shape)

    def predict_proba(self, heightmap: np.ndarray,
                      slope_map: np.ndarray) -> np.ndarray:
        """Return per-class probabilities, shape = (H, W, n_classes)."""
        if not self.fitted:
            raise RuntimeError("TerrainClassifier must be fitted before predicting.")
        X = _extract_features(heightmap, slope_map)
        proba = self.pipeline.predict_proba(X)          # (H*W, n_classes)
        H, W = heightmap.shape
        return proba.reshape(H, W, -1)


# ---------------------------------------------------------------------------
# Combined obstacle map
# ---------------------------------------------------------------------------

class ObstacleDetector:
    """
    Fuses multiple detection methods into a single obstacle probability map.
    """

    def __init__(self,
                 slope_weight: float = 0.40,
                 rock_weight: float = 0.30,
                 shadow_weight: float = 0.10,
                 ai_weight: float = 0.20,
                 slope_threshold: float = 0.30,
                 sun_azimuth_deg: float = 45.0,
                 sun_elevation_deg: float = 12.0):
        self.slope_weight = slope_weight
        self.rock_weight = rock_weight
        self.shadow_weight = shadow_weight
        self.ai_weight = ai_weight
        self.slope_threshold = slope_threshold
        self.sun_azimuth_deg = sun_azimuth_deg
        self.sun_elevation_deg = sun_elevation_deg

        self.classifier = TerrainClassifier()
        self._trained = False

        # Output maps (set after calling detect())
        self.slope_mask: Optional[np.ndarray] = None
        self.rock_mask: Optional[np.ndarray] = None
        self.shadow_mask: Optional[np.ndarray] = None
        self.ai_labels: Optional[np.ndarray] = None
        self.combined_obstacle_map: Optional[np.ndarray] = None
        self.binary_obstacle_map: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def detect(self, heightmap: np.ndarray,
               slope_map: np.ndarray,
               binary_threshold: float = 0.40) -> np.ndarray:
        """
        Run all detectors and fuse results.

        Parameters
        ----------
        heightmap         : 2-D elevation array [0, 1].
        slope_map         : 2-D slope magnitude [0, 1].
        binary_threshold  : Probability above which a cell is an obstacle.

        Returns
        -------
        Binary obstacle mask (True = obstacle), same shape as *heightmap*.
        """
        # Train AI classifier if not yet done
        if not self._trained:
            self.classifier.fit(heightmap, slope_map)
            self._trained = True

        # Individual detection layers
        self.slope_mask = detect_by_slope(slope_map, self.slope_threshold)
        self.rock_mask = detect_rocks_morphology(heightmap)
        self.shadow_mask = compute_shadow_mask(
            heightmap, self.sun_azimuth_deg, self.sun_elevation_deg
        )

        # AI: cells classified as CRATER_RIM (3) or CRATER_FLOOR (4) → obstacle
        ai_preds = self.classifier.predict(heightmap, slope_map)
        ai_hazard = (ai_preds >= 3).astype(float)
        self.ai_labels = ai_preds

        # Weighted fusion → probability map [0, 1]
        total_w = (self.slope_weight + self.rock_weight +
                   self.shadow_weight + self.ai_weight)
        self.combined_obstacle_map = (
            self.slope_weight * self.slope_mask.astype(float) +
            self.rock_weight * self.rock_mask.astype(float) +
            self.shadow_weight * self.shadow_mask.astype(float) +
            self.ai_weight * ai_hazard
        ) / total_w

        self.binary_obstacle_map = self.combined_obstacle_map >= binary_threshold
        return self.binary_obstacle_map

    # ------------------------------------------------------------------
    def get_clearance_map(self) -> np.ndarray:
        """Return 1 - obstacle_probability as a traversability score."""
        if self.combined_obstacle_map is None:
            raise RuntimeError("Call detect() first.")
        return 1.0 - self.combined_obstacle_map
