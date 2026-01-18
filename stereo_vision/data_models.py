"""
Data Models for Stereo Vision Pipeline

Defines all data structures used throughout the system.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np


@dataclass
class CameraParameters:
    """Camera intrinsic parameters and calibration quality metrics."""
    camera_matrix: np.ndarray  # 3x3 intrinsic matrix
    distortion_coeffs: np.ndarray  # 5x1 distortion coefficients
    reprojection_error: float  # RMS reprojection error
    image_size: Tuple[int, int]  # (width, height)


@dataclass
class StereoParameters:
    """Stereo camera system parameters."""
    left_camera: CameraParameters
    right_camera: CameraParameters
    rotation_matrix: np.ndarray  # 3x3 rotation between cameras
    translation_vector: np.ndarray  # 3x1 translation vector
    baseline: float  # Distance between camera centers
    Q_matrix: np.ndarray  # 4x4 reprojection matrix


@dataclass
class RectificationMaps:
    """Stereo rectification mapping data."""
    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray
    roi_left: Tuple[int, int, int, int]  # (x, y, width, height)
    roi_right: Tuple[int, int, int, int]


@dataclass
class DisparityResult:
    """Results from disparity computation."""
    disparity_map: np.ndarray
    validity_mask: np.ndarray
    lrc_error_rate: float
    processing_time: float


@dataclass
class GroundPlaneResult:
    """Results from ground plane detection."""
    plane_parameters: np.ndarray  # [a, b, c] for Z = aX + bY + c
    v_disparity_image: np.ndarray
    hough_line_params: Tuple[float, float]  # slope, intercept
    inlier_ratio: float


@dataclass
class VolumeResult:
    """Results from volume calculation."""
    volume_cubic_meters: float
    volume_liters: float
    volume_cubic_cm: float
    mesh_vertices: int
    mesh_faces: int
    is_watertight: bool
    calculation_method: str
    uncertainty_estimate: float


@dataclass
class RoadAnomaly:
    """Detected road surface anomaly."""
    anomaly_type: str  # "pothole" or "hump"
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    point_cloud: np.ndarray  # Nx3 array of 3D points
    volume_result: VolumeResult
    depth_statistics: Dict[str, float]  # min, max, mean, std depth
    area_square_meters: float


@dataclass
class ProcessingMetrics:
    """System performance and quality metrics."""
    lrc_error_rate: float
    planarity_rmse: float
    temporal_stability: float
    calibration_quality: float
    processing_time_total: float


@dataclass
class BatchResults:
    """Results from batch processing multiple image pairs."""
    anomalies: List[RoadAnomaly]
    metrics: ProcessingMetrics
    summary_statistics: Dict[str, float]
    processed_pairs: int
    failed_pairs: int