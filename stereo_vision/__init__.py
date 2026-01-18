"""
Advanced Stereo Vision Pipeline for Road Anomaly Detection

A state-of-the-art volumetric reconstruction system for detecting and quantifying
road surface anomalies (potholes and humps) using stereo vision.

This package implements:
- CharuCo-based camera calibration for sub-millimeter accuracy
- Advanced disparity estimation with SGBM, LRC checking, and WLS filtering
- V-Disparity ground plane detection using Hough transforms
- 3D point cloud processing with statistical outlier removal
- Watertight mesh generation using Alpha Shapes
- Precise volume calculation using signed tetrahedron integration
"""

__version__ = "1.0.0"
__author__ = "Advanced Stereo Vision Team"

from .calibration import CharuCoCalibrator, StereoCalibrator, CalibrationValidator
from .disparity import SGBMEstimator, LRCValidator, WLSFilter
from .reconstruction import VDisparityGenerator, GroundPlaneModel, PointCloudGenerator, OutlierRemover
from .volume import AlphaShapeGenerator, MeshCapper, VolumeCalculator
from .data_models import (
    CameraParameters, StereoParameters, DisparityResult, 
    GroundPlaneResult, VolumeResult, RoadAnomaly
)

__all__ = [
    # Calibration
    'CharuCoCalibrator', 'StereoCalibrator', 'CalibrationValidator',
    # Disparity
    'SGBMEstimator', 'LRCValidator', 'WLSFilter',
    # Reconstruction
    'VDisparityGenerator', 'GroundPlaneModel', 'PointCloudGenerator', 'OutlierRemover',
    # Volume
    'AlphaShapeGenerator', 'MeshCapper', 'VolumeCalculator',
    # Data Models
    'CameraParameters', 'StereoParameters', 'DisparityResult',
    'GroundPlaneResult', 'VolumeResult', 'RoadAnomaly'
]