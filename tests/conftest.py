"""
Pytest configuration and fixtures for stereo vision tests.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from stereo_vision.data_models import CameraParameters, StereoParameters
from stereo_vision.utils.config_manager import ConfigManager


@pytest.fixture
def config_manager():
    """Fixture providing a configuration manager instance."""
    return ConfigManager()


@pytest.fixture
def sample_camera_params():
    """Fixture providing sample camera parameters."""
    camera_matrix = np.array([
        [721.5, 0, 609.5],
        [0, 721.5, 172.8],
        [0, 0, 1]
    ], dtype=np.float32)
    
    distortion_coeffs = np.array([0.1, -0.2, 0.001, 0.002, 0.05], dtype=np.float32)
    
    return CameraParameters(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        reprojection_error=0.08,
        image_size=(1241, 376)
    )


@pytest.fixture
def sample_stereo_params(sample_camera_params):
    """Fixture providing sample stereo parameters."""
    rotation_matrix = np.eye(3, dtype=np.float32)
    translation_vector = np.array([0.54, 0, 0], dtype=np.float32)
    
    # Create Q matrix for reprojection
    baseline = 0.54
    focal_length = 721.5
    cx = 609.5
    cy = 172.8
    
    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, focal_length],
        [0, 0, 1.0/baseline, 0]
    ], dtype=np.float32)
    
    return StereoParameters(
        left_camera=sample_camera_params,
        right_camera=sample_camera_params,
        rotation_matrix=rotation_matrix,
        translation_vector=translation_vector,
        baseline=baseline,
        Q_matrix=Q
    )


@pytest.fixture
def sample_disparity_map():
    """Fixture providing a sample disparity map."""
    # Create a synthetic disparity map with road-like characteristics
    height, width = 376, 1241
    disparity = np.zeros((height, width), dtype=np.float32)
    
    # Create a gradient representing a road surface
    for y in range(height):
        # Road disparity decreases with distance (higher y values)
        road_disparity = 80 - (y / height) * 60  # From 80 to 20 disparity
        disparity[y, :] = road_disparity
    
    # Add some noise
    noise = np.random.normal(0, 2, (height, width))
    disparity += noise
    
    # Create a pothole (higher disparity = closer = depression)
    pothole_y, pothole_x = 250, 600
    pothole_size = 50
    y_indices, x_indices = np.ogrid[:height, :width]
    pothole_mask = ((y_indices - pothole_y)**2 + (x_indices - pothole_x)**2) < pothole_size**2
    disparity[pothole_mask] += 10  # Pothole appears closer (higher disparity)
    
    return disparity


@pytest.fixture
def sample_point_cloud():
    """Fixture providing a sample 3D point cloud."""
    # Generate a synthetic point cloud representing a road surface with anomalies
    n_points = 1000
    
    # Create a planar road surface
    x = np.random.uniform(-5, 5, n_points)
    y = np.random.uniform(-2, 2, n_points)
    z = 0.1 * x + 0.05 * y + 10  # Slight slope
    
    # Add some noise
    z += np.random.normal(0, 0.02, n_points)
    
    # Add a pothole (depression)
    pothole_indices = (x > 1) & (x < 3) & (y > -0.5) & (y < 0.5)
    z[pothole_indices] += 0.1  # Deeper (positive Z is away from camera)
    
    return np.column_stack([x, y, z])


@pytest.fixture
def test_data_dir():
    """Fixture providing path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def synthetic_stereo_pair():
    """Fixture providing synthetic stereo image pair."""
    height, width = 376, 1241
    
    # Create left image with some texture
    left_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    # Add some structured patterns (road markings, etc.)
    cv2.rectangle(left_img, (500, 200), (600, 220), 255, -1)  # Lane marking
    cv2.circle(left_img, (400, 300), 30, 128, -1)  # Manhole cover
    
    # Create right image by shifting features (simulating disparity)
    right_img = np.zeros_like(left_img)
    shift = 50  # Average disparity shift
    right_img[:, shift:] = left_img[:, :-shift]
    
    return left_img, right_img