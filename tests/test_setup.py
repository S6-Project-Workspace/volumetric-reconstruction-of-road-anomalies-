"""
Test basic setup and imports.
"""

import pytest
import numpy as np
import cv2
import open3d as o3d
import trimesh
from stereo_vision.utils.config_manager import ConfigManager


def test_opencv_import():
    """Test that OpenCV is properly installed and working."""
    assert cv2.__version__ is not None
    
    # Test basic OpenCV functionality
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (100, 100)


def test_open3d_import():
    """Test that Open3D is properly installed and working."""
    # Create a simple point cloud
    points = np.random.rand(100, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    assert len(pcd.points) == 100


def test_trimesh_import():
    """Test that Trimesh is properly installed and working."""
    # Create a simple mesh
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    assert len(mesh.vertices) == 3
    assert len(mesh.faces) == 1


def test_config_manager():
    """Test that configuration manager works."""
    config = ConfigManager()
    
    # Test getting values
    focal_length = config.get('camera.focal_length')
    assert focal_length == 721.5
    
    # Test setting values
    config.set('camera.focal_length', 800.0)
    assert config.get('camera.focal_length') == 800.0


def test_project_structure():
    """Test that project structure is correctly set up."""
    from stereo_vision import __version__
    assert __version__ == "1.0.0"
    
    # Test that modules can be imported
    from stereo_vision.data_models import CameraParameters
    from stereo_vision.utils.config_manager import ConfigManager
    
    # Test data model creation
    camera_matrix = np.eye(3)
    distortion_coeffs = np.zeros(5)
    params = CameraParameters(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        reprojection_error=0.1,
        image_size=(640, 480)
    )
    
    assert params.reprojection_error == 0.1
    assert params.image_size == (640, 480)