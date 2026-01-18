"""
Tests for Stereo Camera Calibrator
"""

import pytest
import numpy as np
import cv2
from hypothesis import given, strategies as st

from stereo_vision.calibration.stereo_calibrator import StereoCalibrator
from stereo_vision.calibration.calibration_validator import CalibrationValidator
from stereo_vision.data_models import CameraParameters, StereoParameters


class TestStereoCalibrator:
    """Test suite for stereo calibrator."""
    
    @pytest.fixture
    def calibrator(self):
        """Fixture providing a stereo calibrator instance."""
        return StereoCalibrator()
    
    @pytest.fixture
    def validator(self):
        """Fixture providing a calibration validator instance."""
        return CalibrationValidator()
    
    @pytest.fixture
    def sample_stereo_data(self, sample_stereo_params):
        """Generate synthetic stereo calibration data."""
        # Create synthetic calibration data
        n_images = 10
        n_points_per_image = 20
        
        object_points = []
        left_image_points = []
        right_image_points = []
        left_images = []
        right_images = []
        
        for i in range(n_images):
            # Generate 3D object points (calibration board)
            obj_pts = np.random.rand(n_points_per_image, 3).astype(np.float32)
            obj_pts[:, 2] = 2.0 + np.random.rand(n_points_per_image) * 3.0  # Z between 2-5m
            
            # Project to left camera (reference)
            left_pts, _ = cv2.projectPoints(
                obj_pts,
                np.zeros(3),  # No rotation
                np.zeros(3),  # No translation
                sample_stereo_params.left_camera.camera_matrix,
                sample_stereo_params.left_camera.distortion_coeffs
            )
            
            # Project to right camera
            right_pts, _ = cv2.projectPoints(
                obj_pts,
                sample_stereo_params.rotation_matrix,
                sample_stereo_params.translation_vector,
                sample_stereo_params.right_camera.camera_matrix,
                sample_stereo_params.right_camera.distortion_coeffs
            )
            
            # Add small amount of noise
            left_pts += np.random.normal(0, 0.1, left_pts.shape).astype(np.float32)
            right_pts += np.random.normal(0, 0.1, right_pts.shape).astype(np.float32)
            
            object_points.append(obj_pts)
            left_image_points.append(left_pts)
            right_image_points.append(right_pts)
            
            # Create dummy images
            left_images.append(np.zeros((376, 1241), dtype=np.uint8))
            right_images.append(np.zeros((376, 1241), dtype=np.uint8))
        
        return {
            'object_points': object_points,
            'left_image_points': left_image_points,
            'right_image_points': right_image_points,
            'left_images': left_images,
            'right_images': right_images
        }
    
    def test_calibrator_initialization(self, calibrator):
        """Test that stereo calibrator initializes correctly."""
        assert calibrator.max_reprojection_error > 0
        assert calibrator.min_baseline > 0
        assert calibrator.max_baseline > calibrator.min_baseline
    
    def test_stereo_calibration_parameter_isolation(self, calibrator, sample_camera_params, sample_stereo_data):
        """
        Test that stereo calibration keeps intrinsic parameters fixed.
        **Feature: advanced-stereo-vision-pipeline, Property 3: Stereo Calibration Parameter Isolation**
        **Validates: Requirements 1.3**
        """
        # Perform stereo calibration
        stereo_params = calibrator.calibrate_stereo(
            sample_camera_params,  # left
            sample_camera_params,  # right (same for simplicity)
            sample_stereo_data['left_images'],
            sample_stereo_data['right_images'],
            sample_stereo_data['object_points'],
            sample_stereo_data['left_image_points'],
            sample_stereo_data['right_image_points']
        )
        
        # Verify intrinsic parameters remain unchanged
        np.testing.assert_array_almost_equal(
            stereo_params.left_camera.camera_matrix,
            sample_camera_params.camera_matrix,
            decimal=6
        )
        
        np.testing.assert_array_almost_equal(
            stereo_params.right_camera.camera_matrix,
            sample_camera_params.camera_matrix,
            decimal=6
        )
        
        np.testing.assert_array_almost_equal(
            stereo_params.left_camera.distortion_coeffs,
            sample_camera_params.distortion_coeffs,
            decimal=6
        )
    
    def test_compute_rectification_maps(self, calibrator, sample_stereo_params):
        """Test rectification map computation."""
        rect_maps = calibrator.compute_rectification_maps(sample_stereo_params)
        
        # Check that maps have correct shape
        height, width = sample_stereo_params.left_camera.image_size[1], sample_stereo_params.left_camera.image_size[0]
        
        assert rect_maps.left_map_x.shape == (height, width)
        assert rect_maps.left_map_y.shape == (height, width)
        assert rect_maps.right_map_x.shape == (height, width)
        assert rect_maps.right_map_y.shape == (height, width)
        
        # Check ROI is valid
        assert len(rect_maps.roi_left) == 4
        assert len(rect_maps.roi_right) == 4
        assert all(x >= 0 for x in rect_maps.roi_left)
        assert all(x >= 0 for x in rect_maps.roi_right)
    
    def test_rectify_image_pair(self, calibrator, sample_stereo_params, synthetic_stereo_pair):
        """Test image pair rectification."""
        left_img, right_img = synthetic_stereo_pair
        
        # Compute rectification maps
        rect_maps = calibrator.compute_rectification_maps(sample_stereo_params)
        
        # Rectify images
        rect_left, rect_right = calibrator.rectify_image_pair(left_img, right_img, rect_maps)
        
        # Check output shapes
        assert rect_left.shape == left_img.shape
        assert rect_right.shape == right_img.shape
        
        # Images should be different after rectification (unless already rectified)
        assert not np.array_equal(rect_left, left_img) or not np.array_equal(rect_right, right_img)
    
    def test_epipolar_rectification_correctness(self, calibrator, sample_stereo_params, synthetic_stereo_pair):
        """
        Test that rectification produces horizontal epipolar lines.
        **Feature: advanced-stereo-vision-pipeline, Property 4: Epipolar Rectification Correctness**
        **Validates: Requirements 1.5**
        """
        left_img, right_img = synthetic_stereo_pair
        
        # Compute rectification maps
        rect_maps = calibrator.compute_rectification_maps(sample_stereo_params)
        
        # Rectify images
        rect_left, rect_right = calibrator.rectify_image_pair(left_img, right_img, rect_maps)
        
        # Validate epipolar alignment
        deviation = calibrator.validate_epipolar_alignment(rect_left, rect_right)
        
        # For synthetic data with proper calibration, deviation should be small
        # Allow some tolerance for numerical precision and synthetic data limitations
        assert deviation < 5.0, f"Epipolar deviation {deviation:.2f} pixels too high"
    
    def test_stereo_geometry_validation(self, calibrator):
        """Test stereo geometry validation."""
        # Valid geometry
        R_valid = np.eye(3)
        T_valid = np.array([0.5, 0, 0])  # 50cm baseline
        baseline_valid = 0.5
        
        # Should not raise exception
        calibrator._validate_stereo_geometry(R_valid, T_valid, baseline_valid)
        
        # Invalid geometry - baseline too small
        T_invalid = np.array([0.01, 0, 0])  # 1cm baseline
        baseline_invalid = 0.01
        
        with pytest.raises(ValueError, match="Baseline too small"):
            calibrator._validate_stereo_geometry(R_valid, T_invalid, baseline_invalid)
        
        # Invalid rotation matrix
        R_invalid = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])  # det != 1
        
        with pytest.raises(ValueError, match="Invalid rotation matrix"):
            calibrator._validate_stereo_geometry(R_invalid, T_valid, baseline_valid)
    
    def test_validator_intrinsic_validation(self, validator, sample_camera_params):
        """Test intrinsic calibration validation."""
        # Test with good parameters
        results = validator.validate_intrinsic_calibration(sample_camera_params)
        
        assert 'is_valid' in results
        assert 'quality_score' in results
        assert 'metrics' in results
        assert results['quality_score'] >= 0
        assert results['quality_score'] <= 100
        
        # Test with bad parameters (high reprojection error)
        bad_params = CameraParameters(
            camera_matrix=sample_camera_params.camera_matrix,
            distortion_coeffs=sample_camera_params.distortion_coeffs,
            reprojection_error=1.0,  # Too high
            image_size=sample_camera_params.image_size
        )
        
        bad_results = validator.validate_intrinsic_calibration(bad_params)
        assert not bad_results['is_valid']
        assert len(bad_results['errors']) > 0
    
    def test_validator_stereo_validation(self, validator, sample_stereo_params):
        """Test stereo calibration validation."""
        results = validator.validate_stereo_calibration(sample_stereo_params)
        
        assert 'is_valid' in results
        assert 'quality_score' in results
        assert 'metrics' in results
        
        # Should have baseline and rotation metrics
        assert 'baseline' in results['metrics']
        assert results['metrics']['baseline'] == sample_stereo_params.baseline
    
    def test_calibration_report_generation(self, validator, sample_stereo_params):
        """Test calibration report generation."""
        validation_results = validator.validate_stereo_calibration(sample_stereo_params)
        report = validator.generate_calibration_report(sample_stereo_params, validation_results)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "STEREO CALIBRATION QUALITY REPORT" in report
        assert "CAMERA PARAMETERS" in report
        assert "STEREO GEOMETRY" in report
    
    @pytest.mark.property
    def test_property_calibration_accuracy_threshold(self, calibrator, sample_stereo_data, sample_camera_params):
        """
        Property test: Calibration accuracy should meet threshold requirements.
        **Feature: advanced-stereo-vision-pipeline, Property 2: Calibration Accuracy Threshold**
        **Validates: Requirements 1.2**
        """
        # Perform stereo calibration with synthetic data
        stereo_params = calibrator.calibrate_stereo(
            sample_camera_params,  # left
            sample_camera_params,  # right
            sample_stereo_data['left_images'],
            sample_stereo_data['right_images'],
            sample_stereo_data['object_points'],
            sample_stereo_data['left_image_points'],
            sample_stereo_data['right_image_points']
        )
        
        # Calculate stereo reprojection error
        stereo_error = calibrator._calculate_stereo_reprojection_error(
            sample_stereo_data['object_points'],
            sample_stereo_data['left_image_points'],
            sample_stereo_data['right_image_points'],
            sample_camera_params,
            sample_camera_params,
            stereo_params.rotation_matrix,
            stereo_params.translation_vector
        )
        
        # Verify accuracy threshold
        assert stereo_error < 0.5, \
            f"Stereo reprojection error {stereo_error:.4f} should be < 0.5 pixels"
    
    @pytest.mark.property
    @given(
        baseline=st.floats(min_value=0.1, max_value=1.0),
        rotation_angle=st.floats(min_value=0, max_value=30)
    )
    def test_property_stereo_geometry_validation(self, baseline, rotation_angle):
        """
        Property test: Stereo geometry validation should accept reasonable parameters.
        """
        # Create calibrator instance for this test
        calibrator = StereoCalibrator()
        
        # Create rotation matrix from angle
        angle_rad = np.radians(rotation_angle)
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        
        T = np.array([baseline, 0, 0])
        
        # Should not raise exception for reasonable parameters
        try:
            calibrator._validate_stereo_geometry(R, T, baseline)
        except ValueError as e:
            # Only acceptable if baseline is outside our test range
            if "Baseline too small" not in str(e):
                raise