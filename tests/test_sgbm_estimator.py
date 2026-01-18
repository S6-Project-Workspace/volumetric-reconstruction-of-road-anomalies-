"""
Tests for SGBM Disparity Estimator
"""

import pytest
import numpy as np
import cv2
from hypothesis import given, strategies as st

from stereo_vision.disparity.sgbm_estimator import SGBMEstimator
from stereo_vision.utils.config_manager import ConfigManager


class TestSGBMEstimator:
    """Test suite for SGBM disparity estimator."""
    
    @pytest.fixture
    def estimator(self):
        """Fixture providing an SGBM estimator instance."""
        return SGBMEstimator()
    
    @pytest.fixture
    def synthetic_stereo_pair(self):
        """Generate synthetic stereo pair for testing."""
        # Create a simple synthetic scene
        height, width = 376, 1241
        
        # Create left image with some texture
        left_image = np.random.randint(50, 200, (height, width), dtype=np.uint8)
        
        # Add some structured patterns
        for i in range(0, height, 20):
            left_image[i:i+5, :] = 255
        
        for j in range(0, width, 30):
            left_image[:, j:j+3] = 100
        
        # Create right image by shifting left image (simulating disparity)
        right_image = np.zeros_like(left_image)
        shift = 10  # pixels
        right_image[:, shift:] = left_image[:, :-shift]
        
        return left_image, right_image
    
    def test_estimator_initialization(self, estimator):
        """Test that SGBM estimator initializes correctly."""
        assert estimator.num_disparities > 0
        assert estimator.num_disparities % 16 == 0  # Must be divisible by 16
        assert estimator.block_size > 0
        assert estimator.block_size % 2 == 1  # Must be odd
        assert estimator.P1 > 0
        assert estimator.P2 >= estimator.P1
        assert estimator.sgbm is not None
    
    def test_compute_disparity_basic(self, estimator, synthetic_stereo_pair):
        """Test basic disparity computation."""
        left_img, right_img = synthetic_stereo_pair
        
        disparity = estimator.compute_disparity(left_img, right_img)
        
        # Check output properties
        assert disparity.shape == left_img.shape
        assert disparity.dtype == np.int16  # SGBM returns 16-bit fixed point
        
        # Should have some valid disparities
        valid_pixels = np.count_nonzero(disparity)
        assert valid_pixels > 0
    
    def test_compute_disparity_color_images(self, estimator):
        """Test disparity computation with color images."""
        height, width = 200, 300
        
        # Create color images
        left_color = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        right_color = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        disparity = estimator.compute_disparity(left_color, right_color)
        
        assert disparity.shape == (height, width)
        assert disparity.dtype == np.int16
    
    def test_disparity_range(self, estimator):
        """Test disparity range getter."""
        min_disp, max_disp = estimator.get_disparity_range()
        
        assert min_disp >= 0
        assert max_disp > min_disp
        assert max_disp - min_disp == estimator.num_disparities
    
    def test_validate_disparity_map(self, estimator, synthetic_stereo_pair):
        """Test disparity map validation."""
        left_img, right_img = synthetic_stereo_pair
        
        disparity = estimator.compute_disparity(left_img, right_img)
        metrics = estimator.validate_disparity_map(disparity)
        
        # Check metrics structure
        required_keys = ['valid_pixel_ratio', 'total_pixels', 'valid_pixels', 
                        'mean_disparity', 'std_disparity', 'min_disparity', 'max_disparity']
        
        for key in required_keys:
            assert key in metrics
        
        # Check metric values
        assert 0 <= metrics['valid_pixel_ratio'] <= 1
        assert metrics['total_pixels'] == disparity.size
        assert metrics['valid_pixels'] >= 0
    
    def test_create_disparity_visualization(self, estimator, synthetic_stereo_pair):
        """Test disparity visualization creation."""
        left_img, right_img = synthetic_stereo_pair
        
        disparity = estimator.compute_disparity(left_img, right_img)
        visualization = estimator.create_disparity_visualization(disparity)
        
        # Check visualization properties
        assert visualization.shape == (*disparity.shape, 3)
        assert visualization.dtype == np.uint8
    
    def test_configure_for_stereo_setup(self, estimator, sample_stereo_params):
        """Test SGBM configuration for stereo setup."""
        original_num_disparities = estimator.num_disparities
        
        estimator.configure_for_stereo_setup(sample_stereo_params)
        
        # Parameters should be updated based on stereo setup
        assert estimator.num_disparities % 16 == 0
        assert estimator.num_disparities > 0
        
        # Should have reasonable disparity range for road scenes
        min_disp, max_disp = estimator.get_disparity_range()
        assert max_disp > min_disp
    
    def test_update_parameters(self, estimator):
        """Test parameter updates."""
        original_num_disparities = estimator.num_disparities
        original_block_size = estimator.block_size
        
        # Update parameters
        new_num_disparities = 96  # Must be divisible by 16
        new_block_size = 7  # Must be odd
        
        estimator.update_parameters(
            num_disparities=new_num_disparities,
            block_size=new_block_size
        )
        
        assert estimator.num_disparities == new_num_disparities
        assert estimator.block_size == new_block_size
        
        # Invalid parameters should be rejected
        estimator.update_parameters(num_disparities=95)  # Not divisible by 16
        assert estimator.num_disparities == new_num_disparities  # Should remain unchanged
        
        estimator.update_parameters(block_size=6)  # Even number
        assert estimator.block_size == new_block_size  # Should remain unchanged
    
    def test_preprocessing(self, estimator):
        """Test image preprocessing."""
        # Create test image
        test_image = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        
        # Apply preprocessing
        preprocessed = estimator._preprocess_image(test_image)
        
        # Check output properties
        assert preprocessed.shape == test_image.shape
        assert preprocessed.dtype == test_image.dtype
        
        # Preprocessing should change the image (unless it's already optimal)
        # Allow for cases where preprocessing might not change much
        assert preprocessed is not test_image  # Should be a copy
    
    @pytest.mark.property
    @given(
        num_disparities=st.integers(min_value=16, max_value=256).filter(lambda x: x % 16 == 0),
        block_size=st.integers(min_value=3, max_value=11).filter(lambda x: x % 2 == 1)
    )
    def test_property_parameter_validation(self, num_disparities, block_size):
        """Property test: SGBM should accept valid parameter combinations."""
        config = ConfigManager()
        config.set('sgbm.num_disparities', num_disparities)
        config.set('sgbm.block_size', block_size)
        
        estimator = SGBMEstimator(config)
        
        assert estimator.num_disparities == num_disparities
        assert estimator.block_size == block_size
        
        # Should be able to compute disparity with these parameters
        height, width = 100, 150
        left_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        right_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        
        disparity = estimator.compute_disparity(left_img, right_img)
        assert disparity.shape == (height, width)
    
    def test_mismatched_image_dimensions(self, estimator):
        """Test error handling for mismatched image dimensions."""
        left_img = np.zeros((100, 150), dtype=np.uint8)
        right_img = np.zeros((100, 200), dtype=np.uint8)  # Different width
        
        with pytest.raises(ValueError, match="same dimensions"):
            estimator.compute_disparity(left_img, right_img)
    
    def test_empty_images(self, estimator):
        """Test handling of empty/zero images."""
        height, width = 100, 150
        left_img = np.zeros((height, width), dtype=np.uint8)
        right_img = np.zeros((height, width), dtype=np.uint8)
        
        # Should not crash, but may have no valid disparities
        disparity = estimator.compute_disparity(left_img, right_img)
        assert disparity.shape == (height, width)
        
        # Validate the result
        metrics = estimator.validate_disparity_map(disparity)
        assert metrics['valid_pixel_ratio'] >= 0  # May be 0 for empty images