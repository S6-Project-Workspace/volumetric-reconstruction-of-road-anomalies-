"""
Tests for Left-Right Consistency Validator
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st

from stereo_vision.disparity.lrc_validator import LRCValidator
from stereo_vision.disparity.sgbm_estimator import SGBMEstimator


class TestLRCValidator:
    """Test suite for LRC validator."""
    
    @pytest.fixture
    def validator(self):
        """Fixture providing an LRC validator instance."""
        return LRCValidator()
    
    @pytest.fixture
    def sgbm_estimator(self):
        """Fixture providing an SGBM estimator for testing."""
        return SGBMEstimator()
    
    @pytest.fixture
    def synthetic_disparity_pair(self):
        """Generate synthetic left and right disparity maps."""
        height, width = 200, 300
        
        # Create left disparity map
        left_disparity = np.zeros((height, width), dtype=np.int16)
        
        # Add some disparity values (16-bit fixed point)
        for y in range(50, 150):
            for x in range(50, 250):
                # Simple disparity pattern
                disp_value = int((10 + (x - 50) * 0.1) * 16)  # 10-30 pixels disparity
                left_disparity[y, x] = disp_value
        
        # Create corresponding right disparity map
        right_disparity = np.zeros((height, width), dtype=np.int16)
        
        # Simulate consistent disparities with some noise
        for y in range(50, 150):
            for x in range(50, 250):
                left_d = left_disparity[y, x] / 16.0
                if left_d > 0:
                    right_x = int(x - left_d)
                    if 0 <= right_x < width:
                        # Add small noise to simulate real conditions
                        noise = np.random.normal(0, 0.2)  # Small noise
                        right_disparity[y, right_x] = int((left_d + noise) * 16)
        
        return left_disparity, right_disparity
    
    def test_validator_initialization(self, validator):
        """Test that LRC validator initializes correctly."""
        assert validator.threshold > 0
        assert hasattr(validator, 'logger')
    
    def test_validate_consistency_basic(self, validator, synthetic_disparity_pair):
        """Test basic left-right consistency validation."""
        left_disp, right_disp = synthetic_disparity_pair
        
        validated_disp, metrics = validator.validate_consistency(left_disp, right_disp)
        
        # Check output properties
        assert validated_disp.shape == left_disp.shape
        assert validated_disp.dtype == left_disp.dtype
        
        # Check metrics
        required_keys = ['total_pixels', 'valid_left_pixels', 'consistent_pixels', 
                        'consistency_ratio', 'error_rate']
        
        for key in required_keys:
            assert key in metrics
        
        # Validate metric values
        assert metrics['total_pixels'] == left_disp.size
        assert metrics['valid_left_pixels'] >= 0
        assert metrics['consistent_pixels'] >= 0
        assert metrics['consistent_pixels'] <= metrics['valid_left_pixels']
        assert 0 <= metrics['consistency_ratio'] <= 1
        assert 0 <= metrics['error_rate'] <= 1
        assert abs(metrics['consistency_ratio'] + metrics['error_rate'] - 1.0) < 1e-6
    
    def test_validate_consistency_vectorized(self, validator, synthetic_disparity_pair):
        """Test vectorized consistency validation."""
        left_disp, right_disp = synthetic_disparity_pair
        
        # Test both methods
        validated_basic, metrics_basic = validator.validate_consistency(left_disp, right_disp)
        validated_vec, metrics_vec = validator.validate_consistency_vectorized(left_disp, right_disp)
        
        # Results should be similar (allowing for small numerical differences)
        assert validated_basic.shape == validated_vec.shape
        
        # Metrics should be very close
        assert abs(metrics_basic['consistency_ratio'] - metrics_vec['consistency_ratio']) < 0.01
        assert abs(metrics_basic['error_rate'] - metrics_vec['error_rate']) < 0.01
    
    def test_mismatched_disparity_dimensions(self, validator):
        """Test error handling for mismatched disparity dimensions."""
        left_disp = np.zeros((100, 150), dtype=np.int16)
        right_disp = np.zeros((100, 200), dtype=np.int16)  # Different width
        
        with pytest.raises(ValueError, match="same dimensions"):
            validator.validate_consistency(left_disp, right_disp)
    
    def test_create_consistency_visualization(self, validator, synthetic_disparity_pair):
        """Test consistency visualization creation."""
        left_disp, right_disp = synthetic_disparity_pair
        
        validated_disp, _ = validator.validate_consistency(left_disp, right_disp)
        visualization = validator.create_consistency_visualization(left_disp, validated_disp)
        
        # Check visualization properties
        assert visualization.shape == (*left_disp.shape, 3)
        assert visualization.dtype == np.uint8
        
        # Should have different colors for different pixel types
        unique_colors = np.unique(visualization.reshape(-1, 3), axis=0)
        assert len(unique_colors) >= 2  # At least black and one other color
    
    def test_analyze_occlusion_patterns(self, validator, synthetic_disparity_pair):
        """Test occlusion pattern analysis."""
        left_disp, right_disp = synthetic_disparity_pair
        
        validated_disp, _ = validator.validate_consistency(left_disp, right_disp)
        analysis = validator.analyze_occlusion_patterns(left_disp, validated_disp)
        
        # Check analysis structure
        required_keys = ['total_occluded_pixels', 'total_valid_pixels', 'overall_occlusion_rate',
                        'occlusion_by_disparity', 'occlusion_by_region']
        
        for key in required_keys:
            assert key in analysis
        
        # Validate analysis values
        assert analysis['total_occluded_pixels'] >= 0
        assert analysis['total_valid_pixels'] >= 0
        assert 0 <= analysis['overall_occlusion_rate'] <= 1
        assert isinstance(analysis['occlusion_by_disparity'], dict)
        assert isinstance(analysis['occlusion_by_region'], dict)
    
    def test_set_threshold(self, validator):
        """Test threshold setting."""
        original_threshold = validator.threshold
        
        new_threshold = 2.0
        validator.set_threshold(new_threshold)
        assert validator.threshold == new_threshold
        
        # Test invalid threshold
        with pytest.raises(ValueError, match="non-negative"):
            validator.set_threshold(-1.0)
    
    @pytest.mark.property
    def test_property_lrc_validation(self, validator):
        """
        Property test: Left-Right Consistency validation effectiveness.
        **Feature: advanced-stereo-vision-pipeline, Property 5: Left-Right Consistency Validation**
        **Validates: Requirements 2.2**
        """
        height, width = 100, 150
        
        # Create perfect left disparity
        left_disparity = np.zeros((height, width), dtype=np.int16)
        
        # Add consistent disparity pattern
        for y in range(20, 80):
            for x in range(20, 130):
                disp_value = int(15 * 16)  # 15 pixels disparity
                left_disparity[y, x] = disp_value
        
        # Create perfectly consistent right disparity
        right_disparity = np.zeros((height, width), dtype=np.int16)
        for y in range(20, 80):
            for x in range(20, 130):
                left_d = left_disparity[y, x] / 16.0
                if left_d > 0:
                    right_x = int(x - left_d)
                    if 0 <= right_x < width:
                        right_disparity[y, right_x] = left_disparity[y, x]
        
        # Validate consistency
        validated_disp, metrics = validator.validate_consistency(left_disparity, right_disparity)
        
        # With perfect consistency, most pixels should pass
        assert metrics['consistency_ratio'] > 0.8, \
            f"Consistency ratio {metrics['consistency_ratio']:.3f} should be > 0.8 for perfect data"
        
        # Error rate should be low
        assert metrics['error_rate'] < 0.2, \
            f"Error rate {metrics['error_rate']:.3f} should be < 0.2 for perfect data"
    
    @pytest.mark.property
    @given(
        threshold=st.floats(min_value=0.1, max_value=5.0),
        noise_level=st.floats(min_value=0.0, max_value=2.0)
    )
    def test_property_threshold_sensitivity(self, threshold, noise_level):
        """Property test: LRC validation should be sensitive to threshold changes."""
        validator = LRCValidator()
        validator.set_threshold(threshold)
        
        height, width = 50, 75
        
        # Create left disparity
        left_disparity = np.zeros((height, width), dtype=np.int16)
        for y in range(10, 40):
            for x in range(10, 65):
                left_disparity[y, x] = int(10 * 16)  # 10 pixels disparity
        
        # Create right disparity with controlled noise
        right_disparity = np.zeros((height, width), dtype=np.int16)
        for y in range(10, 40):
            for x in range(10, 65):
                left_d = left_disparity[y, x] / 16.0
                if left_d > 0:
                    right_x = int(x - left_d)
                    if 0 <= right_x < width:
                        # Add controlled noise
                        noise = np.random.normal(0, noise_level)
                        right_disparity[y, right_x] = int((left_d + noise) * 16)
        
        # Validate consistency
        _, metrics = validator.validate_consistency(left_disparity, right_disparity)
        
        # Consistency should depend on noise level relative to threshold
        if noise_level <= threshold:
            # Low noise should result in high consistency
            assert metrics['consistency_ratio'] > 0.5
        
        # Should always have valid metrics
        assert 0 <= metrics['consistency_ratio'] <= 1
        assert 0 <= metrics['error_rate'] <= 1
    
    def test_empty_disparity_maps(self, validator):
        """Test handling of empty disparity maps."""
        height, width = 100, 150
        left_disp = np.zeros((height, width), dtype=np.int16)
        right_disp = np.zeros((height, width), dtype=np.int16)
        
        validated_disp, metrics = validator.validate_consistency(left_disp, right_disp)
        
        # Should handle empty maps gracefully
        assert validated_disp.shape == left_disp.shape
        assert metrics['valid_left_pixels'] == 0
        assert metrics['consistent_pixels'] == 0
        assert metrics['consistency_ratio'] == 0.0
        assert metrics['error_rate'] == 1.0
    
    def test_compute_right_disparity(self, validator, sgbm_estimator):
        """Test right disparity computation."""
        height, width = 100, 150
        
        # Create synthetic stereo pair
        left_image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        right_image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        
        # Compute right disparity
        right_disparity = validator.compute_right_disparity(left_image, right_image, sgbm_estimator)
        
        # Check output properties
        assert right_disparity.shape == (height, width)
        assert right_disparity.dtype == np.int16