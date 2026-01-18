"""
Integration Tests for Disparity Estimation Pipeline

Tests the complete disparity estimation workflow: SGBM -> LRC -> WLS
"""

import pytest
import numpy as np
import cv2

from stereo_vision.disparity.sgbm_estimator import SGBMEstimator
from stereo_vision.disparity.lrc_validator import LRCValidator
from stereo_vision.disparity.wls_filter import WLSFilter


class TestDisparityIntegration:
    """Integration test suite for disparity estimation pipeline."""
    
    @pytest.fixture
    def disparity_pipeline(self):
        """Fixture providing complete disparity estimation pipeline."""
        sgbm = SGBMEstimator()
        lrc = LRCValidator()
        wls = WLSFilter()
        wls.create_filter(sgbm)
        
        return {
            'sgbm': sgbm,
            'lrc': lrc,
            'wls': wls
        }
    
    @pytest.fixture
    def synthetic_stereo_scene(self):
        """Generate a more realistic synthetic stereo scene."""
        height, width = 300, 400
        
        # Create left image with road-like scene
        left_image = np.zeros((height, width), dtype=np.uint8)
        
        # Add road surface (textured)
        road_region = slice(200, height), slice(None)
        left_image[road_region] = np.random.randint(80, 120, left_image[road_region].shape)
        
        # Add lane markings
        left_image[250:260, 100:110] = 255
        left_image[250:260, 290:300] = 255
        
        # Add some objects at different depths
        left_image[150:200, 150:200] = 180  # Closer object
        left_image[100:150, 250:300] = 160  # Farther object
        
        # Create right image by shifting with depth-dependent disparity
        right_image = np.zeros_like(left_image)
        
        # Road surface - far, small disparity
        for y in range(200, height):
            for x in range(width):
                if x >= 5:  # 5 pixel disparity for road
                    right_image[y, x-5] = left_image[y, x]
        
        # Closer object - larger disparity
        for y in range(150, 200):
            for x in range(150, 200):
                if x >= 15:  # 15 pixel disparity for closer object
                    right_image[y, x-15] = left_image[y, x]
        
        # Farther object - smaller disparity
        for y in range(100, 150):
            for x in range(250, 300):
                if x >= 8:  # 8 pixel disparity for farther object
                    right_image[y, x-8] = left_image[y, x]
        
        # Lane markings
        for y in range(250, 260):
            for x in range(100, 110):
                if x >= 5:
                    right_image[y, x-5] = left_image[y, x]
            for x in range(290, 300):
                if x >= 5:
                    right_image[y, x-5] = left_image[y, x]
        
        return left_image, right_image
    
    def test_complete_disparity_pipeline(self, disparity_pipeline, synthetic_stereo_scene):
        """Test the complete disparity estimation pipeline."""
        sgbm = disparity_pipeline['sgbm']
        lrc = disparity_pipeline['lrc']
        wls = disparity_pipeline['wls']
        
        left_img, right_img = synthetic_stereo_scene
        
        # Step 1: Compute initial disparity with SGBM
        left_disparity = sgbm.compute_disparity(left_img, right_img)
        
        # Validate SGBM output
        assert left_disparity.shape == left_img.shape
        assert left_disparity.dtype == np.int16
        
        sgbm_metrics = sgbm.validate_disparity_map(left_disparity)
        assert sgbm_metrics['valid_pixel_ratio'] > 0
        
        # Step 2: Compute right disparity for LRC
        right_disparity = lrc.compute_right_disparity(left_img, right_img, sgbm)
        
        # Step 3: Apply Left-Right Consistency check
        lrc_disparity, lrc_metrics = lrc.validate_consistency_vectorized(left_disparity, right_disparity)
        
        # Validate LRC output
        assert lrc_disparity.shape == left_disparity.shape
        assert lrc_metrics['consistent_pixels'] <= lrc_metrics['valid_left_pixels']
        assert 0 <= lrc_metrics['consistency_ratio'] <= 1
        
        # Step 4: Apply WLS filtering
        wls_disparity = wls.filter_disparity(lrc_disparity, left_img)
        
        # Validate WLS output
        assert wls_disparity.shape == lrc_disparity.shape
        assert wls_disparity.dtype == lrc_disparity.dtype
        
        # Step 5: Validate overall pipeline quality
        wls_metrics = wls.validate_filtering_quality(lrc_disparity, wls_disparity)
        
        # The pipeline should maintain reasonable quality
        assert wls_metrics['pixel_retention_rate'] > 0.5  # Should retain most pixels
        
        return {
            'sgbm_disparity': left_disparity,
            'lrc_disparity': lrc_disparity,
            'wls_disparity': wls_disparity,
            'sgbm_metrics': sgbm_metrics,
            'lrc_metrics': lrc_metrics,
            'wls_metrics': wls_metrics
        }
    
    def test_pipeline_quality_improvement(self, disparity_pipeline, synthetic_stereo_scene):
        """Test that each stage improves disparity quality."""
        results = self.test_complete_disparity_pipeline(disparity_pipeline, synthetic_stereo_scene)
        
        sgbm_disparity = results['sgbm_disparity']
        lrc_disparity = results['lrc_disparity']
        wls_disparity = results['wls_disparity']
        
        # Count valid pixels at each stage
        sgbm_valid = np.count_nonzero(sgbm_disparity)
        lrc_valid = np.count_nonzero(lrc_disparity)
        wls_valid = np.count_nonzero(wls_disparity)
        
        # LRC should remove some invalid pixels (quality over quantity)
        assert lrc_valid <= sgbm_valid
        
        # WLS might change the number of valid pixels, but should improve quality
        # The exact relationship depends on the filtering parameters
        
        # Calculate disparity consistency in a known region
        # (This is a simplified quality measure)
        test_region = slice(200, 250), slice(50, 350)  # Road region
        
        def calculate_consistency(disparity, region):
            """Calculate disparity consistency in a region."""
            region_disp = disparity[region].astype(np.float32) / 16.0
            valid_disp = region_disp[region_disp > 0]
            if len(valid_disp) > 10:
                return np.std(valid_disp)
            return float('inf')
        
        sgbm_consistency = calculate_consistency(sgbm_disparity, test_region)
        lrc_consistency = calculate_consistency(lrc_disparity, test_region)
        wls_consistency = calculate_consistency(wls_disparity, test_region)
        
        # LRC should improve consistency by removing outliers
        if lrc_consistency < float('inf') and sgbm_consistency < float('inf'):
            assert lrc_consistency <= sgbm_consistency * 1.1  # Allow small tolerance
        
        # WLS should further improve consistency through smoothing
        if wls_consistency < float('inf') and lrc_consistency < float('inf'):
            assert wls_consistency <= lrc_consistency * 1.1  # Allow small tolerance
    
    def test_pipeline_with_stereo_parameters(self, disparity_pipeline, sample_stereo_params):
        """Test pipeline configuration with stereo parameters."""
        sgbm = disparity_pipeline['sgbm']
        
        # Configure SGBM for stereo setup
        original_num_disparities = sgbm.num_disparities
        sgbm.configure_for_stereo_setup(sample_stereo_params)
        
        # Parameters should be updated
        assert sgbm.num_disparities % 16 == 0
        
        # Should still be able to process images
        height, width = 200, 300
        left_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        right_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        
        disparity = sgbm.compute_disparity(left_img, right_img)
        assert disparity.shape == (height, width)
    
    def test_pipeline_error_handling(self, disparity_pipeline):
        """Test pipeline error handling with invalid inputs."""
        sgbm = disparity_pipeline['sgbm']
        lrc = disparity_pipeline['lrc']
        wls = disparity_pipeline['wls']
        
        # Test mismatched image dimensions
        left_img = np.zeros((100, 150), dtype=np.uint8)
        right_img = np.zeros((100, 200), dtype=np.uint8)  # Different width
        
        with pytest.raises(ValueError, match="same dimensions"):
            sgbm.compute_disparity(left_img, right_img)
        
        # Test mismatched disparity dimensions for LRC
        left_disp = np.zeros((100, 150), dtype=np.int16)
        right_disp = np.zeros((100, 200), dtype=np.int16)  # Different width
        
        with pytest.raises(ValueError, match="same dimensions"):
            lrc.validate_consistency(left_disp, right_disp)
    
    def test_pipeline_visualization_outputs(self, disparity_pipeline, synthetic_stereo_scene):
        """Test that pipeline generates proper visualizations."""
        results = self.test_complete_disparity_pipeline(disparity_pipeline, synthetic_stereo_scene)
        
        sgbm = disparity_pipeline['sgbm']
        lrc = disparity_pipeline['lrc']
        wls = disparity_pipeline['wls']
        
        sgbm_disparity = results['sgbm_disparity']
        lrc_disparity = results['lrc_disparity']
        wls_disparity = results['wls_disparity']
        
        # Test SGBM visualization
        sgbm_vis = sgbm.create_disparity_visualization(sgbm_disparity)
        assert sgbm_vis.shape == (*sgbm_disparity.shape, 3)
        assert sgbm_vis.dtype == np.uint8
        
        # Test LRC visualization
        lrc_vis = lrc.create_consistency_visualization(sgbm_disparity, lrc_disparity)
        assert lrc_vis.shape == (*sgbm_disparity.shape, 3)
        assert lrc_vis.dtype == np.uint8
        
        # Test WLS visualization
        wls_vis = wls.create_filtering_visualization(lrc_disparity, wls_disparity)
        height, width = lrc_disparity.shape
        assert wls_vis.shape == (height, width * 2, 3)
        assert wls_vis.dtype == np.uint8
    
    def test_pipeline_parameter_updates(self, disparity_pipeline):
        """Test dynamic parameter updates in pipeline."""
        sgbm = disparity_pipeline['sgbm']
        lrc = disparity_pipeline['lrc']
        wls = disparity_pipeline['wls']
        
        # Update SGBM parameters
        sgbm.update_parameters(num_disparities=96, block_size=7)
        assert sgbm.num_disparities == 96
        assert sgbm.block_size == 7
        
        # Update LRC threshold
        lrc.set_threshold(2.0)
        assert lrc.threshold == 2.0
        
        # Update WLS parameters
        wls.update_parameters(lambda_param=10000.0, sigma=2.0)
        assert wls.lambda_param == 10000.0
        assert wls.sigma == 2.0
        
        # Pipeline should still work with updated parameters
        height, width = 100, 150
        left_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        right_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        
        # Test updated pipeline
        left_disparity = sgbm.compute_disparity(left_img, right_img)
        right_disparity = lrc.compute_right_disparity(left_img, right_img, sgbm)
        lrc_disparity, _ = lrc.validate_consistency(left_disparity, right_disparity)
        wls_disparity = wls.filter_disparity(lrc_disparity, left_img)
        
        # Should produce valid outputs
        assert wls_disparity.shape == (height, width)
    
    def test_pipeline_metrics_aggregation(self, disparity_pipeline, synthetic_stereo_scene):
        """Test aggregation of metrics from all pipeline stages."""
        results = self.test_complete_disparity_pipeline(disparity_pipeline, synthetic_stereo_scene)
        
        # Aggregate metrics from all stages
        pipeline_metrics = {
            'sgbm': results['sgbm_metrics'],
            'lrc': results['lrc_metrics'],
            'wls': results['wls_metrics']
        }
        
        # Calculate overall pipeline quality score
        sgbm_quality = min(1.0, results['sgbm_metrics']['valid_pixel_ratio'])
        lrc_quality = results['lrc_metrics']['consistency_ratio']
        wls_quality = results['wls_metrics']['pixel_retention_rate']
        
        overall_quality = (sgbm_quality + lrc_quality + wls_quality) / 3.0
        
        # Overall quality should be reasonable
        assert 0 <= overall_quality <= 1
        
        # Store aggregated metrics
        pipeline_metrics['overall'] = {
            'quality_score': overall_quality,
            'sgbm_contribution': sgbm_quality,
            'lrc_contribution': lrc_quality,
            'wls_contribution': wls_quality
        }
        
        return pipeline_metrics