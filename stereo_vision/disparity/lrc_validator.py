"""
Left-Right Consistency (LRC) Validator

Implements left-right consistency checking for occlusion detection and disparity validation.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

from ..utils.config_manager import ConfigManager


class LRCValidator:
    """Left-Right Consistency validator for disparity map validation."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize LRC validator.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Get LRC configuration
        lrc_config = self.config.get_lrc_params()
        
        # LRC threshold in pixels
        self.threshold = lrc_config.get('threshold', 1.0)
        
        self.logger.info(f"LRC validator initialized: threshold={self.threshold} pixels")
    
    def compute_right_disparity(self, 
                               left_image: np.ndarray, 
                               right_image: np.ndarray,
                               sgbm_estimator) -> np.ndarray:
        """
        Compute right-to-left disparity map.
        
        Args:
            left_image: Left rectified image
            right_image: Right rectified image
            sgbm_estimator: SGBM estimator instance
            
        Returns:
            Right disparity map (16-bit fixed point)
        """
        # Swap images to compute right-to-left disparity
        right_disparity = sgbm_estimator.compute_disparity(right_image, left_image, preprocess=True)
        
        return right_disparity
    
    def validate_consistency(self, 
                           left_disparity: np.ndarray, 
                           right_disparity: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform left-right consistency check.
        
        Args:
            left_disparity: Left-to-right disparity map (16-bit fixed point)
            right_disparity: Right-to-left disparity map (16-bit fixed point)
            
        Returns:
            Tuple of (validated_disparity, metrics)
        """
        if left_disparity.shape != right_disparity.shape:
            raise ValueError("Left and right disparity maps must have same dimensions")
        
        height, width = left_disparity.shape
        
        # Convert to float for calculations
        left_disp_float = left_disparity.astype(np.float32) / 16.0
        right_disp_float = right_disparity.astype(np.float32) / 16.0
        
        # Create consistency mask
        consistency_mask = np.zeros((height, width), dtype=bool)
        
        # Check consistency for each pixel
        for y in range(height):
            for x in range(width):
                left_d = left_disp_float[y, x]
                
                # Skip invalid disparities
                if left_d <= 0:
                    continue
                
                # Calculate corresponding pixel in right image
                right_x = int(x - left_d)
                
                # Check bounds
                if right_x < 0 or right_x >= width:
                    continue
                
                # Get right disparity at corresponding location
                right_d = right_disp_float[y, right_x]
                
                # Skip invalid right disparities
                if right_d <= 0:
                    continue
                
                # Check consistency: |left_d - right_d| <= threshold
                if abs(left_d - right_d) <= self.threshold:
                    consistency_mask[y, x] = True
        
        # Create validated disparity map
        validated_disparity = left_disparity.copy()
        validated_disparity[~consistency_mask] = 0
        
        # Calculate metrics
        total_valid_left = np.count_nonzero(left_disparity)
        total_consistent = np.count_nonzero(consistency_mask)
        
        metrics = {
            'total_pixels': left_disparity.size,
            'valid_left_pixels': total_valid_left,
            'consistent_pixels': total_consistent,
            'consistency_ratio': total_consistent / total_valid_left if total_valid_left > 0 else 0.0,
            'error_rate': 1.0 - (total_consistent / total_valid_left) if total_valid_left > 0 else 1.0
        }
        
        self.logger.debug(f"LRC validation: {total_consistent}/{total_valid_left} pixels consistent "
                         f"({metrics['consistency_ratio']:.3f})")
        
        return validated_disparity, metrics
    
    def validate_consistency_vectorized(self, 
                                      left_disparity: np.ndarray, 
                                      right_disparity: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Vectorized left-right consistency check for better performance.
        
        Args:
            left_disparity: Left-to-right disparity map (16-bit fixed point)
            right_disparity: Right-to-left disparity map (16-bit fixed point)
            
        Returns:
            Tuple of (validated_disparity, metrics)
        """
        if left_disparity.shape != right_disparity.shape:
            raise ValueError("Left and right disparity maps must have same dimensions")
        
        height, width = left_disparity.shape
        
        # Convert to float for calculations
        left_disp_float = left_disparity.astype(np.float32) / 16.0
        right_disp_float = right_disparity.astype(np.float32) / 16.0
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Calculate corresponding x coordinates in right image
        right_x_coords = x_coords - left_disp_float.astype(int)
        
        # Create valid mask for bounds checking
        valid_bounds = (right_x_coords >= 0) & (right_x_coords < width)
        valid_left = left_disp_float > 0
        
        # Initialize consistency mask
        consistency_mask = np.zeros((height, width), dtype=bool)
        
        # Get valid pixels
        valid_pixels = valid_bounds & valid_left
        
        if np.any(valid_pixels):
            # Extract coordinates for valid pixels
            valid_y = y_coords[valid_pixels]
            valid_right_x = right_x_coords[valid_pixels].astype(int)
            
            # Get corresponding right disparities
            right_disparities = right_disp_float[valid_y, valid_right_x]
            left_disparities = left_disp_float[valid_pixels]
            
            # Check consistency
            consistent = (right_disparities > 0) & (np.abs(left_disparities - right_disparities) <= self.threshold)
            
            # Update consistency mask
            consistency_mask[valid_pixels] = consistent
        
        # Create validated disparity map
        validated_disparity = left_disparity.copy()
        validated_disparity[~consistency_mask] = 0
        
        # Calculate metrics
        total_valid_left = np.count_nonzero(left_disparity)
        total_consistent = np.count_nonzero(consistency_mask)
        
        metrics = {
            'total_pixels': left_disparity.size,
            'valid_left_pixels': total_valid_left,
            'consistent_pixels': total_consistent,
            'consistency_ratio': total_consistent / total_valid_left if total_valid_left > 0 else 0.0,
            'error_rate': 1.0 - (total_consistent / total_valid_left) if total_valid_left > 0 else 1.0
        }
        
        self.logger.debug(f"LRC validation (vectorized): {total_consistent}/{total_valid_left} pixels consistent "
                         f"({metrics['consistency_ratio']:.3f})")
        
        return validated_disparity, metrics
    
    def create_consistency_visualization(self, 
                                       left_disparity: np.ndarray,
                                       validated_disparity: np.ndarray) -> np.ndarray:
        """
        Create visualization showing LRC validation results.
        
        Args:
            left_disparity: Original left disparity map
            validated_disparity: LRC-validated disparity map
            
        Returns:
            Color-coded visualization (green=consistent, red=inconsistent, black=invalid)
        """
        height, width = left_disparity.shape
        visualization = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Valid pixels in original disparity
        valid_original = left_disparity > 0
        
        # Consistent pixels (survived LRC)
        consistent = validated_disparity > 0
        
        # Inconsistent pixels (removed by LRC)
        inconsistent = valid_original & ~consistent
        
        # Color coding
        visualization[consistent] = [0, 255, 0]  # Green for consistent
        visualization[inconsistent] = [0, 0, 255]  # Red for inconsistent
        # Black for invalid (already initialized to 0)
        
        return visualization
    
    def analyze_occlusion_patterns(self, 
                                 left_disparity: np.ndarray,
                                 validated_disparity: np.ndarray) -> Dict[str, Any]:
        """
        Analyze occlusion patterns in the disparity map.
        
        Args:
            left_disparity: Original left disparity map
            validated_disparity: LRC-validated disparity map
            
        Returns:
            Occlusion analysis metrics
        """
        # Convert to float
        left_disp_float = left_disparity.astype(np.float32) / 16.0
        
        # Find occluded regions
        valid_original = left_disparity > 0
        consistent = validated_disparity > 0
        occluded = valid_original & ~consistent
        
        # Analyze occlusion by disparity range
        occlusion_by_disparity = {}
        
        if np.any(occluded):
            occluded_disparities = left_disp_float[occluded]
            
            # Bin disparities
            disparity_bins = np.arange(0, np.max(left_disp_float) + 5, 5)
            hist_occluded, _ = np.histogram(occluded_disparities, bins=disparity_bins)
            hist_total, _ = np.histogram(left_disp_float[valid_original], bins=disparity_bins)
            
            # Calculate occlusion rate per disparity range
            for i, (bin_start, bin_end) in enumerate(zip(disparity_bins[:-1], disparity_bins[1:])):
                total_in_bin = hist_total[i]
                occluded_in_bin = hist_occluded[i]
                
                if total_in_bin > 0:
                    occlusion_rate = occluded_in_bin / total_in_bin
                    occlusion_by_disparity[f"{bin_start:.0f}-{bin_end:.0f}"] = {
                        'total': int(total_in_bin),
                        'occluded': int(occluded_in_bin),
                        'rate': float(occlusion_rate)
                    }
        
        # Spatial analysis
        height, width = left_disparity.shape
        
        # Analyze occlusion by image regions
        regions = {
            'top_left': occluded[:height//2, :width//2],
            'top_right': occluded[:height//2, width//2:],
            'bottom_left': occluded[height//2:, :width//2],
            'bottom_right': occluded[height//2:, width//2:]
        }
        
        region_analysis = {}
        for region_name, region_mask in regions.items():
            total_region = valid_original[region_mask.shape[0]:region_mask.shape[0]+region_mask.shape[0],
                                        region_mask.shape[1]:region_mask.shape[1]+region_mask.shape[1]]
            
            region_analysis[region_name] = {
                'total_pixels': int(np.sum(total_region)),
                'occluded_pixels': int(np.sum(region_mask)),
                'occlusion_rate': float(np.sum(region_mask) / np.sum(total_region)) if np.sum(total_region) > 0 else 0.0
            }
        
        return {
            'total_occluded_pixels': int(np.sum(occluded)),
            'total_valid_pixels': int(np.sum(valid_original)),
            'overall_occlusion_rate': float(np.sum(occluded) / np.sum(valid_original)) if np.sum(valid_original) > 0 else 0.0,
            'occlusion_by_disparity': occlusion_by_disparity,
            'occlusion_by_region': region_analysis
        }
    
    def set_threshold(self, threshold: float) -> None:
        """
        Update LRC threshold.
        
        Args:
            threshold: New threshold in pixels
        """
        if threshold < 0:
            raise ValueError("LRC threshold must be non-negative")
        
        self.threshold = threshold
        self.logger.info(f"LRC threshold updated to {threshold} pixels")