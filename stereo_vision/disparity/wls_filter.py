"""
Weighted Least Squares (WLS) Disparity Filter

Implements edge-preserving disparity refinement using weighted least squares filtering.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
import logging

from ..utils.config_manager import ConfigManager


class WLSFilter:
    """Weighted Least Squares filter for sub-pixel disparity refinement."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize WLS filter.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Get WLS configuration
        wls_config = self.config.get_wls_params()
        
        # WLS parameters
        self.lambda_param = wls_config.get('lambda', 8000.0)  # Regularization strength
        self.sigma = wls_config.get('sigma', 1.5)  # Color sensitivity
        
        # Create WLS filter
        self.wls_filter = None
        
        self.logger.info(f"WLS filter initialized: lambda={self.lambda_param}, sigma={self.sigma}")
    
    def create_filter(self, sgbm_estimator) -> None:
        """
        Create WLS filter from SGBM estimator.
        
        Args:
            sgbm_estimator: SGBM estimator instance
        """
        # Create WLS filter from SGBM matcher
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(sgbm_estimator.sgbm)
        
        # Set WLS parameters
        self.wls_filter.setLambda(self.lambda_param)
        self.wls_filter.setSigmaColor(self.sigma)
        
        self.logger.debug("WLS filter created from SGBM estimator")
    
    def filter_disparity(self, 
                        disparity: np.ndarray, 
                        left_image: np.ndarray,
                        right_disparity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply WLS filtering to disparity map.
        
        Args:
            disparity: Input disparity map (16-bit fixed point)
            left_image: Left guide image for edge information
            right_disparity: Optional right disparity for better filtering
            
        Returns:
            Filtered disparity map (16-bit fixed point)
        """
        if self.wls_filter is None:
            raise RuntimeError("WLS filter not created. Call create_filter() first.")
        
        # Convert guide image to grayscale if needed
        if len(left_image.shape) == 3:
            guide_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        else:
            guide_image = left_image.copy()
        
        # If no right disparity provided, create a dummy one
        if right_disparity is None:
            # Create a simple right disparity by shifting left disparity
            right_disparity = np.zeros_like(disparity, dtype=np.float32)
            left_float = disparity.astype(np.float32) / 16.0
            
            height, width = disparity.shape
            for y in range(height):
                for x in range(width):
                    if left_float[y, x] > 0:
                        right_x = int(x - left_float[y, x])
                        if 0 <= right_x < width:
                            right_disparity[y, right_x] = left_float[y, x]
        else:
            # Convert right disparity to float32 as required by OpenCV
            right_disparity = right_disparity.astype(np.float32) / 16.0
        
        # Apply WLS filtering
        filtered_disparity = self.wls_filter.filter(
            disparity, guide_image, None, right_disparity
        )
        
        self.logger.debug("WLS filtering applied")
        
        return filtered_disparity
    
    def filter_with_confidence(self, 
                             disparity: np.ndarray, 
                             left_image: np.ndarray,
                             right_disparity: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply WLS filtering and return confidence map.
        
        Args:
            disparity: Input disparity map (16-bit fixed point)
            left_image: Left guide image for edge information
            right_disparity: Optional right disparity for better filtering
            
        Returns:
            Tuple of (filtered_disparity, confidence_map)
        """
        if self.wls_filter is None:
            raise RuntimeError("WLS filter not created. Call create_filter() first.")
        
        # Convert guide image to grayscale if needed
        if len(left_image.shape) == 3:
            guide_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        else:
            guide_image = left_image.copy()
        
        # Create confidence map
        confidence_map = np.zeros_like(disparity, dtype=np.float32)
        
        # If no right disparity provided, create a dummy one
        if right_disparity is None:
            # Create a simple right disparity by shifting left disparity
            right_disparity = np.zeros_like(disparity, dtype=np.float32)
            left_float = disparity.astype(np.float32) / 16.0
            
            height, width = disparity.shape
            for y in range(height):
                for x in range(width):
                    if left_float[y, x] > 0:
                        right_x = int(x - left_float[y, x])
                        if 0 <= right_x < width:
                            right_disparity[y, right_x] = left_float[y, x]
        else:
            # Convert right disparity to float32 as required by OpenCV
            right_disparity = right_disparity.astype(np.float32) / 16.0
        
        # Apply WLS filtering
        filtered_disparity = self.wls_filter.filter(
            disparity, guide_image, confidence_map, right_disparity
        )
        
        self.logger.debug("WLS filtering with confidence applied")
        
        return filtered_disparity, confidence_map
    
    def validate_filtering_quality(self, 
                                 original_disparity: np.ndarray,
                                 filtered_disparity: np.ndarray) -> Dict[str, Any]:
        """
        Validate the quality of WLS filtering.
        
        Args:
            original_disparity: Original disparity map
            filtered_disparity: WLS-filtered disparity map
            
        Returns:
            Quality metrics
        """
        # Convert to float for calculations
        orig_float = original_disparity.astype(np.float32) / 16.0
        filt_float = filtered_disparity.astype(np.float32) / 16.0
        
        # Find valid pixels in both maps
        valid_orig = original_disparity > 0
        valid_filt = filtered_disparity > 0
        
        # Calculate metrics
        metrics = {
            'original_valid_pixels': int(np.sum(valid_orig)),
            'filtered_valid_pixels': int(np.sum(valid_filt)),
            'pixel_retention_rate': 0.0,
            'mean_change': 0.0,
            'std_change': 0.0,
            'smoothness_improvement': 0.0
        }
        
        if np.sum(valid_orig) > 0:
            metrics['pixel_retention_rate'] = float(np.sum(valid_filt) / np.sum(valid_orig))
        
        # Calculate changes for pixels valid in both maps
        both_valid = valid_orig & valid_filt
        
        if np.sum(both_valid) > 0:
            changes = filt_float[both_valid] - orig_float[both_valid]
            metrics['mean_change'] = float(np.mean(changes))
            metrics['std_change'] = float(np.std(changes))
            
            # Calculate smoothness improvement
            orig_smoothness = self._calculate_smoothness(orig_float, valid_orig)
            filt_smoothness = self._calculate_smoothness(filt_float, valid_filt)
            
            if orig_smoothness > 0:
                metrics['smoothness_improvement'] = float((orig_smoothness - filt_smoothness) / orig_smoothness)
        
        return metrics
    
    def _calculate_smoothness(self, disparity: np.ndarray, valid_mask: np.ndarray) -> float:
        """
        Calculate disparity map smoothness using gradient magnitude.
        
        Args:
            disparity: Disparity map (float)
            valid_mask: Valid pixel mask
            
        Returns:
            Average gradient magnitude (lower is smoother)
        """
        # Calculate gradients
        grad_x = cv2.Sobel(disparity, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(disparity, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Average over valid pixels
        if np.sum(valid_mask) > 0:
            return float(np.mean(grad_magnitude[valid_mask]))
        else:
            return 0.0
    
    def create_filtering_visualization(self, 
                                     original_disparity: np.ndarray,
                                     filtered_disparity: np.ndarray) -> np.ndarray:
        """
        Create visualization comparing original and filtered disparity.
        
        Args:
            original_disparity: Original disparity map
            filtered_disparity: Filtered disparity map
            
        Returns:
            Side-by-side comparison visualization
        """
        # Convert to visualization format
        orig_vis = self._disparity_to_color(original_disparity)
        filt_vis = self._disparity_to_color(filtered_disparity)
        
        # Create side-by-side comparison
        height, width = orig_vis.shape[:2]
        comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        comparison[:, :width] = orig_vis
        comparison[:, width:] = filt_vis
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "WLS Filtered", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return comparison
    
    def _disparity_to_color(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to color visualization.
        
        Args:
            disparity: Disparity map (16-bit fixed point)
            
        Returns:
            Color-coded disparity visualization
        """
        # Convert to float and normalize
        disp_float = disparity.astype(np.float32) / 16.0
        
        # Mask invalid disparities
        valid_mask = disparity > 0
        
        # Normalize to 0-255 range
        if np.any(valid_mask):
            disp_norm = np.zeros_like(disp_float, dtype=np.uint8)
            valid_disp = disp_float[valid_mask]
            
            if len(valid_disp) > 0:
                min_disp = np.min(valid_disp)
                max_disp = np.max(valid_disp)
                
                if max_disp > min_disp:
                    disp_norm[valid_mask] = ((disp_float[valid_mask] - min_disp) / 
                                           (max_disp - min_disp) * 255).astype(np.uint8)
        else:
            disp_norm = np.zeros_like(disp_float, dtype=np.uint8)
        
        # Apply colormap
        disparity_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        
        # Set invalid pixels to black
        disparity_color[~valid_mask] = [0, 0, 0]
        
        return disparity_color
    
    def update_parameters(self, lambda_param: Optional[float] = None, sigma: Optional[float] = None) -> None:
        """
        Update WLS filter parameters.
        
        Args:
            lambda_param: Regularization strength
            sigma: Color sensitivity
        """
        if lambda_param is not None:
            self.lambda_param = max(0.0, lambda_param)
            if self.wls_filter is not None:
                self.wls_filter.setLambda(self.lambda_param)
        
        if sigma is not None:
            self.sigma = max(0.0, sigma)
            if self.wls_filter is not None:
                self.wls_filter.setSigmaColor(self.sigma)
        
        self.logger.info(f"WLS parameters updated: lambda={self.lambda_param}, sigma={self.sigma}")
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current WLS parameters.
        
        Returns:
            Dictionary of current parameters
        """
        return {
            'lambda': self.lambda_param,
            'sigma': self.sigma
        }