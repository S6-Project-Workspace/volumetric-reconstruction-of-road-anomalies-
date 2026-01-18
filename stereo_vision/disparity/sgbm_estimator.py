"""
Semi-Global Block Matching (SGBM) Disparity Estimator

Implements optimized SGBM with road-specific parameters for robust disparity estimation.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

from ..data_models import StereoParameters
from ..utils.config_manager import ConfigManager


class SGBMEstimator:
    """Optimized SGBM disparity estimator with road-specific parameter tuning."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize SGBM estimator.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Get SGBM configuration
        sgbm_config = self.config.get_sgbm_params()
        
        # SGBM parameters optimized for road scenes
        self.min_disparity = sgbm_config.get('min_disparity', 0)
        self.num_disparities = sgbm_config.get('num_disparities', 160)  # Must be divisible by 16
        self.block_size = sgbm_config.get('block_size', 5)
        
        # Smoothness parameters - higher for road scenes to handle textureless areas
        self.P1 = sgbm_config.get('P1', 600)  # 8 * 3 * block_size^2
        self.P2 = sgbm_config.get('P2', 2400)  # 32 * 3 * block_size^2
        
        # Quality control parameters
        self.disp12_max_diff = sgbm_config.get('disp12_max_diff', 1)
        self.uniqueness_ratio = sgbm_config.get('uniqueness_ratio', 10)
        self.speckle_window_size = sgbm_config.get('speckle_window_size', 100)
        self.speckle_range = sgbm_config.get('speckle_range', 32)
        
        # SGBM mode
        mode_str = sgbm_config.get('mode', 'StereoSGBM_MODE_SGBM_3WAY')
        self.mode = getattr(cv2, mode_str)
        
        # Create SGBM matcher
        self._create_sgbm_matcher()
        
        self.logger.info(f"SGBM estimator initialized: {self.num_disparities} disparities, block_size={self.block_size}")
    
    def _create_sgbm_matcher(self) -> None:
        """Create SGBM matcher with configured parameters."""
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=self.P1,
            P2=self.P2,
            disp12MaxDiff=self.disp12_max_diff,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            mode=self.mode
        )
    
    def configure_for_stereo_setup(self, stereo_params: StereoParameters) -> None:
        """
        Configure SGBM parameters based on stereo camera setup.
        
        Args:
            stereo_params: Stereo calibration parameters
        """
        # Calculate optimal disparity range based on baseline and focal length
        focal_length = stereo_params.left_camera.camera_matrix[0, 0]  # fx
        baseline = stereo_params.baseline
        
        # Estimate disparity range for road scenes (0.5m to 50m depth)
        min_depth = 0.5  # meters
        max_depth = 50.0  # meters
        
        max_disparity = int((focal_length * baseline) / min_depth)
        min_disparity_calc = int((focal_length * baseline) / max_depth)
        
        # Ensure disparity range is reasonable and divisible by 16
        self.num_disparities = ((max_disparity // 16) + 1) * 16
        self.min_disparity = max(0, min_disparity_calc)
        
        # Adjust smoothness parameters based on baseline
        # Larger baseline -> more reliable matching -> can reduce smoothness
        baseline_factor = min(2.0, baseline / 0.5)  # Normalize to 0.5m baseline
        
        self.P1 = int(600 / baseline_factor)
        self.P2 = int(2400 / baseline_factor)
        
        # Recreate matcher with new parameters
        self._create_sgbm_matcher()
        
        self.logger.info(f"SGBM configured for stereo setup: baseline={baseline:.3f}m, "
                        f"disparities={self.min_disparity}-{self.min_disparity + self.num_disparities}")
    
    def compute_disparity(self, 
                         left_image: np.ndarray, 
                         right_image: np.ndarray,
                         preprocess: bool = True) -> np.ndarray:
        """
        Compute disparity map using SGBM.
        
        Args:
            left_image: Left rectified image
            right_image: Right rectified image
            preprocess: Whether to apply preprocessing
            
        Returns:
            Disparity map (16-bit fixed point, divide by 16 for actual disparity)
        """
        if left_image.shape != right_image.shape:
            raise ValueError("Left and right images must have same dimensions")
        
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image.copy()
            right_gray = right_image.copy()
        
        # Preprocessing for better matching
        if preprocess:
            left_gray = self._preprocess_image(left_gray)
            right_gray = self._preprocess_image(right_gray)
        
        # Compute disparity
        disparity = self.sgbm.compute(left_gray, right_gray)
        
        # Handle invalid disparities
        disparity[disparity <= self.min_disparity * 16] = 0
        disparity[disparity >= (self.min_disparity + self.num_disparities) * 16] = 0
        
        self.logger.debug(f"Computed disparity map: {np.count_nonzero(disparity)} valid pixels")
        
        return disparity
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better SGBM matching.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image
        """
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        
        return blurred
    
    def get_disparity_range(self) -> Tuple[int, int]:
        """
        Get the current disparity range.
        
        Returns:
            Tuple of (min_disparity, max_disparity)
        """
        return self.min_disparity, self.min_disparity + self.num_disparities
    
    def validate_disparity_map(self, disparity: np.ndarray) -> Dict[str, Any]:
        """
        Validate disparity map quality.
        
        Args:
            disparity: Disparity map (16-bit fixed point)
            
        Returns:
            Validation metrics
        """
        # Convert to float disparity
        disp_float = disparity.astype(np.float32) / 16.0
        
        # Calculate metrics
        valid_pixels = np.count_nonzero(disparity)
        total_pixels = disparity.size
        valid_ratio = valid_pixels / total_pixels
        
        # Calculate disparity statistics for valid pixels
        valid_disparities = disp_float[disparity > 0]
        
        metrics = {
            'valid_pixel_ratio': valid_ratio,
            'total_pixels': total_pixels,
            'valid_pixels': valid_pixels,
            'mean_disparity': 0.0,
            'std_disparity': 0.0,
            'min_disparity': 0.0,
            'max_disparity': 0.0
        }
        
        if len(valid_disparities) > 0:
            metrics.update({
                'mean_disparity': float(np.mean(valid_disparities)),
                'std_disparity': float(np.std(valid_disparities)),
                'min_disparity': float(np.min(valid_disparities)),
                'max_disparity': float(np.max(valid_disparities))
            })
        
        return metrics
    
    def create_disparity_visualization(self, disparity: np.ndarray) -> np.ndarray:
        """
        Create a visualization of the disparity map.
        
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
    
    def update_parameters(self, **kwargs) -> None:
        """
        Update SGBM parameters dynamically.
        
        Args:
            **kwargs: Parameter updates
        """
        updated = False
        
        if 'num_disparities' in kwargs:
            new_val = kwargs['num_disparities']
            if new_val % 16 == 0 and new_val > 0:
                self.num_disparities = new_val
                updated = True
        
        if 'block_size' in kwargs:
            new_val = kwargs['block_size']
            if new_val % 2 == 1 and new_val >= 3:  # Must be odd and >= 3
                self.block_size = new_val
                updated = True
        
        if 'P1' in kwargs:
            self.P1 = max(1, kwargs['P1'])
            updated = True
        
        if 'P2' in kwargs:
            self.P2 = max(self.P1, kwargs['P2'])
            updated = True
        
        if 'uniqueness_ratio' in kwargs:
            self.uniqueness_ratio = max(0, kwargs['uniqueness_ratio'])
            updated = True
        
        if updated:
            self._create_sgbm_matcher()
            self.logger.info("SGBM parameters updated")