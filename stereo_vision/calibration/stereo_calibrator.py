"""
Stereo Camera Calibrator

Implements two-stage stereo calibration with fixed intrinsic parameters for metric accuracy.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from ..data_models import CameraParameters, StereoParameters, RectificationMaps
from ..utils.config_manager import ConfigManager


class StereoCalibrator:
    """Manages two-stage stereo calibration process with fixed intrinsic parameters."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize stereo calibrator.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Calibration quality thresholds
        self.max_reprojection_error = 0.5  # pixels
        self.min_baseline = 0.1  # meters
        self.max_baseline = 2.0  # meters
        
        self.logger.info("Stereo calibrator initialized")
    
    def calibrate_stereo(self, 
                        left_params: CameraParameters, 
                        right_params: CameraParameters,
                        left_images: List[np.ndarray],
                        right_images: List[np.ndarray],
                        object_points: List[np.ndarray],
                        left_image_points: List[np.ndarray],
                        right_image_points: List[np.ndarray]) -> StereoParameters:
        """
        Perform stereo calibration with fixed intrinsic parameters.
        
        Args:
            left_params: Left camera intrinsic parameters
            right_params: Right camera intrinsic parameters
            left_images: Left calibration images
            right_images: Right calibration images
            object_points: 3D object points for each image
            left_image_points: 2D image points in left images
            right_image_points: 2D image points in right images
            
        Returns:
            Stereo parameters including baseline and Q matrix
        """
        if len(left_images) != len(right_images):
            raise ValueError("Number of left and right images must match")
        
        if len(object_points) != len(left_images):
            raise ValueError("Number of object points must match number of image pairs")
        
        self.logger.info(f"Starting stereo calibration with {len(left_images)} image pairs")
        
        # Image size (assume all images have same size)
        image_size = left_params.image_size
        
        # Stereo calibration flags - fix intrinsic parameters
        calibration_flags = (
            cv2.CALIB_FIX_INTRINSIC |  # Keep intrinsic parameters fixed
            cv2.CALIB_RATIONAL_MODEL |  # Use rational distortion model
            cv2.CALIB_THIN_PRISM_MODEL  # Include thin prism distortion
        )
        
        # Perform stereo calibration
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            object_points,
            left_image_points,
            right_image_points,
            left_params.camera_matrix,
            left_params.distortion_coeffs,
            right_params.camera_matrix,
            right_params.distortion_coeffs,
            image_size,
            flags=calibration_flags
        )
        
        if not ret:
            raise RuntimeError("Stereo calibration failed")
        
        # Calculate baseline (distance between camera centers)
        baseline = np.linalg.norm(T)
        
        self.logger.info(f"Stereo calibration completed: baseline = {baseline:.4f}m")
        
        # Validate stereo geometry
        self._validate_stereo_geometry(R, T, baseline)
        
        # Calculate reprojection error
        stereo_error = self._calculate_stereo_reprojection_error(
            object_points, left_image_points, right_image_points,
            left_params, right_params, R, T
        )
        
        self.logger.info(f"Stereo reprojection error: {stereo_error:.4f} pixels")
        
        if stereo_error > self.max_reprojection_error:
            self.logger.warning(f"High stereo reprojection error: {stereo_error:.4f} > {self.max_reprojection_error}")
        
        # Compute rectification and Q matrix
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            left_params.camera_matrix,
            left_params.distortion_coeffs,
            right_params.camera_matrix,
            right_params.distortion_coeffs,
            image_size,
            R, T,
            alpha=0.0  # Crop to valid pixels only
        )
        
        return StereoParameters(
            left_camera=left_params,
            right_camera=right_params,
            rotation_matrix=R,
            translation_vector=T,
            baseline=baseline,
            Q_matrix=Q
        )
    
    def compute_rectification_maps(self, stereo_params: StereoParameters) -> RectificationMaps:
        """
        Compute rectification maps for stereo pair.
        
        Args:
            stereo_params: Stereo calibration parameters
            
        Returns:
            Rectification maps for both cameras
        """
        image_size = stereo_params.left_camera.image_size
        
        # Compute rectification transforms
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            stereo_params.left_camera.camera_matrix,
            stereo_params.left_camera.distortion_coeffs,
            stereo_params.right_camera.camera_matrix,
            stereo_params.right_camera.distortion_coeffs,
            image_size,
            stereo_params.rotation_matrix,
            stereo_params.translation_vector,
            alpha=0.0
        )
        
        # Compute rectification maps
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(
            stereo_params.left_camera.camera_matrix,
            stereo_params.left_camera.distortion_coeffs,
            R1, P1, image_size, cv2.CV_32FC1
        )
        
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(
            stereo_params.right_camera.camera_matrix,
            stereo_params.right_camera.distortion_coeffs,
            R2, P2, image_size, cv2.CV_32FC1
        )
        
        self.logger.info("Rectification maps computed successfully")
        
        return RectificationMaps(
            left_map_x=left_map_x,
            left_map_y=left_map_y,
            right_map_x=right_map_x,
            right_map_y=right_map_y,
            roi_left=roi1,
            roi_right=roi2
        )
    
    def rectify_image_pair(self, 
                          left_image: np.ndarray, 
                          right_image: np.ndarray,
                          rectification_maps: RectificationMaps) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify a stereo image pair using precomputed maps.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            rectification_maps: Precomputed rectification maps
            
        Returns:
            Tuple of (rectified_left, rectified_right)
        """
        # Apply rectification
        rectified_left = cv2.remap(
            left_image,
            rectification_maps.left_map_x,
            rectification_maps.left_map_y,
            cv2.INTER_LINEAR
        )
        
        rectified_right = cv2.remap(
            right_image,
            rectification_maps.right_map_x,
            rectification_maps.right_map_y,
            cv2.INTER_LINEAR
        )
        
        return rectified_left, rectified_right
    
    def validate_epipolar_alignment(self, 
                                   rectified_left: np.ndarray,
                                   rectified_right: np.ndarray,
                                   num_lines: int = 10) -> float:
        """
        Validate epipolar alignment by checking horizontal line correspondence.
        
        Args:
            rectified_left: Rectified left image
            rectified_right: Rectified right image
            num_lines: Number of horizontal lines to check
            
        Returns:
            Average vertical deviation in pixels
        """
        height, width = rectified_left.shape[:2]
        
        # Convert to grayscale if needed
        if len(rectified_left.shape) == 3:
            left_gray = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = rectified_left
            right_gray = rectified_right
        
        # Detect features on horizontal lines
        detector = cv2.ORB_create(nfeatures=500)
        
        total_deviation = 0.0
        valid_matches = 0
        
        # Check features along horizontal lines
        for i in range(num_lines):
            y = int((i + 1) * height / (num_lines + 1))
            
            # Create masks for horizontal strips
            mask = np.zeros_like(left_gray)
            mask[max(0, y-10):min(height, y+10), :] = 255
            
            # Detect keypoints
            kp1, desc1 = detector.detectAndCompute(left_gray, mask)
            kp2, desc2 = detector.detectAndCompute(right_gray, mask)
            
            if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
                # Match features
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(desc1, desc2)
                
                # Calculate vertical deviations
                for match in matches:
                    pt1 = kp1[match.queryIdx].pt
                    pt2 = kp2[match.trainIdx].pt
                    
                    vertical_deviation = abs(pt1[1] - pt2[1])
                    total_deviation += vertical_deviation
                    valid_matches += 1
        
        if valid_matches == 0:
            self.logger.warning("No valid matches found for epipolar validation")
            return float('inf')
        
        average_deviation = total_deviation / valid_matches
        self.logger.info(f"Epipolar alignment: {average_deviation:.2f} pixels average deviation")
        
        return average_deviation
    
    def _validate_stereo_geometry(self, R: np.ndarray, T: np.ndarray, baseline: float) -> None:
        """
        Validate stereo geometry parameters.
        
        Args:
            R: Rotation matrix between cameras
            T: Translation vector between cameras
            baseline: Distance between camera centers
        """
        # Check baseline is reasonable
        if baseline < self.min_baseline:
            raise ValueError(f"Baseline too small: {baseline:.4f}m < {self.min_baseline}m")
        
        if baseline > self.max_baseline:
            self.logger.warning(f"Large baseline: {baseline:.4f}m > {self.max_baseline}m")
        
        # Check rotation matrix is valid
        det_R = np.linalg.det(R)
        if abs(det_R - 1.0) > 0.01:
            raise ValueError(f"Invalid rotation matrix: det(R) = {det_R:.4f}")
        
        # Check if rotation is reasonable (not too large)
        rotation_angle = np.arccos((np.trace(R) - 1) / 2)
        rotation_degrees = np.degrees(rotation_angle)
        
        if rotation_degrees > 45:
            self.logger.warning(f"Large rotation between cameras: {rotation_degrees:.1f} degrees")
        
        self.logger.debug(f"Stereo geometry validation passed: baseline={baseline:.4f}m, rotation={rotation_degrees:.1f}°")
    
    def _calculate_stereo_reprojection_error(self,
                                           object_points: List[np.ndarray],
                                           left_image_points: List[np.ndarray],
                                           right_image_points: List[np.ndarray],
                                           left_params: CameraParameters,
                                           right_params: CameraParameters,
                                           R: np.ndarray,
                                           T: np.ndarray) -> float:
        """
        Calculate stereo reprojection error.
        
        Args:
            object_points: 3D object points
            left_image_points: Left image points
            right_image_points: Right image points
            left_params: Left camera parameters
            right_params: Right camera parameters
            R: Rotation matrix
            T: Translation vector
            
        Returns:
            RMS reprojection error in pixels
        """
        total_error = 0.0
        total_points = 0
        
        for i in range(len(object_points)):
            # Project to left camera
            left_projected, _ = cv2.projectPoints(
                object_points[i],
                np.zeros(3),  # No rotation for left camera (reference)
                np.zeros(3),  # No translation for left camera
                left_params.camera_matrix,
                left_params.distortion_coeffs
            )
            
            # Project to right camera
            right_projected, _ = cv2.projectPoints(
                object_points[i],
                R,  # Rotation relative to left camera
                T,  # Translation relative to left camera
                right_params.camera_matrix,
                right_params.distortion_coeffs
            )
            
            # Calculate errors
            left_error = cv2.norm(left_image_points[i], left_projected, cv2.NORM_L2)
            right_error = cv2.norm(right_image_points[i], right_projected, cv2.NORM_L2)
            
            n_points = len(object_points[i])
            total_error += (left_error + right_error)
            total_points += 2 * n_points
        
        return total_error / total_points if total_points > 0 else float('inf')