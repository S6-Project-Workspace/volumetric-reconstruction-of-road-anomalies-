"""
CharuCo Board Calibrator

Implements robust CharuCo board detection and camera calibration for metric accuracy.
CharuCo boards combine ChArUco markers with chessboard patterns for occlusion robustness.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import logging
from pathlib import Path

from ..data_models import CameraParameters
from ..utils.config_manager import ConfigManager


class CharuCoCalibrator:
    """Handles CharuCo board detection and corner refinement for metric calibration."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize CharuCo calibrator.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Get CharuCo configuration
        charuco_config = self.config.get_calibration_params()
        
        # Create CharuCo board
        self.squares_x = charuco_config.get('squares_x', 7)
        self.squares_y = charuco_config.get('squares_y', 5)
        self.square_length = charuco_config.get('square_length', 0.04)  # meters
        self.marker_length = charuco_config.get('marker_length', 0.02)  # meters
        
        # Create ArUco dictionary
        dict_name = charuco_config.get('dictionary', 'DICT_6X6_250')
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dict_name)
        )
        
        # Create CharuCo board
        self.charuco_board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict
        )
        
        # Detection parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 23
        self.detector_params.adaptiveThreshWinSizeStep = 10
        self.detector_params.minMarkerPerimeterRate = 0.03
        self.detector_params.maxMarkerPerimeterRate = 4.0
        
        # Create ArUco detector
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, 
            self.detector_params
        )
        
        # Minimum detections required for calibration
        self.min_detections = charuco_config.get('min_detections', 15)
        
        self.logger.info(f"CharuCo calibrator initialized: {self.squares_x}x{self.squares_y} board")
    
    def detect_corners(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect CharuCo corners in image with occlusion robustness.
        
        Args:
            image: Input grayscale or color image
            
        Returns:
            Tuple of (corners, ids) or (None, None) if detection fails
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        try:
            # Detect ArUco markers first
            marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)
            
            if len(marker_corners) == 0:
                self.logger.debug("No ArUco markers detected")
                return None, None
            
            # Interpolate CharuCo corners from detected markers
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.charuco_board
            )
            
            if ret < 4:  # Need at least 4 corners for pose estimation
                self.logger.debug(f"Insufficient CharuCo corners detected: {ret}")
                return None, None
            
            self.logger.debug(f"Detected {ret} CharuCo corners from {len(marker_corners)} markers")
            return charuco_corners, charuco_ids
            
        except Exception as e:
            self.logger.error(f"Error in CharuCo detection: {e}")
            return None, None
    
    def refine_corners(self, corners: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Refine corner positions to sub-pixel accuracy.
        
        Args:
            corners: Initial corner positions
            image: Input grayscale image
            
        Returns:
            Refined corner positions
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Sub-pixel corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Reshape corners for cornerSubPix
        corners_2d = corners.reshape(-1, 1, 2)
        
        refined_corners = cv2.cornerSubPix(
            gray, corners_2d, (11, 11), (-1, -1), criteria
        )
        
        return refined_corners.reshape(-1, 2)
    
    def collect_calibration_data(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Collect calibration data from multiple images.
        
        Args:
            images: List of calibration images
            
        Returns:
            Tuple of (all_corners, all_ids, object_points)
        """
        all_corners = []
        all_ids = []
        all_object_points = []
        
        valid_detections = 0
        
        for i, image in enumerate(images):
            corners, ids = self.detect_corners(image)
            
            if corners is not None and ids is not None:
                # Refine corners for better accuracy
                refined_corners = self.refine_corners(corners, image)
                
                # Get corresponding object points
                obj_points, _ = self.charuco_board.matchImagePoints(
                    refined_corners.reshape(-1, 1, 2), ids
                )
                
                if obj_points is not None and len(obj_points) >= 4:
                    all_corners.append(refined_corners.reshape(-1, 1, 2))
                    all_ids.append(ids)
                    all_object_points.append(obj_points)
                    valid_detections += 1
                    
                    self.logger.debug(f"Image {i+1}: {len(refined_corners)} corners detected")
                else:
                    self.logger.debug(f"Image {i+1}: Insufficient valid corners")
            else:
                self.logger.debug(f"Image {i+1}: No CharuCo detection")
        
        self.logger.info(f"Collected {valid_detections} valid detections from {len(images)} images")
        
        if valid_detections < self.min_detections:
            raise ValueError(
                f"Insufficient calibration data: {valid_detections} < {self.min_detections} required"
            )
        
        return all_corners, all_ids, all_object_points
    
    def calibrate_intrinsics(self, images: List[np.ndarray]) -> CameraParameters:
        """
        Calibrate camera intrinsic parameters using CharuCo board.
        
        Args:
            images: List of calibration images
            
        Returns:
            Camera parameters with quality metrics
        """
        if not images:
            raise ValueError("No calibration images provided")
        
        # Get image size
        if len(images[0].shape) == 3:
            image_size = (images[0].shape[1], images[0].shape[0])
        else:
            image_size = (images[0].shape[1], images[0].shape[0])
        
        self.logger.info(f"Starting intrinsic calibration with {len(images)} images")
        
        # Collect calibration data
        all_corners, all_ids, all_object_points = self.collect_calibration_data(images)
        
        # Prepare data for calibration
        object_points = []
        image_points = []
        
        for corners, ids, obj_pts in zip(all_corners, all_ids, all_object_points):
            object_points.append(obj_pts)
            image_points.append(corners)
        
        # Perform calibration
        calibration_flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS |
            cv2.CALIB_RATIONAL_MODEL |
            cv2.CALIB_THIN_PRISM_MODEL
        )
        
        # Initial guess for camera matrix
        focal_length_guess = max(image_size) * 0.8  # Reasonable initial guess
        camera_matrix_guess = np.array([
            [focal_length_guess, 0, image_size[0] / 2],
            [0, focal_length_guess, image_size[1] / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Initial distortion coefficients (5 parameters: k1, k2, p1, p2, k3)
        dist_coeffs_guess = np.zeros(5, dtype=np.float32)
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size,
            camera_matrix_guess, dist_coeffs_guess,
            flags=calibration_flags
        )
        
        if not ret:
            raise RuntimeError("Camera calibration failed")
        
        # Calculate reprojection error
        total_error = 0
        total_points = 0
        
        for i in range(len(object_points)):
            projected_points, _ = cv2.projectPoints(
                object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
            total_error += error * len(projected_points)
            total_points += len(projected_points)
        
        mean_error = total_error / total_points
        
        self.logger.info(f"Calibration completed: RMS error = {mean_error:.4f} pixels")
        
        # Validate calibration quality
        if mean_error > 0.5:
            self.logger.warning(f"High reprojection error: {mean_error:.4f} > 0.5 pixels")
        
        return CameraParameters(
            camera_matrix=camera_matrix,
            distortion_coeffs=dist_coeffs,
            reprojection_error=mean_error,
            image_size=image_size
        )
    
    def generate_charuco_board(self, output_path: str, dpi: int = 300) -> None:
        """
        Generate and save a CharuCo board for printing.
        
        Args:
            output_path: Path to save the board image
            dpi: Dots per inch for printing
        """
        # Calculate board size in pixels for given DPI
        board_width_inches = self.squares_x * self.square_length * 39.3701  # meters to inches
        board_height_inches = self.squares_y * self.square_length * 39.3701
        
        board_width_pixels = int(board_width_inches * dpi)
        board_height_pixels = int(board_height_inches * dpi)
        
        # Generate board image
        board_image = self.charuco_board.generateImage(
            (board_width_pixels, board_height_pixels), marginSize=50
        )
        
        # Save board
        cv2.imwrite(output_path, board_image)
        self.logger.info(f"CharuCo board saved to {output_path}")
        self.logger.info(f"Board size: {board_width_pixels}x{board_height_pixels} pixels at {dpi} DPI")
        self.logger.info(f"Physical size: {board_width_inches:.2f}x{board_height_inches:.2f} inches")
    
    def validate_detection_quality(self, image: np.ndarray, corners: np.ndarray, ids: np.ndarray) -> Dict[str, Any]:
        """
        Validate the quality of CharuCo detection.
        
        Args:
            image: Input image
            corners: Detected corners
            ids: Corner IDs
            
        Returns:
            Quality metrics dictionary
        """
        metrics = {
            'num_corners': len(corners) if corners is not None else 0,
            'num_markers': len(np.unique(ids // 4)) if ids is not None else 0,
            'coverage_ratio': 0.0,
            'corner_distribution': 'poor'
        }
        
        if corners is None or ids is None:
            return metrics
        
        # Calculate coverage ratio (corners detected vs total possible)
        total_corners = (self.squares_x - 1) * (self.squares_y - 1)
        metrics['coverage_ratio'] = len(corners) / total_corners
        
        # Analyze corner distribution across image
        if len(corners) > 4:
            image_height, image_width = image.shape[:2]
            
            # Divide image into quadrants and check distribution
            quadrant_counts = [0, 0, 0, 0]
            for corner in corners:
                x, y = corner[0], corner[1]
                if x < image_width / 2 and y < image_height / 2:
                    quadrant_counts[0] += 1  # Top-left
                elif x >= image_width / 2 and y < image_height / 2:
                    quadrant_counts[1] += 1  # Top-right
                elif x < image_width / 2 and y >= image_height / 2:
                    quadrant_counts[2] += 1  # Bottom-left
                else:
                    quadrant_counts[3] += 1  # Bottom-right
            
            # Check if corners are well distributed
            min_corners_per_quadrant = min(quadrant_counts)
            if min_corners_per_quadrant >= 2:
                metrics['corner_distribution'] = 'good'
            elif min_corners_per_quadrant >= 1:
                metrics['corner_distribution'] = 'moderate'
        
        return metrics