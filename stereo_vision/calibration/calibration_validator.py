"""
Calibration Quality Validator

Validates calibration quality metrics and ensures system meets accuracy requirements.
"""

import numpy as np
import logging
from typing import Dict, Any, List

from ..data_models import CameraParameters, StereoParameters


class CalibrationValidator:
    """Validates calibration quality metrics against system requirements."""
    
    def __init__(self):
        """Initialize calibration validator."""
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds based on requirements
        self.max_reprojection_error = 0.1  # pixels (requirement 1.2)
        self.min_focal_length = 300  # pixels (reasonable for most cameras)
        self.max_focal_length = 3000  # pixels
        self.max_distortion_k1 = 1.0  # reasonable radial distortion
        self.min_baseline = 0.05  # meters
        self.max_baseline = 2.0  # meters
        self.max_rotation_degrees = 45  # degrees between cameras
        
        self.logger.info("Calibration validator initialized")
    
    def validate_intrinsic_calibration(self, params: CameraParameters) -> Dict[str, Any]:
        """
        Validate intrinsic calibration quality against requirements.
        
        Args:
            params: Camera parameters to validate
            
        Returns:
            Dictionary with validation results and quality metrics
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 0.0,
            'metrics': {}
        }
        
        # Check reprojection error (critical requirement)
        results['metrics']['reprojection_error'] = params.reprojection_error
        if params.reprojection_error > self.max_reprojection_error:
            results['is_valid'] = False
            results['errors'].append(
                f"Reprojection error {params.reprojection_error:.4f} > {self.max_reprojection_error} pixels"
            )
        
        # Validate camera matrix
        fx = params.camera_matrix[0, 0]
        fy = params.camera_matrix[1, 1]
        cx = params.camera_matrix[0, 2]
        cy = params.camera_matrix[1, 2]
        
        results['metrics']['focal_length_x'] = fx
        results['metrics']['focal_length_y'] = fy
        results['metrics']['principal_point'] = (cx, cy)
        
        # Check focal lengths are reasonable
        if fx < self.min_focal_length or fx > self.max_focal_length:
            results['warnings'].append(f"Unusual focal length fx: {fx:.1f} pixels")
        
        if fy < self.min_focal_length or fy > self.max_focal_length:
            results['warnings'].append(f"Unusual focal length fy: {fy:.1f} pixels")
        
        # Check aspect ratio (should be close to 1.0)
        aspect_ratio = fx / fy
        results['metrics']['aspect_ratio'] = aspect_ratio
        if abs(aspect_ratio - 1.0) > 0.1:
            results['warnings'].append(f"Unusual aspect ratio: {aspect_ratio:.3f}")
        
        # Check principal point is near image center
        image_center_x = params.image_size[0] / 2
        image_center_y = params.image_size[1] / 2
        
        cx_offset = abs(cx - image_center_x) / image_center_x
        cy_offset = abs(cy - image_center_y) / image_center_y
        
        results['metrics']['principal_point_offset'] = (cx_offset, cy_offset)
        
        if cx_offset > 0.2 or cy_offset > 0.2:
            results['warnings'].append(
                f"Principal point far from center: ({cx_offset:.2%}, {cy_offset:.2%})"
            )
        
        # Validate distortion coefficients
        if len(params.distortion_coeffs) >= 5:
            k1, k2, p1, p2, k3 = params.distortion_coeffs[:5]
            
            results['metrics']['distortion_k1'] = k1
            results['metrics']['distortion_k2'] = k2
            
            # Check radial distortion is reasonable
            if abs(k1) > self.max_distortion_k1:
                results['warnings'].append(f"High radial distortion k1: {k1:.4f}")
            
            if abs(k2) > self.max_distortion_k1:
                results['warnings'].append(f"High radial distortion k2: {k2:.4f}")
        
        # Calculate quality score (0-100)
        quality_score = 100.0
        
        # Penalize high reprojection error
        if params.reprojection_error > 0.05:
            quality_score -= (params.reprojection_error - 0.05) * 200
        
        # Penalize unusual parameters
        quality_score -= len(results['warnings']) * 5
        quality_score -= len(results['errors']) * 20
        
        results['quality_score'] = max(0.0, min(100.0, quality_score))
        
        # Log results
        if results['is_valid']:
            self.logger.info(f"Intrinsic calibration valid: quality={results['quality_score']:.1f}%")
        else:
            self.logger.error(f"Intrinsic calibration invalid: {len(results['errors'])} errors")
        
        for warning in results['warnings']:
            self.logger.warning(warning)
        
        return results
    
    def validate_stereo_calibration(self, params: StereoParameters) -> Dict[str, Any]:
        """
        Validate stereo calibration quality against requirements.
        
        Args:
            params: Stereo parameters to validate
            
        Returns:
            Dictionary with validation results and quality metrics
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 0.0,
            'metrics': {}
        }
        
        # Validate individual camera calibrations first
        left_validation = self.validate_intrinsic_calibration(params.left_camera)
        right_validation = self.validate_intrinsic_calibration(params.right_camera)
        
        if not left_validation['is_valid']:
            results['is_valid'] = False
            results['errors'].append("Left camera calibration invalid")
        
        if not right_validation['is_valid']:
            results['is_valid'] = False
            results['errors'].append("Right camera calibration invalid")
        
        # Validate baseline
        results['metrics']['baseline'] = params.baseline
        if params.baseline < self.min_baseline:
            results['errors'].append(f"Baseline too small: {params.baseline:.4f}m < {self.min_baseline}m")
            results['is_valid'] = False
        
        if params.baseline > self.max_baseline:
            results['warnings'].append(f"Large baseline: {params.baseline:.4f}m")
        
        # Validate rotation matrix
        R = params.rotation_matrix
        det_R = np.linalg.det(R)
        results['metrics']['rotation_determinant'] = det_R
        
        if abs(det_R - 1.0) > 0.01:
            results['errors'].append(f"Invalid rotation matrix: det(R) = {det_R:.4f}")
            results['is_valid'] = False
        
        # Calculate rotation angle
        try:
            rotation_angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
            rotation_degrees = np.degrees(rotation_angle)
            results['metrics']['rotation_angle_degrees'] = rotation_degrees
            
            if rotation_degrees > self.max_rotation_degrees:
                results['warnings'].append(f"Large rotation: {rotation_degrees:.1f}°")
        except:
            results['errors'].append("Failed to calculate rotation angle")
            results['is_valid'] = False
        
        # Validate Q matrix
        Q = params.Q_matrix
        if Q is None or Q.shape != (4, 4):
            results['errors'].append("Invalid Q matrix")
            results['is_valid'] = False
        else:
            # Check Q matrix elements are reasonable
            focal_length_from_Q = Q[2, 3]
            baseline_from_Q = -1.0 / Q[3, 2] if Q[3, 2] != 0 else 0
            
            results['metrics']['focal_length_from_Q'] = focal_length_from_Q
            results['metrics']['baseline_from_Q'] = baseline_from_Q
            
            # Consistency check
            if abs(baseline_from_Q - params.baseline) > 0.01:
                results['warnings'].append(
                    f"Q matrix baseline inconsistency: {baseline_from_Q:.4f} vs {params.baseline:.4f}"
                )
        
        # Calculate stereo quality score
        quality_score = (left_validation['quality_score'] + right_validation['quality_score']) / 2
        
        # Adjust for stereo-specific factors
        if params.baseline > 0.3:  # Good baseline for depth accuracy
            quality_score += 5
        
        if 'rotation_angle_degrees' in results['metrics']:
            if results['metrics']['rotation_angle_degrees'] < 10:  # Well-aligned cameras
                quality_score += 5
        
        # Penalize errors and warnings
        quality_score -= len(results['warnings']) * 3
        quality_score -= len(results['errors']) * 15
        
        results['quality_score'] = max(0.0, min(100.0, quality_score))
        
        # Log results
        if results['is_valid']:
            self.logger.info(f"Stereo calibration valid: quality={results['quality_score']:.1f}%")
        else:
            self.logger.error(f"Stereo calibration invalid: {len(results['errors'])} errors")
        
        for warning in results['warnings']:
            self.logger.warning(warning)
        
        return results
    
    def validate_rectification_quality(self, 
                                     epipolar_deviation: float,
                                     roi_left: tuple,
                                     roi_right: tuple,
                                     image_size: tuple) -> Dict[str, Any]:
        """
        Validate rectification quality.
        
        Args:
            epipolar_deviation: Average vertical deviation in pixels
            roi_left: Left image ROI after rectification
            roi_right: Right image ROI after rectification  
            image_size: Original image size
            
        Returns:
            Validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check epipolar alignment
        results['metrics']['epipolar_deviation'] = epipolar_deviation
        if epipolar_deviation > 2.0:  # Should be < 2 pixels for good rectification
            results['errors'].append(f"Poor epipolar alignment: {epipolar_deviation:.2f} pixels")
            results['is_valid'] = False
        elif epipolar_deviation > 1.0:
            results['warnings'].append(f"Moderate epipolar deviation: {epipolar_deviation:.2f} pixels")
        
        # Check ROI coverage (how much of the image is valid after rectification)
        left_coverage = (roi_left[2] * roi_left[3]) / (image_size[0] * image_size[1])
        right_coverage = (roi_right[2] * roi_right[3]) / (image_size[0] * image_size[1])
        
        results['metrics']['left_roi_coverage'] = left_coverage
        results['metrics']['right_roi_coverage'] = right_coverage
        
        min_coverage = min(left_coverage, right_coverage)
        if min_coverage < 0.5:  # Less than 50% coverage
            results['warnings'].append(f"Low ROI coverage: {min_coverage:.1%}")
        
        self.logger.info(f"Rectification validation: deviation={epipolar_deviation:.2f}px, coverage={min_coverage:.1%}")
        
        return results
    
    def generate_calibration_report(self, 
                                  stereo_params: StereoParameters,
                                  validation_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive calibration quality report.
        
        Args:
            stereo_params: Stereo calibration parameters
            validation_results: Validation results
            
        Returns:
            Formatted calibration report
        """
        report = []
        report.append("=" * 60)
        report.append("STEREO CALIBRATION QUALITY REPORT")
        report.append("=" * 60)
        
        # Overall status
        status = "✅ VALID" if validation_results['is_valid'] else "❌ INVALID"
        report.append(f"Status: {status}")
        report.append(f"Quality Score: {validation_results['quality_score']:.1f}/100")
        report.append("")
        
        # Camera parameters
        report.append("📷 CAMERA PARAMETERS")
        report.append("-" * 30)
        
        left_fx = stereo_params.left_camera.camera_matrix[0, 0]
        left_fy = stereo_params.left_camera.camera_matrix[1, 1]
        right_fx = stereo_params.right_camera.camera_matrix[0, 0]
        right_fy = stereo_params.right_camera.camera_matrix[1, 1]
        
        report.append(f"Left focal length:  {left_fx:.1f} x {left_fy:.1f} pixels")
        report.append(f"Right focal length: {right_fx:.1f} x {right_fy:.1f} pixels")
        report.append(f"Left reprojection error:  {stereo_params.left_camera.reprojection_error:.4f} pixels")
        report.append(f"Right reprojection error: {stereo_params.right_camera.reprojection_error:.4f} pixels")
        report.append("")
        
        # Stereo geometry
        report.append("📐 STEREO GEOMETRY")
        report.append("-" * 30)
        report.append(f"Baseline: {stereo_params.baseline:.4f} meters")
        
        if 'rotation_angle_degrees' in validation_results['metrics']:
            rotation = validation_results['metrics']['rotation_angle_degrees']
            report.append(f"Camera rotation: {rotation:.2f} degrees")
        
        report.append("")
        
        # Quality assessment
        if validation_results['errors']:
            report.append("❌ ERRORS")
            report.append("-" * 30)
            for error in validation_results['errors']:
                report.append(f"  • {error}")
            report.append("")
        
        if validation_results['warnings']:
            report.append("⚠️  WARNINGS")
            report.append("-" * 30)
            for warning in validation_results['warnings']:
                report.append(f"  • {warning}")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 30)
        
        if validation_results['quality_score'] < 70:
            report.append("  • Consider recalibrating with more images")
            report.append("  • Ensure calibration board is perfectly flat")
            report.append("  • Use better lighting conditions")
        
        if validation_results['quality_score'] >= 90:
            report.append("  • Excellent calibration quality!")
            report.append("  • Ready for high-precision measurements")
        
        report.append("=" * 60)
        
        return "\n".join(report)