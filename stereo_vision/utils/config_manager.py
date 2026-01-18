"""
Configuration Management System

Handles loading, validation, and management of system parameters.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration parameters for the stereo vision pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._validate_config()
    
    def _get_default_config_path(self) -> str:
        """Get path to default configuration file."""
        current_dir = Path(__file__).parent.parent.parent
        return str(current_dir / "config" / "default_config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _validate_config(self) -> None:
        """Validate configuration parameters for consistency and feasibility."""
        # Validate SGBM parameters
        sgbm = self.config.get('sgbm', {})
        num_disparities = sgbm.get('num_disparities', 160)
        if num_disparities % 16 != 0:
            raise ValueError("SGBM num_disparities must be divisible by 16")
        
        # Validate depth ranges
        pc = self.config.get('point_cloud', {})
        min_depth = pc.get('min_depth', 0.5)
        max_depth = pc.get('max_depth', 30.0)
        if min_depth >= max_depth:
            raise ValueError("min_depth must be less than max_depth")
        
        # Validate thresholds
        gp = self.config.get('ground_plane', {})
        pothole_thresh = gp.get('pothole_threshold', 0.02)
        hump_thresh = gp.get('hump_threshold', 0.02)
        if pothole_thresh <= 0 or hump_thresh <= 0:
            raise ValueError("Anomaly thresholds must be positive")
        
        # Validate volume constraints
        vol = self.config.get('volume', {})
        min_vol = float(vol.get('min_volume_threshold', 1e-6))
        max_vol = float(vol.get('max_volume_threshold', 1.0))
        if min_vol >= max_vol:
            raise ValueError("min_volume_threshold must be less than max_volume_threshold")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'sgbm.block_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'sgbm.block_size')
            value: Value to set
        """
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
        self._validate_config()
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration. If None, overwrites current file.
        """
        save_path = output_path or self.config_path
        
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
    
    def update_camera_parameters(self, baseline: float, focal_length: float) -> None:
        """
        Update camera parameters and recalculate dependent values.
        
        Args:
            baseline: Camera baseline in meters
            focal_length: Focal length in pixels
        """
        self.set('camera.baseline', baseline)
        self.set('camera.focal_length', focal_length)
        
        # Update SGBM parameters based on new camera geometry
        block_size = self.get('sgbm.block_size', 5)
        self.set('sgbm.P1', 8 * 3 * block_size**2)
        self.set('sgbm.P2', 32 * 3 * block_size**2)
    
    def get_sgbm_params(self) -> Dict[str, Any]:
        """Get SGBM parameters as a dictionary."""
        return self.config.get('sgbm', {})
    
    def get_calibration_params(self) -> Dict[str, Any]:
        """Get calibration parameters as a dictionary."""
        return self.config.get('charuco', {})
    
    def get_processing_params(self) -> Dict[str, Any]:
        """Get point cloud processing parameters as a dictionary."""
        return self.config.get('point_cloud', {})
    
    def get_lrc_params(self) -> Dict[str, Any]:
        """Get Left-Right Consistency parameters as a dictionary."""
        return self.config.get('lrc', {})
    
    def get_wls_params(self) -> Dict[str, Any]:
        """Get WLS filtering parameters as a dictionary."""
        return self.config.get('wls', {})