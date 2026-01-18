"""Point Cloud Generator - Placeholder"""
import numpy as np

class PointCloudGenerator:
    def __init__(self): pass
    def reproject_to_3d(self, disparity: np.ndarray, Q_matrix: np.ndarray) -> np.ndarray: 
        raise NotImplementedError("Will be implemented in task 6.1")
    def filter_depth_range(self, points: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray: 
        raise NotImplementedError("Will be implemented in task 6.1")