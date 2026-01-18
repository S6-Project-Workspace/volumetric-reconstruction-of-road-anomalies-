"""Ground Plane Model - Placeholder"""
import numpy as np
from typing import Tuple

class GroundPlaneModel:
    def __init__(self): pass
    def fit_from_v_disparity(self, v_disp: np.ndarray) -> None: 
        raise NotImplementedError("Will be implemented in task 5.3")
    def get_expected_disparity(self, row: int) -> float: 
        raise NotImplementedError("Will be implemented in task 5.3")
    def segment_anomalies(self, disparity_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
        raise NotImplementedError("Will be implemented in task 5.5")