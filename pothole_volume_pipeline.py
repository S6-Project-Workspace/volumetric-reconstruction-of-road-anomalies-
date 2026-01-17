import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
import glob
import os
import scipy.io as sio

# =========================================================
# STEP 0: LOAD ALL STEREO IMAGE PAIRS (DATASET1)
# =========================================================

LEFT_IMAGES = sorted(glob.glob("data/dataset1/rgb/*_left.png"))

if len(LEFT_IMAGES) == 0:
    raise IOError("No stereo pairs found. Check naming convention.")

print(f"Found {len(LEFT_IMAGES)} stereo pairs\n")

# =========================================================
# CAMERA PARAMETERS (Unit 1: Calibration)
# These would typically come from cv2.calibrateCamera() and 
# cv2.stereoCalibrate() using Zhang's checkerboard method.
# Using approximate values for the dataset.
# =========================================================
# Focal length in pixels (typical for standard cameras)
focal_length = 721.5  # pixels
# Baseline (distance between stereo cameras) in meters
baseline = 0.54  # meters (typical stereo baseline)
# Principal point (image center)
cx = 609.5
cy = 172.8

# =========================================================
# PROCESS EACH STEREO PAIR
# =========================================================
for idx, left_path in enumerate(LEFT_IMAGES, start=1):

    right_path = left_path.replace("_left.png", "_right.png")

    left_img  = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    if left_img is None or right_img is None:
        print(f"Skipping pair {idx}: images not found")
        continue

    print(f"Processing stereo pair {idx}: {os.path.basename(left_path)}")
    
    h, w = left_img.shape[:2]
    
    # Update principal point to image center if not calibrated
    cx_used = w / 2.0
    cy_used = h / 2.0

    # =====================================================
    # Image Formation – Grayscale (Unit 1)
    # =====================================================
    left_gray  = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # =====================================================
    # Preprocessing (Image Processing)
    # =====================================================
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    left_gray  = clahe.apply(left_gray)
    right_gray = clahe.apply(right_gray)

    left_gray  = cv2.GaussianBlur(left_gray, (5, 5), 0)
    right_gray = cv2.GaussianBlur(right_gray, (5, 5), 0)

    # =====================================================
    # Dense Disparity (SGM – Unit 2 & 3)
    # Semi-Global Block Matching for dense correspondence
    # =====================================================
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # Reduced for better performance
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity_raw = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    
    # Create valid disparity mask (disparity > 0)
    valid_disp_mask = disparity_raw > 0
    
    print(
        "  Disparity stats:",
        "min =", np.min(disparity_raw[valid_disp_mask]) if np.any(valid_disp_mask) else 0,
        "max =", np.max(disparity_raw[valid_disp_mask]) if np.any(valid_disp_mask) else 0,
        "valid pixels =", np.sum(valid_disp_mask)
    )

    # =====================================================
    # 3D REPROJECTION using Q Matrix (Unit 3: Stereo)
    # Q matrix: Converts disparity to 3D coordinates
    # Z = f * B / d  (Depth formula from stereo geometry)
    # =====================================================
    # Build the Q (reprojection) matrix
    # Standard form from OpenCV stereoRectify:
    # [ 1  0  0   -cx        ]
    # [ 0  1  0   -cy        ]
    # [ 0  0  0    f         ]
    # [ 0  0 1/B   0         ]  (positive 1/B for correct depth sign)
    
    Q = np.float32([
        [1, 0, 0, -cx_used],
        [0, 1, 0, -cy_used],
        [0, 0, 0, focal_length],
        [0, 0, 1.0/baseline, 0]  # Positive to get positive depth
    ])

    # Reproject disparity to 3D points
    points_3D = cv2.reprojectImageTo3D(disparity_raw, Q, handleMissingValues=True)

    X = points_3D[:, :, 0]
    Y = points_3D[:, :, 1]
    Z = points_3D[:, :, 2]
    
    
    # ---------------------------------------------------------
# Clamp usable depth range (realistic road distances)
    # ---------------------------------------------------------
    MAX_DEPTH = 30.0   # meters (realistic for road scenes)
    MIN_DEPTH = 0.5    # meters (avoid very near noise)
    
    Z[(Z < MIN_DEPTH) | (Z > MAX_DEPTH)] = np.nan
    
    valid_mask = (valid_disp_mask &
              np.isfinite(Z))
    
    # Debug: Check actual depth range
    valid_disp_Z = Z[valid_disp_mask]
    finite_Z = valid_disp_Z[np.isfinite(valid_disp_Z)]
    
    
    if len(finite_Z) > 0:
        print(f"  Depth range: min={np.min(finite_Z):.2f}m, max={np.max(finite_Z):.2f}m")
    
    
    
    # Valid mask: finite values, positive depth, reasonable range
    MAX_DEPTH = 1000.0  # meters (relaxed for testing)
    MIN_DEPTH = 0.1     # meters
    valid_mask = (valid_disp_mask & 
                  np.isfinite(Z) & 
                  (Z > MIN_DEPTH) & 
                  (Z < MAX_DEPTH))

    num_valid = np.sum(valid_mask)
    print(f"  Valid 3D points: {num_valid}")

    XY = np.column_stack((X[valid_mask], Y[valid_mask]))
    Z_vals = Z[valid_mask]

    if num_valid < 1000:
        print("  → Skipping frame: insufficient valid 3D points\n")
        continue

    # =====================================================
    # GROUND PLANE ESTIMATION (RANSAC - Unit 3)
    # Fit a plane to the road surface using RANSAC
    # Plane equation: Z = aX + bY + c
    # =====================================================
    ransac = RANSACRegressor(residual_threshold=0.05, max_trials=1000)
    ransac.fit(XY, Z_vals)

    Z_plane = ransac.predict(XY)
    
    # Inlier mask from RANSAC (ground points)
    inlier_mask = ransac.inlier_mask_

    # =====================================================
    # POTHOLE SEGMENTATION (Morphological - Unit 1)
    # Points below the ground plane are potential potholes
    # Points above the ground plane are potential bumps/humps
    # =====================================================
    depth_difference = Z_vals - Z_plane  # Positive = deeper (pothole)
    
    # Threshold for pothole detection (in meters)
    POTHOLE_THRESHOLD = 0.02  # 2 cm below ground plane
    HUMP_THRESHOLD = 0.02     # 2 cm above ground plane
    
    pothole_mask = depth_difference > POTHOLE_THRESHOLD  # Deeper than plane
    hump_mask = depth_difference < -HUMP_THRESHOLD       # Higher than plane

    # =====================================================
    # VOLUMETRIC QUANTIFICATION (Unit 3: 3D Metrology)
    # Volume = sum of (depth_difference * pixel_area)
    # Pixel area in 3D depends on depth: area ≈ (Z/f)^2
    # =====================================================
    
    # Compute approximate area per point (in square meters)
    # At depth Z, pixel represents (Z/f)^2 square meters
    pixel_area = (Z_vals / focal_length) ** 2
    
    # Pothole volume (cubic meters)
    if np.any(pothole_mask):
        pothole_depths = depth_difference[pothole_mask]
        pothole_areas = pixel_area[pothole_mask]
        pothole_volume_m3 = np.sum(pothole_depths * pothole_areas)
        pothole_volume_cm3 = pothole_volume_m3 * 1e6  # Convert to cm³
        pothole_volume_liters = pothole_volume_m3 * 1000  # Convert to liters
        num_pothole_pts = np.sum(pothole_mask)
    else:
        pothole_volume_cm3 = 0
        pothole_volume_liters = 0
        num_pothole_pts = 0
    
    # Hump volume (cubic meters)
    if np.any(hump_mask):
        hump_heights = -depth_difference[hump_mask]  # Make positive
        hump_areas = pixel_area[hump_mask]
        hump_volume_m3 = np.sum(hump_heights * hump_areas)
        hump_volume_cm3 = hump_volume_m3 * 1e6
        num_hump_pts = np.sum(hump_mask)
    else:
        hump_volume_cm3 = 0
        num_hump_pts = 0

    # =====================================================
    # OUTPUT RESULTS
    # =====================================================
    print(f"  Ground plane inliers: {np.sum(inlier_mask)}")
    print(f"  Pothole points: {num_pothole_pts}")
    print(f"  Hump points: {num_hump_pts}")
    print(f"  → Pothole Volume: {pothole_volume_cm3:.2f} cm³ ({pothole_volume_liters:.4f} liters)")
    print(f"  → Hump Volume: {hump_volume_cm3:.2f} cm³\n")

print("=" * 60)
print("All stereo pairs processed successfully.")
print("=" * 60)