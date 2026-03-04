# Final Presentation Results
Generated: 2026-03-04 21:29:48

## Overview
Comprehensive processing results for 5 test cases demonstrating the complete stereo vision pipeline.

## Examples

### Example 1
Regular Example 1 - Standard Pothole

**Directory:** `example_1/`

**Files:**
- `00_SUMMARY.png` - Complete overview of all stages
- `01_input_*.png` - Original stereo pair
- `02_preprocessed_*.png` - After preprocessing
- `03_disparity_map.png` - Computed disparity
- `04_v_disparity.png` - V-disparity for ground plane
- `05_point_cloud.png` - 3D reconstruction
- `06_anomaly_mask.png` - Detected anomalies
- `07-08_anomaly_*` - Individual anomaly visualizations
- `results.json` - Detailed numerical results

### Example 2
Regular Example 2 - Large Anomaly

**Directory:** `example_2/`

**Files:**
- `00_SUMMARY.png` - Complete overview of all stages
- `01_input_*.png` - Original stereo pair
- `02_preprocessed_*.png` - After preprocessing
- `03_disparity_map.png` - Computed disparity
- `04_v_disparity.png` - V-disparity for ground plane
- `05_point_cloud.png` - 3D reconstruction
- `06_anomaly_mask.png` - Detected anomalies
- `07-08_anomaly_*` - Individual anomaly visualizations
- `results.json` - Detailed numerical results

### Edge Case 1
Edge Case 1 - Night Time (Low Light)

**Directory:** `edge_case_1/`

**Files:**
- `00_SUMMARY.png` - Complete overview of all stages
- `01_input_*.png` - Original stereo pair
- `02_preprocessed_*.png` - After preprocessing
- `03_disparity_map.png` - Computed disparity
- `04_v_disparity.png` - V-disparity for ground plane
- `05_point_cloud.png` - 3D reconstruction
- `06_anomaly_mask.png` - Detected anomalies
- `07-08_anomaly_*` - Individual anomaly visualizations
- `results.json` - Detailed numerical results

### Edge Case 2
Edge Case 2 - Night Time with Noise

**Directory:** `edge_case_2/`

**Files:**
- `00_SUMMARY.png` - Complete overview of all stages
- `01_input_*.png` - Original stereo pair
- `02_preprocessed_*.png` - After preprocessing
- `03_disparity_map.png` - Computed disparity
- `04_v_disparity.png` - V-disparity for ground plane
- `05_point_cloud.png` - 3D reconstruction
- `06_anomaly_mask.png` - Detected anomalies
- `07-08_anomaly_*` - Individual anomaly visualizations
- `results.json` - Detailed numerical results

### Edge Case 3
Edge Case 3 - Extreme Shadows

**Directory:** `edge_case_3/`

**Files:**
- `00_SUMMARY.png` - Complete overview of all stages
- `01_input_*.png` - Original stereo pair
- `02_preprocessed_*.png` - After preprocessing
- `03_disparity_map.png` - Computed disparity
- `04_v_disparity.png` - V-disparity for ground plane
- `05_point_cloud.png` - 3D reconstruction
- `06_anomaly_mask.png` - Detected anomalies
- `07-08_anomaly_*` - Individual anomaly visualizations
- `results.json` - Detailed numerical results

## Processing Pipeline

The pipeline consists of the following stages:

1. **Preprocessing**: Contrast enhancement (CLAHE), brightness normalization, noise filtering
2. **Disparity Estimation**: SGBM stereo matching with LRC validation
3. **V-Disparity**: Ground plane detection using V-disparity analysis
4. **3D Reconstruction**: Point cloud generation with outlier removal
5. **Anomaly Segmentation**: Detection of potholes and humps
6. **Volume Calculation**: Alpha shape meshing and volume computation

## Edge Cases

- **Edge Case 1**: Night-time simulation (25% brightness reduction)
- **Edge Case 2**: Night-time with sensor noise (Gaussian noise added)
- **Edge Case 3**: Extreme diagonal shadows

## Results Format

Each example directory contains:
- Visual outputs for each processing stage
- JSON file with numerical results including volumes, areas, and point counts
- Summary visualization combining all stages

## Usage

1. View `00_SUMMARY.png` for a quick overview
2. Check `results.json` for detailed metrics
3. Examine individual stage outputs for debugging

## Metrics

Results include:
- Disparity statistics (min, max, mean)
- Point cloud size and depth range
- Number of detected anomalies
- Volume measurements (m³, liters, cm³)
- Surface area estimates (m²)
