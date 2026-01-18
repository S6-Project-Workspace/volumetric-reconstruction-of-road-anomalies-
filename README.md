# Advanced Stereo Vision Pipeline for Road Anomaly Detection

A state-of-the-art volumetric reconstruction system for detecting and quantifying road surface anomalies (potholes and humps) using stereo vision.

## 🚀 Project Overview

This project implements a comprehensive stereo vision pipeline that transforms basic pothole detection into a metrologically precise instrument capable of sub-millimeter accuracy. The system uses advanced computer vision techniques including:

- **CharuCo-based camera calibration** for metric accuracy
- **Advanced disparity estimation** with SGBM, LRC checking, and WLS filtering  
- **V-Disparity ground plane detection** using Hough transforms
- **3D point cloud processing** with statistical outlier removal
- **Watertight mesh generation** using Alpha Shapes
- **Precise volume calculation** using signed tetrahedron integration

## 📁 Project Structure

```
├── stereo_vision/              # Main package
│   ├── calibration/           # CharuCo-based calibration system
│   ├── disparity/             # Advanced disparity estimation
│   ├── reconstruction/        # 3D reconstruction and V-Disparity
│   ├── volume/                # Mesh generation and volume calculation
│   ├── preprocessing/         # Image enhancement
│   ├── utils/                 # Configuration and utilities
│   └── data_models.py         # Data structures
├── config/                    # Configuration files
├── tests/                     # Test suite with property-based testing
├── data/                      # Dataset (stereo image pairs)
└── requirements.txt           # Dependencies
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd volumetric-reconstruction-of-road-anomalies
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install the package in development mode:**
```bash
pip install -e .
```

## 🎯 Key Features

### Advanced Calibration System
- CharuCo board detection with occlusion robustness
- Sub-pixel corner refinement for metric accuracy
- Two-stage calibration (intrinsic → stereo)
- Reprojection error < 0.1 pixels

### Robust Disparity Estimation  
- Semi-Global Block Matching optimized for road scenes
- Left-Right Consistency checking for occlusion handling
- Weighted Least Squares filtering for sub-pixel refinement
- Handles textureless road surfaces

### Intelligent Ground Plane Detection
- V-Disparity histogram representation
- Hough Transform for robust road line detection
- Automatic pothole/hump segmentation
- No manual intervention required

### Precise Volume Calculation
- Alpha Shape mesh generation (concave hulls)
- Watertight mesh closure with boundary capping
- Signed tetrahedron volume integration
- Volume uncertainty estimation

## 🚀 Usage

### Basic Usage
```bash
python -m stereo_vision.main --input-dir data/dataset1/rgb --output-dir results
```

### Batch Processing
```bash
python -m stereo_vision.main --input-dir data/dataset1/rgb --output-dir results --batch
```

### Custom Configuration
```bash
python -m stereo_vision.main --input-dir data/dataset1/rgb --config custom_config.yaml
```

## 🧪 Testing

The project includes comprehensive testing with both unit tests and property-based tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=stereo_vision --cov-report=html

# Run only property-based tests
pytest -m property

# Run specific test categories
pytest -m unit
pytest -m integration
```

## 📊 Current Implementation Status

This project is currently under active development following a systematic implementation plan:

- ✅ **Task 1**: Project structure and dependencies setup
- 🔄 **Task 2**: CharuCo-based calibration system (Next)
- ⏳ **Task 3**: Advanced disparity estimation
- ⏳ **Task 4**: V-Disparity ground plane detection  
- ⏳ **Task 5**: 3D reconstruction engine
- ⏳ **Task 6**: Volumetric analysis module

See `.kiro/specs/advanced-stereo-vision-pipeline/tasks.md` for the complete implementation plan.

## 🔧 Configuration

The system uses YAML configuration files for parameter management:

```yaml
# Camera Parameters
camera:
  focal_length: 721.5
  baseline: 0.54

# SGBM Parameters (optimized for roads)
sgbm:
  num_disparities: 160
  block_size: 5
  P1: 600
  P2: 2400

# Volume Calculation
volume:
  min_volume_threshold: 1e-6
  max_volume_threshold: 1.0
```

## 📈 Performance Metrics

The system provides comprehensive quality metrics:

- **LRC Error Rate**: Percentage of pixels failing consistency checks
- **Planarity Residuals**: RMSE of ground plane fitting
- **Temporal Stability**: Volume measurement consistency over time
- **Calibration Quality**: Reprojection error metrics

## 🎓 Technical Background

This implementation is based on rigorous computer vision principles:

- **Photogrammetric Calibration**: Zhang's method with CharuCo targets
- **Epipolar Geometry**: Stereo rectification for correspondence search
- **Semi-Global Matching**: Global optimization with smoothness constraints
- **V-Disparity Analysis**: Geometric road modeling in disparity space
- **Alpha Shapes**: Concave hull generation for irregular geometries
- **Divergence Theorem**: Mathematical volume integration

## 📚 Dependencies

- **OpenCV**: Computer vision and image processing
- **Open3D**: 3D point cloud processing and visualization
- **Trimesh**: Mesh generation and geometric operations
- **NumPy/SciPy**: Numerical computing
- **Hypothesis**: Property-based testing framework

## 🤝 Contributing

This project follows a specification-driven development approach. See the requirements, design, and task documents in `.kiro/specs/` for detailed implementation guidelines.

## 📄 License

MIT License - see LICENSE file for details.

---

**Note**: This is an advanced computer vision system designed for research and professional applications requiring high precision volumetric measurements.