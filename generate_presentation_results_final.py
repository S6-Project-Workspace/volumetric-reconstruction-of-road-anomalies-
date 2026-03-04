#!/usr/bin/env python3
"""
Generate final presentation results with all processing stages visualized.
Based on the working Gradio app implementation.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

from stereo_vision.pipeline import StereoVisionPipeline
from stereo_vision.config import PipelineConfig, CameraConfig
from stereo_vision.calibration import CameraParameters, StereoParameters
from stereo_vision.ground_plane import VDisparityGenerator, HoughLineDetector, GroundPlaneModel
from stereo_vision.reconstruction import PointCloudGenerator, OutlierRemover
from stereo_vision.volumetric import AlphaShapeGenerator, MeshCapper, VolumeCalculator


class PresentationResultsGenerator:
    def __init__(self, output_dir="final_presentation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.examples = {
            "example_1": "Regular Example 1 - Standard Pothole",
            "example_2": "Regular Example 2 - Large Anomaly",
            "edge_case_1": "Edge Case 1 - Night Time (Low Light)",
            "edge_case_2": "Edge Case 2 - Night Time with Noise",
            "edge_case_3": "Edge Case 3 - Extreme Shadows"
        }
        
        for key in self.examples.keys():
            (self.output_dir / key).mkdir(exist_ok=True)
    
    def simulate_night_time(self, image, brightness_factor=0.3, add_noise=False):
        """Convert daytime image to night-time appearance"""
        night_img = (image * brightness_factor).astype(np.uint8)
        
        night_img = cv2.cvtColor(night_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        night_img[:, :, 0] = np.clip(night_img[:, :, 0] + 10, 0, 179)
        night_img[:, :, 1] = np.clip(night_img[:, :, 1] * 0.7, 0, 255)
        night_img = cv2.cvtColor(night_img.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        if add_noise:
            noise = np.random.normal(0, 15, night_img.shape).astype(np.int16)
            night_img = np.clip(night_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return night_img
    
    def add_extreme_shadows(self, image):
        """Add extreme shadow effects"""
        h, w = image.shape[:2]
        shadow_mask = np.ones((h, w), dtype=np.float32)
        
        for i in range(h):
            shadow_mask[i, :int(w * i / h)] *= 0.3
        
        shadowed = image.copy().astype(np.float32)
        for c in range(3):
            shadowed[:, :, c] *= shadow_mask
        
        return np.clip(shadowed, 0, 255).astype(np.uint8)
    
    def render_3d_scatter(self, points, title, output_path, cmap='viridis', max_points=10000):
        """Render 3D scatter plot"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
        
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=points[:, 2], cmap=cmap, s=1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_example(self, left_img, right_img, example_name, description,
                       baseline=0.12, focal_length=700.0):
        """Process a single example through the pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing: {description}")
        print(f"{'='*60}")
        
        example_dir = self.output_dir / example_name
        results = {'description': description}
        
        # Save input images
        cv2.imwrite(str(example_dir / "01_input_left.png"), left_img)
        cv2.imwrite(str(example_dir / "01_input_right.png"), right_img)
        
        try:
            # Convert to grayscale
            if len(left_img.shape) == 3:
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_img
                right_gray = right_img
            
            h, w = left_gray.shape
            
            # Setup pipeline
            camera_config = CameraConfig(baseline=baseline, focal_length=focal_length)
            config = PipelineConfig(camera=camera_config)
            config.wls.enabled = False
            pipeline = StereoVisionPipeline(config)
            
            # Create calibration
            camera_matrix = np.array([
                [focal_length, 0, w/2],
                [0, focal_length, h/2],
                [0, 0, 1]
            ], dtype=np.float32)
            
            distortion = np.zeros(5, dtype=np.float32)
            
            left_cam = CameraParameters(
                camera_matrix=camera_matrix,
                distortion_coeffs=distortion,
                reprojection_error=0.0,
                image_size=(w, h)
            )
            right_cam = CameraParameters(
                camera_matrix=camera_matrix,
                distortion_coeffs=distortion,
                reprojection_error=0.0,
                image_size=(w, h)
            )
            
            Q = np.array([
                [1, 0, 0, -w/2],
                [0, 1, 0, -h/2],
                [0, 0, 0, focal_length],
                [0, 0, 1/baseline, 0]
            ], dtype=np.float32)
            
            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)
            for y_coord in range(h):
                for x_coord in range(w):
                    map_x[y_coord, x_coord] = x_coord
                    map_y[y_coord, x_coord] = y_coord
            
            stereo_params = StereoParameters(
                left_camera=left_cam,
                right_camera=right_cam,
                rotation_matrix=np.eye(3, dtype=np.float32),
                translation_vector=np.array([[baseline], [0], [0]], dtype=np.float32),
                baseline=baseline,
                Q_matrix=Q,
                rectification_maps_left=(map_x, map_y),
                rectification_maps_right=(map_x, map_y)
            )
            
            pipeline.stereo_params = stereo_params
            pipeline.is_calibrated = True
            
            pc_gen = PointCloudGenerator(
                Q_matrix=Q,
                min_depth=config.depth_range.min_depth,
                max_depth=config.depth_range.max_depth
            )
            pipeline.point_cloud_generator = pc_gen
            
            # Stage 1: Preprocessing
            print("  Stage 1: Preprocessing...")
            left_proc, right_proc = pipeline.preprocessor.preprocess_stereo_pair(
                left_gray, right_gray
            )
            cv2.imwrite(str(example_dir / "02_preprocessed_left.png"), left_proc)
            cv2.imwrite(str(example_dir / "02_preprocessed_right.png"), right_proc)
            
            # Stage 2: Disparity
            print("  Stage 2: Disparity Estimation...")
            disp_left = pipeline.sgbm_estimator.compute_disparity(left_proc, right_proc)
            disp_right = pipeline.sgbm_estimator.compute_disparity(right_proc, left_proc)
            
            if config.lrc.enabled:
                validity_mask = pipeline.lrc_validator.validate_consistency(disp_left, disp_right)
            else:
                validity_mask = np.ones(disp_left.shape, dtype=np.uint8)
            
            disparity_map = disp_left.astype(np.float32) / 16.0
            disparity_map[validity_mask == 0] = 0
            
            disp_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            disp_colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
            cv2.imwrite(str(example_dir / "03_disparity_map.png"), disp_colored)
            
            valid_disp = disparity_map[disparity_map > 0]
            if len(valid_disp) > 0:
                results['disparity'] = {
                    'min': float(valid_disp.min()),
                    'max': float(valid_disp.max()),
                    'mean': float(valid_disp.mean())
                }
            
            # Stage 3: V-Disparity
            print("  Stage 3: V-Disparity Generation...")
            v_disp_gen = VDisparityGenerator()
            v_disparity = v_disp_gen.generate_v_disparity(disparity_map)
            
            if v_disparity is not None:
                v_vis = cv2.normalize(v_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                v_disp_colored = cv2.applyColorMap(v_vis, cv2.COLORMAP_HOT)
                cv2.imwrite(str(example_dir / "04_v_disparity.png"), v_disp_colored)
            
            # Stage 4: Point Cloud
            print("  Stage 4: 3D Reconstruction...")
            full_points = pc_gen.reproject_to_3d(disparity_map, apply_depth_filter=True)
            
            if len(full_points) > 0:
                outlier_remover = OutlierRemover(
                    k_neighbors=config.outlier_removal.k_neighbors,
                    std_ratio=config.outlier_removal.std_ratio
                )
                full_points = outlier_remover.remove_statistical_outliers(full_points)
                
                self.render_3d_scatter(
                    full_points[:, :3],
                    '3D Point Cloud',
                    example_dir / "05_point_cloud.png"
                )
                
                results['point_cloud'] = {
                    'num_points': len(full_points),
                    'depth_range': [float(full_points[:, 2].min()), float(full_points[:, 2].max())]
                }
            
            # Stage 5: Ground Plane & Anomalies
            print("  Stage 5: Ground Plane Detection...")
            ground_plane_ok = False
            anomalies = []
            
            try:
                hough = HoughLineDetector()
                gp_model = GroundPlaneModel()
                fitted = gp_model.fit_from_v_disparity(v_disparity, hough)
                
                if fitted:
                    ground_plane_ok = True
                    pothole_mask, hump_mask = gp_model.segment_anomalies(disparity_map)
                    anomalies = pipeline._process_anomalies(
                        disparity_map, pothole_mask, hump_mask, left_proc
                    )
                    
                    # Visualize masks
                    combined_mask = np.zeros_like(left_proc)
                    combined_mask[pothole_mask > 0] = 255
                    combined_mask[hump_mask > 0] = 128
                    cv2.imwrite(str(example_dir / "06_anomaly_mask.png"), combined_mask)
            except Exception as e:
                print(f"    Ground plane detection failed: {e}")
            
            results['ground_plane_detected'] = ground_plane_ok
            results['num_anomalies'] = len(anomalies)
            
            # Stage 6: Volume Calculation
            if anomalies:
                print(f"  Stage 6: Volume Calculation ({len(anomalies)} anomalies)...")
                results['anomalies'] = []
                
                for idx, anomaly in enumerate(anomalies):
                    anomaly_info = {
                        'type': anomaly.anomaly_type,
                        'volume_m3': anomaly.volume_cubic_meters,
                        'volume_liters': anomaly.volume_liters,
                        'volume_cm3': anomaly.volume_cubic_cm,
                        'area_m2': anomaly.area_square_meters,
                        'is_valid': anomaly.is_valid
                    }
                    results['anomalies'].append(anomaly_info)
                    
                    # Visualize anomaly point cloud
                    if anomaly.point_cloud is not None and len(anomaly.point_cloud) > 0:
                        self.render_3d_scatter(
                            anomaly.point_cloud[:, :3],
                            f'Anomaly {idx+1} - {anomaly.anomaly_type}',
                            example_dir / f"07_anomaly_{idx+1}_points.png",
                            cmap='plasma'
                        )
                    
                    # Visualize mesh if available
                    if anomaly.mesh is not None and anomaly.mesh.vertices.shape[0] > 0:
                        self.render_3d_scatter(
                            np.array(anomaly.mesh.vertices),
                            f'Anomaly {idx+1} - Mesh (Volume: {anomaly.volume_liters:.2f}L)',
                            example_dir / f"08_anomaly_{idx+1}_mesh.png",
                            cmap='plasma'
                        )
            
            print(f"  ✓ Processing complete!")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results['error'] = str(e)
            import traceback
            traceback.print_exc()
        
        # Save results JSON
        with open(example_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary
        self.create_summary(example_dir, description, results)
        
        return results
    
    def create_summary(self, example_dir, description, results):
        """Create summary visualization"""
        images = []
        titles = []
        
        # Collect images
        image_files = [
            ("01_input_left.png", "Input Left"),
            ("01_input_right.png", "Input Right"),
            ("02_preprocessed_left.png", "Preprocessed"),
            ("03_disparity_map.png", "Disparity"),
            ("04_v_disparity.png", "V-Disparity"),
            ("05_point_cloud.png", "Point Cloud"),
            ("06_anomaly_mask.png", "Anomaly Mask"),
        ]
        
        # Add anomaly visualizations (limit to first 3 for summary)
        max_anomalies_in_summary = 3
        for i in range(1, max_anomalies_in_summary + 1):
            for suffix in ["points", "mesh"]:
                filename = f"07_anomaly_{i}_{suffix}.png"
                if (example_dir / filename).exists():
                    image_files.append((filename, f"Anomaly {i} {suffix.title()}"))
        
        for filename, title in image_files:
            filepath = example_dir / filename
            if filepath.exists():
                img = cv2.imread(str(filepath))
                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    titles.append(title)
        
        if len(images) == 0:
            return
        
        # Create figure
        n_images = len(images)
        n_cols = 3
        n_rows = (n_images + n_cols) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        fig.suptitle(description, fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes = axes.flatten()
        
        for idx, (img, title) in enumerate(zip(images, titles)):
            axes[idx].imshow(img)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        # Add results text
        if idx + 1 < len(axes):
            results_text = "Results:\n\n"
            if 'num_anomalies' in results:
                results_text += f"Anomalies: {results['num_anomalies']}\n\n"
            if 'anomalies' in results:
                for i, anom in enumerate(results['anomalies'], 1):
                    results_text += f"Anomaly {i} ({anom.get('type', 'unknown')}):\n"
                    vol_cm3 = float(anom.get('volume_cm3', 0)) if anom.get('volume_cm3') else 0
                    vol_l = float(anom.get('volume_liters', 0)) if anom.get('volume_liters') else 0
                    area = float(anom.get('area_m2', 0)) if anom.get('area_m2') else 0
                    results_text += f"  Volume: {vol_cm3:.2f} cm³\n"
                    results_text += f"          {vol_l:.4f} L\n"
                    results_text += f"  Area: {area:.4f} m²\n\n"
            if 'point_cloud' in results:
                results_text += f"Points: {results['point_cloud']['num_points']:,}\n"
            
            axes[idx + 1].text(0.1, 0.5, results_text, fontsize=11,
                             verticalalignment='center', family='monospace')
            axes[idx + 1].axis('off')
        
        # Hide remaining
        for i in range(len(images) + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(str(example_dir / "00_SUMMARY.png"), dpi=200, bbox_inches='tight')
        plt.close()
    
    def generate_all(self):
        """Generate all presentation results"""
        print("\n" + "="*60)
        print("FINAL PRESENTATION RESULTS GENERATOR")
        print("="*60)
        
        # Example 1
        left1 = cv2.imread("data/dataset1/rgb/000001_left.png")
        right1 = cv2.imread("data/dataset1/rgb/000001_right.png")
        if left1 is not None and right1 is not None:
            self.process_example(left1, right1, "example_1", self.examples["example_1"])
        
        # Example 2
        left2 = cv2.imread("data/dataset1/rgb/000005_left.png")
        right2 = cv2.imread("data/dataset1/rgb/000005_right.png")
        if left2 is not None and right2 is not None:
            self.process_example(left2, right2, "example_2", self.examples["example_2"])
        
        # Edge Case 1: Night time
        left3 = cv2.imread("data/dataset1/rgb/000003_left.png")
        right3 = cv2.imread("data/dataset1/rgb/000003_right.png")
        if left3 is not None and right3 is not None:
            left3_night = self.simulate_night_time(left3, 0.25)
            right3_night = self.simulate_night_time(right3, 0.25)
            self.process_example(left3_night, right3_night, "edge_case_1", 
                               self.examples["edge_case_1"])
        
        # Edge Case 2: Night with noise
        left4 = cv2.imread("data/dataset1/rgb/000007_left.png")
        right4 = cv2.imread("data/dataset1/rgb/000007_right.png")
        if left4 is not None and right4 is not None:
            left4_night = self.simulate_night_time(left4, 0.2, add_noise=True)
            right4_night = self.simulate_night_time(right4, 0.2, add_noise=True)
            self.process_example(left4_night, right4_night, "edge_case_2",
                               self.examples["edge_case_2"])
        
        # Edge Case 3: Extreme shadows
        left5 = cv2.imread("data/dataset1/rgb/000010_left.png")
        right5 = cv2.imread("data/dataset1/rgb/000010_right.png")
        if left5 is not None and right5 is not None:
            left5_shadow = self.add_extreme_shadows(left5)
            right5_shadow = self.add_extreme_shadows(right5)
            self.process_example(left5_shadow, right5_shadow, "edge_case_3",
                               self.examples["edge_case_3"])
        
        # Create README
        self.create_readme()
        
        print("\n" + "="*60)
        print("✓ ALL RESULTS GENERATED SUCCESSFULLY!")
        print(f"✓ Output directory: {self.output_dir}")
        print("="*60)
    
    def create_readme(self):
        """Create README file"""
        content = f"""# Final Presentation Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
Comprehensive processing results for 5 test cases demonstrating the complete stereo vision pipeline.

## Examples

"""
        for key, desc in self.examples.items():
            content += f"### {key.replace('_', ' ').title()}\n"
            content += f"{desc}\n\n"
            content += f"**Directory:** `{key}/`\n\n"
            content += "**Files:**\n"
            content += "- `00_SUMMARY.png` - Complete overview of all stages\n"
            content += "- `01_input_*.png` - Original stereo pair\n"
            content += "- `02_preprocessed_*.png` - After preprocessing\n"
            content += "- `03_disparity_map.png` - Computed disparity\n"
            content += "- `04_v_disparity.png` - V-disparity for ground plane\n"
            content += "- `05_point_cloud.png` - 3D reconstruction\n"
            content += "- `06_anomaly_mask.png` - Detected anomalies\n"
            content += "- `07-08_anomaly_*` - Individual anomaly visualizations\n"
            content += "- `results.json` - Detailed numerical results\n\n"
        
        content += """## Processing Pipeline

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
"""
        
        with open(self.output_dir / "README.md", 'w') as f:
            f.write(content)


if __name__ == "__main__":
    generator = PresentationResultsGenerator()
    generator.generate_all()
