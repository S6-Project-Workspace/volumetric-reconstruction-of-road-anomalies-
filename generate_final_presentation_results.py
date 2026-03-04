#!/usr/bin/env python3
"""
Generate comprehensive final presentation results with:
- 2 regular examples
- 3 edge cases (including night-time scenarios)
- All processing stages visualized
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from stereo_vision.pipeline import StereoVisionPipeline
from stereo_vision.config import PipelineConfig
import matplotlib.pyplot as plt
from datetime import datetime

class FinalPresentationGenerator:
    def __init__(self, output_dir="final_presentation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each example
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
        # Reduce brightness
        night_img = (image * brightness_factor).astype(np.uint8)
        
        # Add slight blue tint (night effect)
        night_img = cv2.cvtColor(night_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        night_img[:, :, 0] = np.clip(night_img[:, :, 0] + 10, 0, 179)  # Shift hue towards blue
        night_img[:, :, 1] = np.clip(night_img[:, :, 1] * 0.7, 0, 255)  # Reduce saturation
        night_img = cv2.cvtColor(night_img.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        if add_noise:
            # Add Gaussian noise
            noise = np.random.normal(0, 15, night_img.shape).astype(np.int16)
            night_img = np.clip(night_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return night_img
    
    def add_extreme_shadows(self, image):
        """Add extreme shadow effects"""
        # Create shadow mask
        h, w = image.shape[:2]
        shadow_mask = np.ones((h, w), dtype=np.float32)
        
        # Add diagonal shadow
        for i in range(h):
            shadow_mask[i, :int(w * i / h)] *= 0.3
        
        # Apply shadow
        shadowed = image.copy().astype(np.float32)
        for c in range(3):
            shadowed[:, :, c] *= shadow_mask
        
        return np.clip(shadowed, 0, 255).astype(np.uint8)
    
    def process_and_save_all_stages(self, left_img, right_img, example_name, description):
        """Process stereo pair through all pipeline stages and save results"""
        print(f"\n{'='*60}")
        print(f"Processing: {description}")
        print(f"{'='*60}")
        
        example_dir = self.output_dir / example_name
        
        # Save original images
        cv2.imwrite(str(example_dir / "01_input_left.png"), left_img)
        cv2.imwrite(str(example_dir / "01_input_right.png"), right_img)
        
        # Initialize pipeline
        config = PipelineConfig()
        pipeline = StereoVisionPipeline(config)
        
        results = {}
        
        try:
            # Stage 1: Preprocessing
            print("Stage 1: Preprocessing...")
            left_prep, right_prep = pipeline.preprocessor.preprocess_stereo_pair(left_img, right_img)
            cv2.imwrite(str(example_dir / "02_preprocessed_left.png"), left_prep)
            cv2.imwrite(str(example_dir / "02_preprocessed_right.png"), right_prep)
            results['preprocessing'] = 'Success'
            
            # Stage 2: Disparity Estimation
            print("Stage 2: Disparity Estimation...")
            disparity = pipeline.disparity_estimator.compute_disparity(left_prep, right_prep)
            
            # Normalize disparity for visualization
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            cv2.imwrite(str(example_dir / "03_disparity_map.png"), disp_color)
            results['disparity'] = {
                'min': float(np.min(disparity)),
                'max': float(np.max(disparity)),
                'mean': float(np.mean(disparity))
            }
            
            # Stage 3: 3D Reconstruction
            print("Stage 3: 3D Reconstruction...")
            points_3d = pipeline.reconstructor.reconstruct(disparity, left_prep)
            
            # Visualize point cloud (top view)
            if len(points_3d) > 0:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Sample points for visualization
                sample_idx = np.random.choice(len(points_3d), min(10000, len(points_3d)), replace=False)
                sampled_points = points_3d[sample_idx]
                
                ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
                          c=sampled_points[:, 2], cmap='viridis', s=1)
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                ax.set_title('3D Point Cloud')
                plt.savefig(str(example_dir / "04_point_cloud_3d.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                results['reconstruction'] = {
                    'num_points': len(points_3d),
                    'x_range': [float(np.min(points_3d[:, 0])), float(np.max(points_3d[:, 0]))],
                    'y_range': [float(np.min(points_3d[:, 1])), float(np.max(points_3d[:, 1]))],
                    'z_range': [float(np.min(points_3d[:, 2])), float(np.max(points_3d[:, 2]))]
                }
            
            # Stage 4: Ground Plane Detection
            print("Stage 4: Ground Plane Detection...")
            ground_plane = pipeline.ground_plane_detector.detect_ground_plane(points_3d)
            
            if ground_plane is not None:
                # Visualize ground plane
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                sample_idx = np.random.choice(len(points_3d), min(10000, len(points_3d)), replace=False)
                sampled_points = points_3d[sample_idx]
                
                # Color points by distance from ground plane
                distances = np.abs(np.dot(sampled_points, ground_plane[:3]) + ground_plane[3])
                
                scatter = ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
                                   c=distances, cmap='RdYlGn_r', s=1, vmin=0, vmax=100)
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                ax.set_title('Ground Plane Detection (Red = Anomaly)')
                plt.colorbar(scatter, label='Distance from Ground (mm)')
                plt.savefig(str(example_dir / "05_ground_plane_detection.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                results['ground_plane'] = {
                    'coefficients': ground_plane.tolist(),
                    'detected': True
                }
            
            # Stage 5: Anomaly Segmentation
            print("Stage 5: Anomaly Segmentation...")
            anomaly_points = pipeline.reconstructor.segment_anomalies(points_3d, ground_plane)
            
            if len(anomaly_points) > 0:
                # Visualize anomaly points
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                ax.scatter(anomaly_points[:, 0], anomaly_points[:, 1], anomaly_points[:, 2],
                          c='red', s=2, label='Anomaly Points')
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                ax.set_title('Segmented Anomaly Points')
                ax.legend()
                plt.savefig(str(example_dir / "06_anomaly_segmentation.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                results['anomaly_segmentation'] = {
                    'num_anomaly_points': len(anomaly_points),
                    'percentage': float(len(anomaly_points) / len(points_3d) * 100)
                }
            
            # Stage 6: Volume Calculation
            print("Stage 6: Volume Calculation...")
            if len(anomaly_points) > 100:  # Need sufficient points
                volume_result = pipeline.volume_calculator.calculate_volume(anomaly_points)
                
                if volume_result['success']:
                    # Visualize mesh
                    mesh = volume_result['mesh']
                    
                    fig = plt.figure(figsize=(12, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Plot mesh
                    vertices = mesh.vertices
                    faces = mesh.faces
                    
                    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                   triangles=faces, alpha=0.7, cmap='viridis')
                    ax.set_xlabel('X (mm)')
                    ax.set_ylabel('Y (mm)')
                    ax.set_zlabel('Z (mm)')
                    ax.set_title(f'3D Mesh - Volume: {volume_result["volume_cm3"]:.2f} cm³')
                    plt.savefig(str(example_dir / "07_volume_mesh.png"), dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    results['volume'] = {
                        'volume_cm3': volume_result['volume_cm3'],
                        'volume_liters': volume_result['volume_liters'],
                        'mesh_vertices': len(vertices),
                        'mesh_faces': len(faces),
                        'success': True
                    }
                else:
                    results['volume'] = {'success': False, 'error': volume_result.get('error', 'Unknown')}
            else:
                results['volume'] = {'success': False, 'error': 'Insufficient anomaly points'}
            
            # Create summary visualization
            self.create_summary_visualization(example_dir, example_name, description, results)
            
        except Exception as e:
            print(f"Error processing {example_name}: {str(e)}")
            results['error'] = str(e)
        
        # Save results JSON
        with open(example_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Completed: {description}")
        return results
    
    def create_summary_visualization(self, example_dir, example_name, description, results):
        """Create a single summary image with all stages"""
        # Load key images
        images = []
        titles = []
        
        image_files = [
            ("01_input_left.png", "Input (Left)"),
            ("02_preprocessed_left.png", "Preprocessed"),
            ("03_disparity_map.png", "Disparity Map"),
            ("04_point_cloud_3d.png", "3D Reconstruction"),
            ("05_ground_plane_detection.png", "Ground Plane"),
            ("06_anomaly_segmentation.png", "Anomaly Segmentation"),
            ("07_volume_mesh.png", "Volume Mesh")
        ]
        
        for filename, title in image_files:
            filepath = example_dir / filename
            if filepath.exists():
                img = cv2.imread(str(filepath))
                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    titles.append(title)
        
        # Create summary figure
        n_images = len(images)
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))
        fig.suptitle(f"{description}\n{example_name}", fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, (img, title) in enumerate(zip(images, titles)):
            if idx < len(axes):
                axes[idx].imshow(img)
                axes[idx].set_title(title, fontsize=12, fontweight='bold')
                axes[idx].axis('off')
        
        # Add results text in remaining subplot
        if len(images) < len(axes):
            results_text = "Processing Results:\n\n"
            if 'volume' in results and results['volume'].get('success'):
                results_text += f"Volume: {results['volume']['volume_cm3']:.2f} cm³\n"
                results_text += f"        ({results['volume']['volume_liters']:.4f} L)\n\n"
            if 'reconstruction' in results:
                results_text += f"Points: {results['reconstruction']['num_points']:,}\n\n"
            if 'anomaly_segmentation' in results:
                results_text += f"Anomaly Points: {results['anomaly_segmentation']['num_anomaly_points']:,}\n"
                results_text += f"Percentage: {results['anomaly_segmentation']['percentage']:.1f}%\n"
            
            axes[len(images)].text(0.1, 0.5, results_text, fontsize=14, 
                                  verticalalignment='center', family='monospace')
            axes[len(images)].axis('off')
        
        # Hide remaining empty subplots
        for idx in range(len(images) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(str(example_dir / "00_SUMMARY.png"), dpi=200, bbox_inches='tight')
        plt.close()
    
    def generate_all_results(self):
        """Generate all presentation results"""
        # Load calibration data
        calib_file = Path("data/dataset1/calib.txt")
        if not calib_file.exists():
            print("Warning: Calibration file not found. Using default values.")
        
        # Example 1: Regular pothole
        print("\n" + "="*60)
        print("GENERATING EXAMPLE 1: Regular Pothole")
        print("="*60)
        left1 = cv2.imread("data/dataset1/rgb/000001_left.png")
        right1 = cv2.imread("data/dataset1/rgb/000001_right.png")
        if left1 is not None and right1 is not None:
            self.process_and_save_all_stages(left1, right1, "example_1", 
                                            self.examples["example_1"])
        
        # Example 2: Large anomaly
        print("\n" + "="*60)
        print("GENERATING EXAMPLE 2: Large Anomaly")
        print("="*60)
        left2 = cv2.imread("data/dataset1/rgb/000005_left.png")
        right2 = cv2.imread("data/dataset1/rgb/000005_right.png")
        if left2 is not None and right2 is not None:
            self.process_and_save_all_stages(left2, right2, "example_2",
                                            self.examples["example_2"])
        
        # Edge Case 1: Night time (low light)
        print("\n" + "="*60)
        print("GENERATING EDGE CASE 1: Night Time (Low Light)")
        print("="*60)
        left3 = cv2.imread("data/dataset1/rgb/000003_left.png")
        right3 = cv2.imread("data/dataset1/rgb/000003_right.png")
        if left3 is not None and right3 is not None:
            left3_night = self.simulate_night_time(left3, brightness_factor=0.25)
            right3_night = self.simulate_night_time(right3, brightness_factor=0.25)
            self.process_and_save_all_stages(left3_night, right3_night, "edge_case_1",
                                            self.examples["edge_case_1"])
        
        # Edge Case 2: Night time with noise
        print("\n" + "="*60)
        print("GENERATING EDGE CASE 2: Night Time with Noise")
        print("="*60)
        left4 = cv2.imread("data/dataset1/rgb/000007_left.png")
        right4 = cv2.imread("data/dataset1/rgb/000007_right.png")
        if left4 is not None and right4 is not None:
            left4_night = self.simulate_night_time(left4, brightness_factor=0.2, add_noise=True)
            right4_night = self.simulate_night_time(right4, brightness_factor=0.2, add_noise=True)
            self.process_and_save_all_stages(left4_night, right4_night, "edge_case_2",
                                            self.examples["edge_case_2"])
        
        # Edge Case 3: Extreme shadows
        print("\n" + "="*60)
        print("GENERATING EDGE CASE 3: Extreme Shadows")
        print("="*60)
        left5 = cv2.imread("data/dataset1/rgb/000010_left.png")
        right5 = cv2.imread("data/dataset1/rgb/000010_right.png")
        if left5 is not None and right5 is not None:
            left5_shadow = self.add_extreme_shadows(left5)
            right5_shadow = self.add_extreme_shadows(right5)
            self.process_and_save_all_stages(left5_shadow, right5_shadow, "edge_case_3",
                                            self.examples["edge_case_3"])
        
        # Create master summary
        self.create_master_summary()
        
        print("\n" + "="*60)
        print("✓ ALL PRESENTATION RESULTS GENERATED SUCCESSFULLY!")
        print(f"✓ Output directory: {self.output_dir}")
        print("="*60)
    
    def create_master_summary(self):
        """Create a master summary document"""
        summary = {
            'generated_at': datetime.now().isoformat(),
            'examples': self.examples,
            'results': {}
        }
        
        for example_name in self.examples.keys():
            results_file = self.output_dir / example_name / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    summary['results'][example_name] = json.load(f)
        
        with open(self.output_dir / "MASTER_SUMMARY.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create README
        readme_content = f"""# Final Presentation Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This directory contains comprehensive processing results for 5 test cases:
- 2 Regular Examples
- 3 Edge Cases (including night-time scenarios)

## Examples

"""
        for key, desc in self.examples.items():
            readme_content += f"### {key.replace('_', ' ').title()}\n"
            readme_content += f"{desc}\n\n"
            readme_content += f"Directory: `{key}/`\n"
            readme_content += f"- Input images (left/right)\n"
            readme_content += f"- Preprocessed images\n"
            readme_content += f"- Disparity map\n"
            readme_content += f"- 3D point cloud\n"
            readme_content += f"- Ground plane detection\n"
            readme_content += f"- Anomaly segmentation\n"
            readme_content += f"- Volume mesh\n"
            readme_content += f"- Complete summary visualization\n"
            readme_content += f"- Results JSON\n\n"
        
        readme_content += """## Processing Pipeline

Each example goes through the following stages:

1. **Input**: Original stereo image pair
2. **Preprocessing**: Brightness normalization and enhancement
3. **Disparity Estimation**: SGBM stereo matching
4. **3D Reconstruction**: Point cloud generation
5. **Ground Plane Detection**: RANSAC-based plane fitting
6. **Anomaly Segmentation**: Outlier detection
7. **Volume Calculation**: Alpha shape mesh generation and volume computation

## Edge Cases

### Night Time Scenarios
- **Edge Case 1**: Simulated low-light conditions (30% brightness)
- **Edge Case 2**: Low-light with sensor noise

### Extreme Conditions
- **Edge Case 3**: Extreme shadow effects

## Files

- `00_SUMMARY.png`: Complete visualization of all processing stages
- `results.json`: Detailed numerical results
- Individual stage outputs (01-07)

## Usage

View the `00_SUMMARY.png` in each directory for a quick overview of all processing stages.
"""
        
        with open(self.output_dir / "README.md", 'w') as f:
            f.write(readme_content)


if __name__ == "__main__":
    generator = FinalPresentationGenerator()
    generator.generate_all_results()
