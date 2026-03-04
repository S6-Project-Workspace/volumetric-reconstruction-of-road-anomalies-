#!/usr/bin/env python3
"""
Generate comprehensive final presentation results using the existing pipeline.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

class SimplePresentationGenerator:
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
        
        # Add slight blue tint
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
    
    def process_with_pipeline(self, left_img, right_img, example_name, description):
        """Process using the actual pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing: {description}")
        print(f"{'='*60}")
        
        example_dir = self.output_dir / example_name
        
        # Save input images
        cv2.imwrite(str(example_dir / "01_input_left.png"), left_img)
        cv2.imwrite(str(example_dir / "01_input_right.png"), right_img)
        
        results = {'description': description}
        
        try:
            # Import and run the pipeline
            from pothole_volume_pipeline import process_stereo_pair
            
            print("Running stereo vision pipeline...")
            pipeline_result = process_stereo_pair(left_img, right_img)
            
            if pipeline_result is None:
                results['error'] = 'Pipeline returned None'
                print("✗ Pipeline failed")
                return results
            
            # Save disparity map
            if 'disparity_map' in pipeline_result:
                disp = pipeline_result['disparity_map']
                disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
                cv2.imwrite(str(example_dir / "02_disparity_map.png"), disp_color)
                results['disparity'] = {
                    'min': float(np.min(disp)),
                    'max': float(np.max(disp)),
                    'mean': float(np.mean(disp))
                }
            
            # Save diagnostic panel if available
            if 'diagnostic_panel' in pipeline_result and pipeline_result['diagnostic_panel'] is not None:
                cv2.imwrite(str(example_dir / "03_diagnostic_panel.png"), 
                           pipeline_result['diagnostic_panel'])
            
            # Extract anomaly results
            if 'anomalies' in pipeline_result:
                anomalies = pipeline_result['anomalies']
                results['num_anomalies'] = len(anomalies)
                results['anomalies'] = []
                
                for idx, anomaly in enumerate(anomalies):
                    anomaly_info = {
                        'type': anomaly.get('anomaly_type', 'unknown'),
                        'volume_cm3': anomaly.get('volume_cubic_cm', 0),
                        'volume_liters': anomaly.get('volume_liters', 0),
                        'area_m2': anomaly.get('area_square_meters', 0),
                        'is_valid': anomaly.get('is_valid', False)
                    }
                    results['anomalies'].append(anomaly_info)
                    
                    # Visualize point cloud if available
                    if 'point_cloud' in anomaly and anomaly['point_cloud'] is not None:
                        points = anomaly['point_cloud']
                        if len(points) > 0:
                            fig = plt.figure(figsize=(10, 10))
                            ax = fig.add_subplot(111, projection='3d')
                            
                            sample_size = min(5000, len(points))
                            sample_idx = np.random.choice(len(points), sample_size, replace=False)
                            sampled = points[sample_idx]
                            
                            ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2],
                                     c=sampled[:, 2], cmap='viridis', s=1)
                            ax.set_xlabel('X (m)')
                            ax.set_ylabel('Y (m)')
                            ax.set_zlabel('Z (m)')
                            ax.set_title(f'Anomaly {idx+1} Point Cloud')
                            plt.savefig(str(example_dir / f"04_anomaly_{idx+1}_points.png"), 
                                       dpi=150, bbox_inches='tight')
                            plt.close()
            
            # Processing metrics
            if 'processing_time' in pipeline_result:
                results['processing_time'] = pipeline_result['processing_time']
            
            if 'quality_metrics' in pipeline_result:
                results['quality_metrics'] = pipeline_result['quality_metrics']
            
            print(f"✓ Successfully processed: {description}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results['error'] = str(e)
            import traceback
            traceback.print_exc()
        
        # Save results JSON
        with open(example_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary visualization
        self.create_summary_image(example_dir, description, results)
        
        return results
    
    def create_summary_image(self, example_dir, description, results):
        """Create a summary visualization"""
        images = []
        titles = []
        
        # Collect available images
        image_files = [
            ("01_input_left.png", "Input (Left)"),
            ("01_input_right.png", "Input (Right)"),
            ("02_disparity_map.png", "Disparity Map"),
            ("03_diagnostic_panel.png", "Diagnostic Panel"),
        ]
        
        # Add anomaly images
        for i in range(1, 6):
            filename = f"04_anomaly_{i}_points.png"
            if (example_dir / filename).exists():
                image_files.append((filename, f"Anomaly {i}"))
        
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
        n_rows = (n_images + n_cols - 1) // n_cols
        
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
                    results_text += f"Anomaly {i}:\n"
                    results_text += f"  Volume: {anom.get('volume_cm3', 0):.2f} cm³\n"
                    results_text += f"  Type: {anom.get('type', 'unknown')}\n\n"
            if 'processing_time' in results:
                results_text += f"Time: {results['processing_time']:.2f}s\n"
            
            axes[idx + 1].text(0.1, 0.5, results_text, fontsize=12,
                             verticalalignment='center', family='monospace')
            axes[idx + 1].axis('off')
        
        # Hide remaining subplots
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
            self.process_with_pipeline(left1, right1, "example_1", self.examples["example_1"])
        
        # Example 2
        left2 = cv2.imread("data/dataset1/rgb/000005_left.png")
        right2 = cv2.imread("data/dataset1/rgb/000005_right.png")
        if left2 is not None and right2 is not None:
            self.process_with_pipeline(left2, right2, "example_2", self.examples["example_2"])
        
        # Edge Case 1: Night time
        left3 = cv2.imread("data/dataset1/rgb/000003_left.png")
        right3 = cv2.imread("data/dataset1/rgb/000003_right.png")
        if left3 is not None and right3 is not None:
            left3_night = self.simulate_night_time(left3, 0.25)
            right3_night = self.simulate_night_time(right3, 0.25)
            self.process_with_pipeline(left3_night, right3_night, "edge_case_1", 
                                      self.examples["edge_case_1"])
        
        # Edge Case 2: Night with noise
        left4 = cv2.imread("data/dataset1/rgb/000007_left.png")
        right4 = cv2.imread("data/dataset1/rgb/000007_right.png")
        if left4 is not None and right4 is not None:
            left4_night = self.simulate_night_time(left4, 0.2, add_noise=True)
            right4_night = self.simulate_night_time(right4, 0.2, add_noise=True)
            self.process_with_pipeline(left4_night, right4_night, "edge_case_2",
                                      self.examples["edge_case_2"])
        
        # Edge Case 3: Extreme shadows
        left5 = cv2.imread("data/dataset1/rgb/000010_left.png")
        right5 = cv2.imread("data/dataset1/rgb/000010_right.png")
        if left5 is not None and right5 is not None:
            left5_shadow = self.add_extreme_shadows(left5)
            right5_shadow = self.add_extreme_shadows(right5)
            self.process_with_pipeline(left5_shadow, right5_shadow, "edge_case_3",
                                      self.examples["edge_case_3"])
        
        # Create README
        self.create_readme()
        
        print("\n" + "="*60)
        print("✓ GENERATION COMPLETE!")
        print(f"✓ Results saved to: {self.output_dir}")
        print("="*60)
    
    def create_readme(self):
        """Create README file"""
        content = f"""# Final Presentation Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
Comprehensive processing results for 5 test cases:
- 2 Regular Examples (standard potholes/anomalies)
- 3 Edge Cases (night-time and extreme conditions)

## Examples

"""
        for key, desc in self.examples.items():
            content += f"### {key.replace('_', ' ').title()}\n"
            content += f"{desc}\n\n"
            content += f"Directory: `{key}/`\n"
            content += "- 00_SUMMARY.png - Complete overview\n"
            content += "- 01_input_*.png - Input stereo pair\n"
            content += "- 02_disparity_map.png - Computed disparity\n"
            content += "- 03_diagnostic_panel.png - Full diagnostic visualization\n"
            content += "- 04_anomaly_*_points.png - 3D point clouds\n"
            content += "- results.json - Detailed metrics\n\n"
        
        content += """## Processing Pipeline

1. Input: Stereo image pair
2. Preprocessing: Brightness normalization, contrast enhancement
3. Disparity Estimation: SGBM stereo matching
4. Ground Plane Detection: V-disparity analysis
5. Anomaly Segmentation: Outlier detection
6. Volume Calculation: Alpha shape meshing

## Edge Cases

- **Edge Case 1**: Low-light simulation (25% brightness)
- **Edge Case 2**: Low-light with sensor noise
- **Edge Case 3**: Extreme diagonal shadows

## Usage

View `00_SUMMARY.png` in each directory for a complete overview.
"""
        
        with open(self.output_dir / "README.md", 'w') as f:
            f.write(content)


if __name__ == "__main__":
    generator = SimplePresentationGenerator()
    generator.generate_all()
