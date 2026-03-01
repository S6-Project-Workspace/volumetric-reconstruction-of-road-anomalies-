# -*- coding: utf-8 -*-
"""
Gradio UI for Advanced Stereo Vision Pipeline Testing
"""

import gradio as gr
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from stereo_vision.pipeline import StereoVisionPipeline
from stereo_vision.config import PipelineConfig, CameraConfig
from stereo_vision.preprocessing import ImagePreprocessor
from stereo_vision.disparity import SGBMEstimator
from stereo_vision.ground_plane import VDisparityGenerator
from stereo_vision.reconstruction import PointCloudGenerator, OutlierRemover
from stereo_vision.volumetric import AlphaShapeGenerator, MeshCapper, VolumeCalculator
from stereo_vision.quality_metrics import QualityMetrics


# Global state
pipeline = None
current_disparity = None
current_points = None


def _cap_and_close_mesh(alpha_gen, capper, mesh):
    """Extract boundary edges, cap them, and create a watertight mesh."""
    boundary_edges = alpha_gen.extract_boundary_edges(mesh)
    if len(boundary_edges) > 0:
        cap = capper.triangulate_boundary(boundary_edges, mesh.vertices)
        return capper.create_watertight_mesh(mesh, cap)
    return mesh


def _render_scatter_3d(points, title, cmap='viridis', max_points=10000):
    """Render a 3D scatter plot of points and return as numpy image array."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    vis = points
    if len(vis) > max_points:
        idx = np.random.choice(len(vis), max_points, replace=False)
        vis = vis[idx]
    
    if len(vis) > 0:
        ax.scatter(vis[:, 0], vis[:, 1], vis[:, 2], c=vis[:, 2], cmap=cmap, s=1)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return np.array(Image.open(buf))


def _generate_mesh_image(points):
    """Generate an alpha shape mesh visualization from a point cloud.
    
    Returns (mesh_img, volume, alpha) or (scatter_img, 0.0, 0.0) as fallback.
    """
    if points is None or len(points) < 4:
        return None, 0.0, 0.0
    
    mesh_points = points[:, :3] if points.shape[1] > 3 else points
    if len(mesh_points) > 5000:
        idx = np.random.choice(len(mesh_points), 5000, replace=False)
        mesh_points = mesh_points[idx]
    
    # Estimate alpha from point spacing
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh_points)
    dists, _ = tree.query(mesh_points, k=2)
    mean_spacing = np.mean(dists[:, 1])
    alpha = max(mean_spacing * 5.0, 2.0)
    
    alpha_gen = AlphaShapeGenerator(alpha=alpha)
    mesh = alpha_gen.generate_alpha_shape(mesh_points)
    
    has_faces = mesh.faces is not None and len(mesh.faces) > 0
    volume = 0.0
    
    if has_faces:
        capper = MeshCapper()
        capped = _cap_and_close_mesh(alpha_gen, capper, mesh)
        try:
            calc = VolumeCalculator()
            volume = abs(calc.calculate_signed_volume(capped))
        except Exception:
            volume = 0.0
        
        verts = np.array(capped.vertices)
        title = f'Alpha Shape Mesh\nVolume: {volume:.4f} m³ | α={alpha:.2f}'
        img = _render_scatter_3d(verts, title, cmap='plasma', max_points=5000)
    else:
        title = f'Point Cloud ({len(mesh_points)} pts, α={alpha:.2f})'
        img = _render_scatter_3d(mesh_points, title, cmap='plasma', max_points=5000)
    
    return img, volume, alpha


def run_full_pipeline(left_image, right_image, baseline, focal_length):
    """Run the complete pipeline end-to-end.
    
    Runs individual pipeline stages so we can still produce output
    even when ground plane detection fails.
    """
    global pipeline, current_disparity, current_points
    
    if left_image is None or right_image is None:
        return None, None, None, None, "Please upload both images"
    
    try:
        from stereo_vision.calibration import CameraParameters, StereoParameters
        from stereo_vision.ground_plane import VDisparityGenerator, HoughLineDetector, GroundPlaneModel
        
        camera_config = CameraConfig(baseline=baseline, focal_length=focal_length)
        config = PipelineConfig(camera=camera_config)
        config.wls.enabled = False
        pipeline = StereoVisionPipeline(config)
        
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        h, w = left_gray.shape
        
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
        
        # ---- Stage 1: Preprocessing ----
        left_proc, right_proc = pipeline.preprocessor.preprocess_stereo_pair(
            left_gray, right_gray
        )
        
        # ---- Stage 2: Disparity ----
        disp_left = pipeline.sgbm_estimator.compute_disparity(left_proc, right_proc)
        disp_right = pipeline.sgbm_estimator.compute_disparity(right_proc, left_proc)
        
        # LRC + optional WLS
        if config.lrc.enabled:
            from stereo_vision.disparity import LRCValidator
            lrc = LRCValidator(max_diff=config.lrc.max_diff)
            validity_mask = lrc.validate_consistency(disp_left, disp_right)
        else:
            validity_mask = np.ones(disp_left.shape, dtype=np.uint8)
        
        disparity_map = disp_left.astype(np.float32) / 16.0
        disparity_map[validity_mask == 0] = 0
        current_disparity = disparity_map
        
        disp_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disp_colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        
        # ---- Stage 3: V-Disparity ----
        v_disp_gen = VDisparityGenerator()
        v_disparity = v_disp_gen.generate_v_disparity(disparity_map)
        
        v_disp_colored = None
        if v_disparity is not None:
            v_vis = cv2.normalize(v_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            v_disp_colored = cv2.applyColorMap(v_vis, cv2.COLORMAP_HOT)
        
        # ---- Stage 4: Point cloud ----
        full_points = pc_gen.reproject_to_3d(disparity_map, apply_depth_filter=True)
        
        # Outlier removal
        if len(full_points) > 0:
            outlier_remover = OutlierRemover(
                k_neighbors=config.outlier_removal.k_neighbors,
                std_ratio=config.outlier_removal.std_ratio
            )
            full_points = outlier_remover.remove_statistical_outliers(full_points)
        
        current_points = full_points
        
        # ---- Stage 5: Ground plane (may fail) ----
        anomalies = []
        ground_plane_ok = False
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
        except Exception:
            pass
        
        # ---- Visualization: Point Cloud ----
        pointcloud_img = _render_scatter_3d(
            full_points[:, :3] if len(full_points) > 0 else np.zeros((0, 3)),
            '3D Point Cloud' + ('' if ground_plane_ok else ' (No Ground Plane)')
        )
        
        # ---- Visualization: Volume Mesh ----
        mesh_img = None
        
        # Try anomaly meshes first
        if anomalies:
            for anomaly in anomalies:
                if anomaly.mesh is not None and anomaly.mesh.vertices.shape[0] > 0:
                    mesh_img = _render_scatter_3d(
                        np.array(anomaly.mesh.vertices),
                        f'Volume Mesh ({anomaly.anomaly_type})',
                        cmap='plasma', max_points=5000
                    )
                    break
        
        # Fallback: alpha shape from full point cloud
        if mesh_img is None and len(full_points) >= 4:
            try:
                mesh_img, volume, alpha_used = _generate_mesh_image(full_points)
            except Exception:
                mesh_img = _render_scatter_3d(
                    full_points[:, :3], 'Point Cloud (mesh generation failed)',
                    cmap='plasma', max_points=5000
                )
        
        # ---- Status text ----
        stats = "Pipeline completed successfully\n\n"
        
        valid_disp = disparity_map[disparity_map > 0]
        if len(valid_disp) > 0:
            stats += f"Disparity range: {valid_disp.min():.2f} - {valid_disp.max():.2f}\n"
        
        stats += f"Point cloud size: {len(full_points)} points\n"
        
        if len(full_points) > 0:
            stats += f"Depth range: {full_points[:, 2].min():.2f}m - {full_points[:, 2].max():.2f}m\n"
        
        if ground_plane_ok:
            stats += f"Ground plane: detected\n"
            stats += f"Anomalies detected: {len(anomalies)}\n"
            if anomalies:
                total_vol = sum(a.volume_cubic_meters for a in anomalies if a.is_valid)
                stats += f"Total volume: {total_vol:.6f} m³ ({total_vol*1000:.2f} liters)\n"
                for i, a in enumerate(anomalies):
                    stats += f"\nAnomaly {i+1} ({a.anomaly_type}):\n"
                    stats += f"  Volume: {a.volume_cubic_meters:.6f} m³ ({a.volume_liters:.2f} L)\n"
                    stats += f"  Valid: {a.is_valid}\n"
        else:
            stats += "\nGround plane: not detected\n"
            stats += "Alpha shape mesh generated from full point cloud.\n"
            stats += "Use the Volume Calculation tab for detailed analysis.\n"
        
        return disp_colored, v_disp_colored, pointcloud_img, mesh_img, stats
    
    except Exception as e:
        import traceback
        return None, None, None, None, f"Error: {str(e)}\n\n{traceback.format_exc()}"


def test_preprocessing(left_image, right_image, enhance_contrast, normalize_brightness, filter_noise):
    """Test preprocessing functionality."""
    if left_image is None or right_image is None:
        return None, None, "Please upload both images"
    
    try:
        preprocessor = ImagePreprocessor()
        
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        left_processed = left_gray.copy()
        right_processed = right_gray.copy()
        
        if enhance_contrast:
            left_processed = preprocessor.enhance_contrast(left_processed)
            right_processed = preprocessor.enhance_contrast(right_processed)
        
        if normalize_brightness:
            left_processed, right_processed = preprocessor.normalize_brightness(
                left_processed, right_processed
            )
        
        if filter_noise:
            left_processed = preprocessor.filter_noise(left_processed)
            right_processed = preprocessor.filter_noise(right_processed)
        
        status = f"Preprocessing complete\n"
        status += f"Contrast enhancement: {'ON' if enhance_contrast else 'OFF'}\n"
        status += f"Brightness normalization: {'ON' if normalize_brightness else 'OFF'}\n"
        status += f"Noise filtering: {'ON' if filter_noise else 'OFF'}"
        
        return left_processed, right_processed, status
    
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def test_disparity(left_image, right_image):
    """Test disparity estimation."""
    global current_disparity
    
    if left_image is None or right_image is None:
        return None, "Please upload both images"
    
    try:
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        sgbm = SGBMEstimator(baseline=0.12, focal_length=700.0)
        disparity = sgbm.compute_disparity(left_gray, right_gray)
        current_disparity = disparity
        
        disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disparity_colored = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
        
        valid_disparity = disparity[disparity > 0]
        stats = f"Disparity computed\n"
        stats += f"Min disparity: {valid_disparity.min():.2f}\n"
        stats += f"Max disparity: {valid_disparity.max():.2f}\n"
        stats += f"Mean disparity: {valid_disparity.mean():.2f}\n"
        stats += f"Valid pixels: {len(valid_disparity)} ({100*len(valid_disparity)/disparity.size:.1f}%)"
        
        return disparity_colored, stats
    
    except Exception as e:
        return None, f"Error: {str(e)}"


def test_ground_plane():
    """Test ground plane detection."""
    global current_disparity
    
    if current_disparity is None:
        return None, "Please compute disparity first"
    
    try:
        v_disp_gen = VDisparityGenerator(max_disparity=128)
        v_disparity = v_disp_gen.generate_v_disparity(current_disparity)
        
        v_disp_vis = cv2.normalize(v_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        v_disp_colored = cv2.applyColorMap(v_disp_vis, cv2.COLORMAP_HOT)
        
        stats = f"V-disparity generated\n"
        stats += f"Shape: {v_disparity.shape}\n"
        stats += f"Non-zero pixels: {np.count_nonzero(v_disparity)}"
        
        return v_disp_colored, stats
    
    except Exception as e:
        return None, f"Error: {str(e)}"


def test_reconstruction(min_depth, max_depth, remove_outliers):
    """Test 3D reconstruction."""
    global current_disparity, current_points
    
    if current_disparity is None:
        return None, "Please compute disparity first"
    
    try:
        Q = np.array([
            [1, 0, 0, -320],
            [0, 1, 0, -240],
            [0, 0, 0, 700],
            [0, 0, 1/0.12, 0]
        ], dtype=np.float32)
        
        pcg = PointCloudGenerator(Q_matrix=Q, min_depth=min_depth, max_depth=max_depth)
        points = pcg.reproject_to_3d(current_disparity, apply_depth_filter=True)
        
        if remove_outliers and len(points) > 100:
            outlier_remover = OutlierRemover()
            points = outlier_remover.remove_statistical_outliers(points)
        
        current_points = points
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            points_vis = points[indices]
        else:
            points_vis = points
        
        ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
                  c=points_vis[:, 2], cmap='viridis', s=1)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Point Cloud')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        img = Image.open(buf)
        img_array = np.array(img)
        
        stats = f"Point cloud generated\n"
        stats += f"Total points: {len(points)}\n"
        stats += f"Depth range: {points[:, 2].min():.2f}m to {points[:, 2].max():.2f}m\n"
        stats += f"Outlier removal: {'ON' if remove_outliers else 'OFF'}"
        
        return img_array, stats
    
    except Exception as e:
        return None, f"Error: {str(e)}"


def test_volume(alpha, ground_plane_z):
    """Test volume calculation."""
    global current_points
    
    if current_points is None:
        return None, "Please generate point cloud first"
    
    try:
        alpha_gen = AlphaShapeGenerator(alpha=alpha)
        mesh = alpha_gen.generate_alpha_shape(current_points)
        
        capper = MeshCapper()
        capped_mesh = _cap_and_close_mesh(alpha_gen, capper, mesh)
        
        calc = VolumeCalculator()
        volume = calc.calculate_signed_volume(capped_mesh)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = np.array(capped_mesh.vertices)
        if len(vertices) > 5000:
            indices = np.random.choice(len(vertices), 5000, replace=False)
            vertices_vis = vertices[indices]
        else:
            vertices_vis = vertices
        
        ax.scatter(vertices_vis[:, 0], vertices_vis[:, 1], vertices_vis[:, 2], 
                  c=vertices_vis[:, 2], cmap='plasma', s=2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Alpha Shape Mesh')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        img = Image.open(buf)
        img_array = np.array(img)
        
        stats = f"Volume calculated\n"
        stats += f"Volume: {volume:.6f} m3 ({volume*1000:.2f} liters)\n"
        stats += f"Alpha parameter: {alpha}\n"
        stats += f"Mesh vertices: {len(capped_mesh.vertices)}\n"
        stats += f"Mesh faces: {len(capped_mesh.faces)}"
        
        return img_array, stats
    
    except Exception as e:
        return None, f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Advanced Stereo Vision Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Advanced Stereo Vision Pipeline - Interactive Testing Interface
    
    Test all functionalities of the stereo vision pipeline.
    """)
    
    with gr.Tabs():
        with gr.Tab("Full Pipeline"):
            gr.Markdown("### Run the complete pipeline end-to-end")
            
            with gr.Row():
                with gr.Column():
                    full_left = gr.Image(label="Left Image", type="numpy")
                    full_right = gr.Image(label="Right Image", type="numpy")
                    
                    with gr.Row():
                        full_baseline = gr.Slider(0.01, 1.0, value=0.12, label="Baseline (m)")
                        full_focal = gr.Slider(100, 2000, value=700, label="Focal Length (px)")
                    
                    full_run_btn = gr.Button("Run Full Pipeline", variant="primary")
                
                with gr.Column():
                    full_disparity = gr.Image(label="Disparity Map")
                    full_vdisp = gr.Image(label="V-Disparity")
            
            with gr.Row():
                full_pointcloud = gr.Image(label="Point Cloud")
                full_mesh = gr.Image(label="Volume Mesh")
            
            full_status = gr.Textbox(label="Status", lines=10)
            
            full_run_btn.click(
                run_full_pipeline,
                inputs=[full_left, full_right, full_baseline, full_focal],
                outputs=[full_disparity, full_vdisp, full_pointcloud, full_mesh, full_status]
            )
        
        with gr.Tab("Preprocessing"):
            gr.Markdown("### Test image preprocessing capabilities")
            
            with gr.Row():
                with gr.Column():
                    prep_left = gr.Image(label="Left Image", type="numpy")
                    prep_right = gr.Image(label="Right Image", type="numpy")
                    
                    prep_contrast = gr.Checkbox(label="Enhance Contrast (CLAHE)", value=True)
                    prep_brightness = gr.Checkbox(label="Normalize Brightness", value=True)
                    prep_noise = gr.Checkbox(label="Filter Noise (Bilateral)", value=False)
                    
                    prep_btn = gr.Button("Process Images", variant="primary")
                
                with gr.Column():
                    prep_left_out = gr.Image(label="Processed Left")
                    prep_right_out = gr.Image(label="Processed Right")
                    prep_status = gr.Textbox(label="Status", lines=5)
            
            prep_btn.click(
                test_preprocessing,
                inputs=[prep_left, prep_right, prep_contrast, prep_brightness, prep_noise],
                outputs=[prep_left_out, prep_right_out, prep_status]
            )
        
        with gr.Tab("Disparity Estimation"):
            gr.Markdown("### Compute disparity maps using SGBM")
            
            with gr.Row():
                with gr.Column():
                    disp_left = gr.Image(label="Left Image", type="numpy")
                    disp_right = gr.Image(label="Right Image", type="numpy")
                    disp_btn = gr.Button("Compute Disparity", variant="primary")
                
                with gr.Column():
                    disp_out = gr.Image(label="Disparity Map (Colored)")
                    disp_status = gr.Textbox(label="Statistics", lines=8)
            
            disp_btn.click(
                test_disparity,
                inputs=[disp_left, disp_right],
                outputs=[disp_out, disp_status]
            )
        
        with gr.Tab("Ground Plane Detection"):
            gr.Markdown("### Detect ground plane using V-disparity")
            
            with gr.Row():
                with gr.Column():
                    gp_btn = gr.Button("Detect Ground Plane", variant="primary")
                
                with gr.Column():
                    gp_out = gr.Image(label="V-Disparity Map")
                    gp_status = gr.Textbox(label="Results", lines=8)
            
            gp_btn.click(
                test_ground_plane,
                inputs=[],
                outputs=[gp_out, gp_status]
            )
        
        with gr.Tab("3D Reconstruction"):
            gr.Markdown("### Generate 3D point cloud from disparity")
            
            with gr.Row():
                with gr.Column():
                    recon_min_depth = gr.Slider(0.1, 5.0, value=0.5, label="Min Depth (m)")
                    recon_max_depth = gr.Slider(5.0, 50.0, value=20.0, label="Max Depth (m)")
                    recon_outliers = gr.Checkbox(label="Remove Outliers", value=True)
                    recon_btn = gr.Button("Generate Point Cloud", variant="primary")
                
                with gr.Column():
                    recon_out = gr.Image(label="Point Cloud Visualization")
                    recon_status = gr.Textbox(label="Statistics", lines=8)
            
            recon_btn.click(
                test_reconstruction,
                inputs=[recon_min_depth, recon_max_depth, recon_outliers],
                outputs=[recon_out, recon_status]
            )
        
        with gr.Tab("Volume Calculation"):
            gr.Markdown("### Calculate volume using alpha shapes")
            
            with gr.Row():
                with gr.Column():
                    vol_alpha = gr.Slider(0.01, 1.0, value=0.1, label="Alpha Parameter")
                    vol_ground_z = gr.Slider(-5.0, 5.0, value=0.0, label="Ground Plane Z (m)")
                    vol_btn = gr.Button("Calculate Volume", variant="primary")
                
                with gr.Column():
                    vol_out = gr.Image(label="Alpha Shape Mesh")
                    vol_status = gr.Textbox(label="Results", lines=8)
            
            vol_btn.click(
                test_volume,
                inputs=[vol_alpha, vol_ground_z],
                outputs=[vol_out, vol_status]
            )
    
    gr.Markdown("""
    ---
    ### Usage Instructions
    
    1. **Full Pipeline**: Upload stereo images and run the complete pipeline
    2. **Preprocessing**: Test contrast enhancement, brightness normalization, and noise filtering
    3. **Disparity Estimation**: Compute disparity maps
    4. **Ground Plane Detection**: Detect ground plane using V-disparity
    5. **3D Reconstruction**: Generate point clouds
    6. **Volume Calculation**: Calculate volumes using alpha shapes
    """)


if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860, show_api=False)
