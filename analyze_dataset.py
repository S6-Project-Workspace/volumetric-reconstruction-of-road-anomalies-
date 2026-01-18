#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes the stereo image dataset for quality and lighting conditions.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def analyze_image_pair(left_path, right_path, pair_name):
    """Analyze a stereo image pair for quality metrics."""
    
    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        print(f"Missing files for {pair_name}")
        return None
    
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        print(f"Failed to load images for {pair_name}")
        return None
    
    print(f"\n=== {pair_name} ===")
    print(f"Resolution: {left_img.shape[1]}x{left_img.shape[0]}")
    
    # Convert to grayscale for analysis
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Analyze brightness and contrast
    left_mean = np.mean(left_gray)
    right_mean = np.mean(right_gray)
    left_std = np.std(left_gray)
    right_std = np.std(right_gray)
    
    print(f"Left - Mean brightness: {left_mean:.1f}, Contrast (std): {left_std:.1f}")
    print(f"Right - Mean brightness: {right_mean:.1f}, Contrast (std): {right_std:.1f}")
    
    # Estimate lighting condition
    avg_brightness = (left_mean + right_mean) / 2
    if avg_brightness < 80:
        lighting = "Dark/Night"
    elif avg_brightness < 120:
        lighting = "Low light/Dusk"
    elif avg_brightness < 180:
        lighting = "Normal/Day"
    else:
        lighting = "Bright/Overexposed"
    
    print(f"Estimated lighting: {lighting}")
    
    # Check for texture (important for stereo matching)
    grad_x = cv2.Sobel(left_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(left_gray, cv2.CV_64F, 0, 1, ksize=3)
    texture_measure = np.mean(np.sqrt(grad_x**2 + grad_y**2))
    print(f"Texture richness: {texture_measure:.1f}")
    
    if texture_measure < 10:
        texture_quality = "Low (challenging for stereo)"
    elif texture_measure < 20:
        texture_quality = "Moderate"
    else:
        texture_quality = "Good"
    print(f"Texture quality: {texture_quality}")
    
    # Check brightness difference between stereo pair
    brightness_diff = abs(left_mean - right_mean)
    print(f"Stereo brightness difference: {brightness_diff:.1f}")
    
    if brightness_diff > 20:
        print("⚠️  Large brightness difference - may need normalization")
    
    return {
        'pair_name': pair_name,
        'resolution': (left_img.shape[1], left_img.shape[0]),
        'lighting': lighting,
        'avg_brightness': avg_brightness,
        'texture_measure': texture_measure,
        'texture_quality': texture_quality,
        'brightness_diff': brightness_diff,
        'left_contrast': left_std,
        'right_contrast': right_std
    }

def main():
    """Main analysis function."""
    print("🔍 DATASET ANALYSIS FOR ADVANCED STEREO VISION PIPELINE")
    print("=" * 60)
    
    # Analyze representative image pairs
    pairs_to_analyze = [
        ("000001_left.png", "000001_right.png", "Pair 001"),
        ("000003_left.png", "000003_right.png", "Pair 003"),
        ("000006_left.png", "000006_right.png", "Pair 006"),
        ("000009_left.png", "000009_right.png", "Pair 009"),
        ("000011_left.png", "000011_right.png", "Pair 011")
    ]
    
    results = []
    base_path = Path("data/dataset1/rgb")
    
    for left_name, right_name, pair_name in pairs_to_analyze:
        left_path = base_path / left_name
        right_path = base_path / right_name
        
        result = analyze_image_pair(str(left_path), str(right_path), pair_name)
        if result:
            results.append(result)
    
    # Summary analysis
    if results:
        print(f"\n📊 DATASET SUMMARY ({len(results)} pairs analyzed)")
        print("=" * 60)
        
        # Lighting distribution
        lighting_counts = {}
        texture_scores = []
        brightness_diffs = []
        
        for result in results:
            lighting = result['lighting']
            lighting_counts[lighting] = lighting_counts.get(lighting, 0) + 1
            texture_scores.append(result['texture_measure'])
            brightness_diffs.append(result['brightness_diff'])
        
        print("Lighting conditions:")
        for lighting, count in lighting_counts.items():
            print(f"  - {lighting}: {count} pairs")
        
        avg_texture = np.mean(texture_scores)
        avg_brightness_diff = np.mean(brightness_diffs)
        
        print(f"\nAverage texture richness: {avg_texture:.1f}")
        print(f"Average stereo brightness difference: {avg_brightness_diff:.1f}")
        
        # Dataset quality assessment
        print(f"\n🎯 DATASET QUALITY ASSESSMENT")
        print("=" * 60)
        
        # Check for day/night diversity
        has_day = any("Day" in result['lighting'] or "Bright" in result['lighting'] for result in results)
        has_night = any("Dark" in result['lighting'] or "Night" in result['lighting'] for result in results)
        
        print(f"✅ Day images: {'Yes' if has_day else 'No'}")
        print(f"✅ Night images: {'Yes' if has_night else 'No'}")
        
        if has_day and has_night:
            print("✅ Good lighting diversity for day/night testing")
        else:
            print("⚠️  Limited lighting diversity - consider adding more varied conditions")
        
        # Texture assessment
        if avg_texture > 15:
            print("✅ Good texture richness for stereo matching")
        elif avg_texture > 10:
            print("⚠️  Moderate texture - advanced algorithms needed")
        else:
            print("❌ Low texture - challenging for stereo matching")
        
        # Stereo consistency
        if avg_brightness_diff < 15:
            print("✅ Good stereo pair consistency")
        else:
            print("⚠️  Stereo pairs may need brightness normalization")
        
        # Resolution check
        resolution = results[0]['resolution']
        if resolution[0] >= 1200 and resolution[1] >= 300:
            print("✅ Good resolution for detailed analysis")
        else:
            print("⚠️  Low resolution may limit accuracy")
        
        print(f"\n🚀 RECOMMENDATIONS FOR ADVANCED PIPELINE")
        print("=" * 60)
        
        if not (has_day and has_night):
            print("📸 Add more images with diverse lighting conditions")
            print("   - Capture potholes in bright daylight")
            print("   - Capture potholes under street lighting at night")
            print("   - Include dawn/dusk transition periods")
        
        if avg_texture < 15:
            print("🔧 Enable advanced disparity algorithms:")
            print("   - Use SGBM with optimized parameters")
            print("   - Apply WLS filtering for smooth regions")
            print("   - Consider deep learning methods for challenging areas")
        
        if avg_brightness_diff > 10:
            print("⚖️  Enable brightness normalization in preprocessing")
        
        print("🎯 Current dataset is suitable for:")
        print("   - Algorithm development and testing")
        print("   - Proof of concept validation")
        print("   - Baseline performance measurement")
        
        if has_day and has_night and avg_texture > 10:
            print("✅ Dataset is GOOD for advanced stereo vision development!")
        else:
            print("⚠️  Dataset is ADEQUATE but could be improved for production use")

if __name__ == "__main__":
    main()