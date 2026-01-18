#!/usr/bin/env python3
"""
CharuCo Board Generator

Generates a CharuCo calibration board for printing and camera calibration.
"""

import argparse
from pathlib import Path
from stereo_vision.calibration.charuco_calibrator import CharuCoCalibrator
from stereo_vision.utils.config_manager import ConfigManager


def main():
    """Generate CharuCo calibration board."""
    parser = argparse.ArgumentParser(
        description="Generate CharuCo calibration board for camera calibration"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="charuco_board.png",
        help="Output file path for the board image"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for printing (default: 300)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager(args.config) if args.config else ConfigManager()
    
    # Create calibrator
    calibrator = CharuCoCalibrator(config)
    
    # Generate board
    print("Generating CharuCo calibration board...")
    print(f"Board configuration: {calibrator.squares_x}x{calibrator.squares_y}")
    print(f"Square size: {calibrator.square_length * 1000:.1f}mm")
    print(f"Marker size: {calibrator.marker_length * 1000:.1f}mm")
    
    calibrator.generate_charuco_board(args.output, args.dpi)
    
    print(f"\n✅ CharuCo board saved to: {args.output}")
    print(f"📏 Print at {args.dpi} DPI for correct physical dimensions")
    print("\n📋 Calibration Instructions:")
    print("1. Print the board on a flat, rigid surface")
    print("2. Ensure the board is perfectly flat when capturing images")
    print("3. Capture 20-30 images from different angles and distances")
    print("4. Include images with the board at image edges for distortion modeling")
    print("5. Vary pitch and yaw by up to 45 degrees")
    print("6. Ensure good lighting and sharp focus")


if __name__ == "__main__":
    main()