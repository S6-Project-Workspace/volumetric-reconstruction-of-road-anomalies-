"""
Main entry point for Advanced Stereo Vision Pipeline

This will be the integrated pipeline controller (Task 14.1)
"""

import argparse
import sys
from pathlib import Path
from stereo_vision.utils.config_manager import ConfigManager


def main():
    """Main entry point for the stereo vision pipeline."""
    parser = argparse.ArgumentParser(
        description="Advanced Stereo Vision Pipeline for Road Anomaly Detection"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--input-dir", 
        type=str, 
        required=True,
        help="Directory containing stereo image pairs"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--calibration-dir", 
        type=str,
        help="Directory containing calibration images (if not using existing calibration)"
    )
    
    parser.add_argument(
        "--batch", 
        action="store_true",
        help="Process all image pairs in batch mode"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = ConfigManager(args.config)
        print(f"Loaded configuration from: {config.config_path}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Advanced Stereo Vision Pipeline")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch processing: {args.batch}")
    
    # TODO: Implement full pipeline in Task 14.1
    print("\nPipeline implementation will be completed in Task 14.1")
    print("Current setup is complete and ready for development!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())