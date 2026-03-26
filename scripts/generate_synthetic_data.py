#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, Optional
from datetime import datetime
import time


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.synthetic import batch_generate_synthetic_data


def validate_args(args):
    errors = []
    input_path = Path(args.input_dir)
    if not input_path.exists():
        errors.append(f"Input directory does not exist: {args.input_dir}")
    
    if args.condition_min > args.condition_max:
        errors.append(f"condition-min ({args.condition_min}) must be <= condition-max ({args.condition_max})")
    
    if args.num_variations < 1:
        errors.append(f"num-variations must be >= 1, got {args.num_variations}")
    
    if args.max_images is not None and args.max_images < 1:
        errors.append(f"max-images must be >= 1, got {args.max_images}")
    
    return errors


def estimate_generation(input_dir: str, num_variations: int, max_images: int = None):
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in input_path.rglob('*') 
        if f.suffix.lower() in image_extensions
    ]
    
    if max_images:
        image_files = image_files[:max_images]
    
    total_source = len(image_files)
    total_generated = total_source * num_variations
    estimated_size_mb = (total_generated * 200) / 1024
    estimated_time_min = (total_generated * 0.5) / 60
    
    return {
        'total_source': total_source,
        'total_generated': total_generated,
        'estimated_size_mb': estimated_size_mb,
        'estimated_time_min': estimated_time_min
    }


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic clothing degradation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/generate_synthetic_data.py --input-dir data/deepfashion/original --output-dir data/deepfashion/synthetic_degraded --num-variations 5 --condition-min 3 --condition-max 8
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing source images'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save generated images'
    )
    
    parser.add_argument(
        '--num-variations',
        type=int,
        default=5,
        help='Number of variations to generate per image (default: 5)'
    )
    
    parser.add_argument(
        '--condition-min',
        type=int,
        default=1,
        choices=range(1, 11),
        metavar='[1-10]',
        help='Minimum condition level (default: 1)'
    )
    
    parser.add_argument(
        '--condition-max',
        type=int,
        default=10,
        choices=range(1, 11),
        metavar='[1-10]',
        help='Maximum condition level (default: 10)'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of source images to process (default: all)'
    )
    
    parser.add_argument(
        '--save-metadata',
        action='store_true',
        help='Save detailed metadata JSON file'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without actually processing'
    )
    
    args = parser.parse_args(argv)
    errors = validate_args(args)
    if errors:
        print("Validation Errors:")
        for error in errors:
            print(f"{error}")
        print()
        sys.exit(1)
    
    print("Estimating generation statistics...")
    estimates = estimate_generation(args.input_dir, args.num_variations, args.max_images)
    
    print(f"Source Images:        {estimates['total_source']:,}")
    print(f"Generated Images:     {estimates['total_generated']:,}")
    print(f"Estimated Size:       {estimates['estimated_size_mb']:.1f} MB")
    print(f"Estimated Time:       {estimates['estimated_time_min']:.1f} minutes")
    
    if estimates['total_source'] == 0:
        print("No images found in input directory!")
        sys.exit(1)
    
    if args.dry_run:
        print("Dry run complete. No files were generated.")
        print(f"Run without --dry-run to generate {estimates['total_generated']:,} images.")
        sys.exit(0)
    
    print("This will generate a large number of files.")
    response = input("   Continue? (y/N): ").strip().lower()
    
    if response != 'y':
        print("Generation cancelled by user.")
        sys.exit(0)
    
    print()
    print("Starting generation...")
    
    start_time = time.time()
    try:
        stats = batch_generate_synthetic_data(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_variations=args.num_variations,
            condition_range=(args.condition_min, args.condition_max),
            max_images=args.max_images
        )
        end_time = time.time()
        duration = end_time - start_time
        if args.save_metadata:
            metadata_path = Path(args.output_dir) / 'generation_metadata.json'
            
            metadata = {
                'generation_date': datetime.now().isoformat(),
                'configuration': {
                    'input_dir': args.input_dir,
                    'output_dir': args.output_dir,
                    'num_variations': args.num_variations,
                    'condition_range': [args.condition_min, args.condition_max],
                    'max_images': args.max_images
                },
                'statistics': stats,
                'duration_seconds': duration
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Metadata saved to: {metadata_path}")
        
        print("Synthetic degradation dataset generation complete!")
        
    except Exception as e:
        print(" Generation Failed!")
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
