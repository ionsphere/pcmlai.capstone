#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_kaggle_credentials() -> bool:
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    
    if not username or not key:
        logger.error("Kaggle credentials not found!")
        logger.error("Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        logger.error("Or configure them in your .env file")
        return False
    
    logger.info(f"Kaggle credentials found for user: {username}")
    return True


def download_kaggle_dataset(
    dataset: str,
    output_dir: Path,
    unzip: bool = True,
    force: bool = False,
) -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initializing Kaggle API...")
        api = KaggleApi()
        api.authenticate()
        logger.info(" Authentication successful")
        
        if not force:
            existing_files = list(output_dir.glob("*"))
            if existing_files:
                logger.warning(f"Output directory already contains {len(existing_files)} files")
                logger.warning("Use --force to re-download")
                response = input("Continue anyway? [y/N]: ")
                if response.lower() != 'y':
                    logger.info("Download cancelled")
                    return False
        
        logger.info(f"Downloading dataset: {dataset}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("This may take a while depending on dataset size...")
        
        api.dataset_download_files(
            dataset,
            path=str(output_dir),
            unzip=unzip,
            quiet=False,
        )
        
        logger.info(" Download complete!")
        
        downloaded_files = list(output_dir.rglob("*"))
        file_count = len([f for f in downloaded_files if f.is_file()])
        dir_count = len([f for f in downloaded_files if f.is_dir()])
        
        logger.info(f"Downloaded: {file_count} files in {dir_count} directories")
        
        logger.info("\nTop-level structure:")
        for item in sorted(output_dir.iterdir())[:10]:
            if item.is_dir():
                file_count = len(list(item.rglob("*")))
                logger.info(f"   deg {item.name}/ ({file_count} items)")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                logger.info(f"   deg {item.name} ({size_mb:.1f} MB)")
        
        return True
        
    except ImportError:
        logger.error("Kaggle package not installed!")
        logger.error("Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.exception("Full traceback:")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download DeepFashion datasets from Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_kaggle_datasets.py --dataset thusharanair/deepfashion2-original-with-dataframes --output ./data/raw/deepfashion/kaggle_thushan
  python download_kaggle_datasets.py --dataset vishalbsadanand/deepfashion-1 --output ./data/raw/deepfashion/kaggle_vishal
        """,
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Kaggle dataset identifier (e.g., 'user/dataset-name')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Don't unzip downloaded files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )
    
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    if not check_kaggle_credentials():
        sys.exit(1)
    
    success = download_kaggle_dataset(
        dataset=args.dataset,
        output_dir=args.output,
        unzip=not args.no_unzip,
        force=args.force,
    )
    
    if success:
        logger.info(" Dataset download successful!")
        sys.exit(0)
    else:
        logger.error(" Dataset download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
