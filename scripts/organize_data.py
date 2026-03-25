#!/usr/bin/env python3

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import sys


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataOrganizer:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        
        self.deepfashion_dir = self.base_dir / "deepfashion"
        self.deepfashion_original = self.deepfashion_dir / "original"
        self.deepfashion_synthetic = self.deepfashion_dir / "synthetic_degraded"
        self.deepfashion_metadata = self.deepfashion_dir / "metadata"
        
        self.scraped_dir = self.base_dir / "scraped"
        self.poshmark_dir = self.scraped_dir / "poshmark"
        self.thredup_dir = self.scraped_dir / "thredup"
        self.depop_dir = self.scraped_dir / "depop"
        self.unified_dir = self.scraped_dir / "unified"
        
        self.processed_dir = self.base_dir / "processed"
        self.train_dir = self.processed_dir / "train"
        self.val_dir = self.processed_dir / "val"
        self.test_dir = self.processed_dir / "test"
        
        self.stats = defaultdict(int)

    def link_or_copy_path(self, source: Path, target: Path, use_symlinks: bool = True):
        if not source.exists():
            logger.warning(f"Source path not found, skipping: {source}")
            return

        if target.exists() or target.is_symlink():
            logger.info(f"Target already exists: {target}")
            return

        if use_symlinks:
            try:
                target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
                logger.info(f"Created symlink: {target} -> {source.resolve()}")
                return
            except Exception as e:
                logger.warning(f"Failed to create symlink for {source}: {e}")
                logger.info("Falling back to copying...")

        if source.is_dir():
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        logger.info(f"Copied: {source} -> {target}")

    def expose_classic_deepfashion_layout(self, raw_deepfashion: Path, use_symlinks: bool = True):
        classic_root = raw_deepfashion / "original"
        if not classic_root.exists():
            return

        logger.info("Exposing classic DeepFashion layout at canonical paths...")
        canonical_items = ["Anno", "Eval", "img", "img_highres", "README.txt"]
        for item_name in canonical_items:
            source = classic_root / item_name
            target = self.deepfashion_dir / item_name
            self.link_or_copy_path(source, target, use_symlinks=use_symlinks)
        
    def create_directory_structure(self, force: bool = False):
        logger.info("Creating directory structure...")
        
        directories = [
            self.deepfashion_original,
            self.deepfashion_synthetic,
            self.deepfashion_metadata,
            self.poshmark_dir,
            self.thredup_dir,
            self.depop_dir,
            self.unified_dir,
            self.train_dir,
            self.val_dir,
            self.test_dir,
        ]
        
        for directory in directories:
            if force and directory.exists():
                logger.warning(f"Removing existing directory: {directory}")
                shutil.rmtree(directory)
            
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f" Created: {directory}")
        
        logger.info("Directory structure created successfully!")
    
    def organize_deepfashion_data(self, use_symlinks: bool = True, source_dir: Optional[str] = None):
        logger.info("Organizing DeepFashion data...")
        
        if source_dir is None:
            raw_deepfashion = self.base_dir / "raw" / "deepfashion"
        else:
            raw_deepfashion = Path(source_dir)
        
        if not raw_deepfashion.exists():
            logger.warning(f"Source directory not found: {raw_deepfashion}")
            logger.info("No DeepFashion data to organize")
            return
        
        self.deepfashion_dir.mkdir(parents=True, exist_ok=True)
        self.deepfashion_original.mkdir(parents=True, exist_ok=True)
        self.expose_classic_deepfashion_layout(raw_deepfashion, use_symlinks=use_symlinks)
        
        source_dirs = [d for d in raw_deepfashion.iterdir() if d.is_dir()]
        
        if not source_dirs:
            logger.warning(f"No subdirectories found in {raw_deepfashion}")
            return
        
        logger.info(f"Found {len(source_dirs)} source directories:")
        for src_dir in source_dirs:
            logger.info(f"  - {src_dir.name}")
        
        files_organized = 0
        dirs_organized = 0
        
        for src_dir in source_dirs:
            if src_dir.name in ['original', 'synthetic_degraded', 'metadata']:
                if src_dir.name != "original":
                    logger.info(f"Skipping {src_dir.name} (already organized)")
                    continue
            
            target_name = "classic" if src_dir.name == "original" else src_dir.name
            target_subdir = self.deepfashion_original / target_name
            
            if target_subdir.exists():
                logger.info(f"Target already exists: {target_subdir}")
                continue
            
            if use_symlinks:
                try:
                    abs_src = src_dir.resolve()
                    target_subdir.symlink_to(abs_src)
                    logger.info(f"Created symlink: {target_subdir} -> {abs_src}")
                    dirs_organized += 1
                except Exception as e:
                    logger.error(f"Failed to create symlink for {src_dir.name}: {e}")
                    logger.info(f"Falling back to copying...")
                    try:
                        shutil.copytree(src_dir, target_subdir)
                        logger.info(f"Copied: {src_dir} -> {target_subdir}")
                        dirs_organized += 1
                    except Exception as copy_error:
                        logger.error(f"Failed to copy {src_dir.name}: {copy_error}")
            else:
                try:
                    logger.info(f"Copying {src_dir.name}... (this may take a while)")
                    shutil.copytree(src_dir, target_subdir)
                    
                    file_count = sum(1 for _ in target_subdir.rglob("*") if _.is_file())
                    files_organized += file_count
                    
                    logger.info(f"Copied: {src_dir.name} ({file_count} files)")
                    dirs_organized += 1
                except Exception as e:
                    logger.error(f"Failed to copy {src_dir.name}: {e}")
        
        logger.info(f"DeepFashion data organization complete!")
        logger.info(f"Directories organized: {dirs_organized}")
        if not use_symlinks:
            logger.info(f"Files copied: {files_organized}")
        logger.info(f"Target location: {self.deepfashion_original}")
        
    def verify_data_sources(self) -> Dict[str, bool]:
        logger.info("Verifying data sources...")
        
        sources = {
            "deepfashion_original": self.deepfashion_original.exists() and any(self.deepfashion_original.iterdir()),
            "deepfashion_synthetic": self.deepfashion_synthetic.exists() and any(self.deepfashion_synthetic.iterdir()),
            "poshmark": self.poshmark_dir.exists() and any(self.poshmark_dir.glob("*.json")),
            "thredup": self.thredup_dir.exists() and any(self.thredup_dir.glob("*.json")),
            "depop": self.depop_dir.exists() and any(self.depop_dir.glob("*.json")),
        }
        
        for source, available in sources.items():
            status = "Available" if available else " Not Found"
            logger.info(f"{source}: {status}")
        
        return sources
        
    def inventory_deepfashion(self) -> Dict:
        logger.info("Creating DeepFashion inventory...")
        
        inventory = {
            "original": {
                "count": 0,
                "size_bytes": 0,
                "extensions": Counter()
            },
            "synthetic": {
                "count": 0,
                "size_bytes": 0,
                "extensions": Counter(),
                "condition_levels": Counter()
            }
        }
        
        if self.deepfashion_original.exists():
            for img_path in self.deepfashion_original.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    inventory["original"]["count"] += 1
                    inventory["original"]["size_bytes"] += img_path.stat().st_size
                    inventory["original"]["extensions"][img_path.suffix.lower()] += 1
        
        if self.deepfashion_synthetic.exists():
            for img_path in self.deepfashion_synthetic.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    inventory["synthetic"]["count"] += 1
                    inventory["synthetic"]["size_bytes"] += img_path.stat().st_size
                    inventory["synthetic"]["extensions"][img_path.suffix.lower()] += 1
                    
                    if "condition" in img_path.stem.lower():
                        try:
                            condition = int(img_path.stem.split("condition_")[1].split("_")[0])
                            inventory["synthetic"]["condition_levels"][condition] += 1
                        except (IndexError, ValueError):
                            pass
        
        metadata_files = list(self.deepfashion_metadata.glob("*.json")) if self.deepfashion_metadata.exists() else []
        inventory["metadata_files"] = len(metadata_files)
        
        logger.info(f"DeepFashion Original: {inventory['original']['count']} images")
        logger.info(f"DeepFashion Synthetic: {inventory['synthetic']['count']} images")
        logger.info(f"Metadata files: {inventory['metadata_files']}")
        
        return inventory
        
    def inventory_scraped_data(self) -> Dict:
        logger.info("Creating scraped data inventory...")
        inventory = {
            "poshmark": self.count_scraped_items(self.poshmark_dir),
            "thredup": self.count_scraped_items(self.thredup_dir),
            "depop": self.count_scraped_items(self.depop_dir),
        }
        total = sum(inv["count"] for inv in inventory.values())
        logger.info(f"Total scraped items: {total}")
        
        return inventory
        
    def count_scraped_items(self, directory: Path) -> Dict:
        if not directory.exists():
            return {"count": 0, "has_batch": False, "batch_count": 0}
        
        json_files = list(directory.glob("*.json"))
        count = len([f for f in json_files if not f.stem.startswith("batch_")])
        
        batch_dir = directory / "batches"
        batch_files = list(batch_dir.glob("batch_*.json")) if batch_dir.exists() else []
        
        return {
            "count": count,
            "has_batch": len(batch_files) > 0,
            "batch_count": len(batch_files)
        }
        
    def unify_scraped_data(self, output_format: str = "json"):
        logger.info("Unifying scraped data from all platforms...")
        unified_data = []
        platforms = [
            ("poshmark", self.poshmark_dir),
            ("thredup", self.thredup_dir),
            ("depop", self.depop_dir)
        ]
        
        for platform, directory in platforms:
            if not directory.exists():
                logger.warning(f"Directory not found: {directory}")
                continue
                
            logger.info(f"Processing {platform} data...")
            
            for json_file in directory.glob("*.json"):
                if json_file.stem.startswith("batch_") or json_file.stem in ["scraping_stats", "data_summary"]:
                    continue
                    
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        item_data = json.load(f)
                    
                    unified_item = self.standardize_item(item_data, platform)
                    unified_data.append(unified_item)
                    
                except Exception as e:
                    logger.error(f"Error processing {json_file}: {e}")
            
            batch_dir = directory / "batches"
            if batch_dir.exists():
                for batch_file in batch_dir.glob("batch_*.json"):
                    try:
                        with open(batch_file, 'r', encoding='utf-8') as f:
                            batch_data = json.load(f)
                        
                        if isinstance(batch_data, list):
                            items = batch_data
                        elif isinstance(batch_data, dict) and "items" in batch_data:
                            items = batch_data["items"]
                        else:
                            continue
                        
                        for item_data in items:
                            unified_item = self.standardize_item(item_data, platform)
                            unified_data.append(unified_item)
                            
                    except Exception as e:
                        logger.error(f"Error processing {batch_file}: {e}")
        
        logger.info(f"Unified {len(unified_data)} items from all platforms")
        self.unified_dir.mkdir(parents=True, exist_ok=True)
        if output_format in ["json", "both"]:
            output_file = self.unified_dir / "unified_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(unified_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved unified JSON: {output_file}")
        
        if output_format in ["csv", "both"]:
            try:
                import pandas as pd
                df = pd.DataFrame(unified_data)
                output_file = self.unified_dir / "unified_data.csv"
                df.to_csv(output_file, index=False, encoding='utf-8')
                logger.info(f"Saved unified CSV: {output_file}")
            except ImportError:
                logger.warning("pandas not available, skipping CSV export")
        
        stats = {
            "total_items": len(unified_data),
            "platforms": {
                platform: len([item for item in unified_data if item["platform"] == platform])
                for platform, _ in platforms
            },
            "price_stats": self.calculate_price_stats(unified_data),
            "condition_distribution": Counter([item.get("condition") for item in unified_data if item.get("condition")]),
        }
        
        stats_file = self.unified_dir / "unified_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics: {stats_file}")
        
        return unified_data
        
    def standardize_item(self, item_data: Dict, platform: str) -> Dict:
        standardized = {
            "id": item_data.get("id", item_data.get("item_id", "")),
            "platform": platform,
            "title": item_data.get("title", ""),
            "description": item_data.get("description", ""),
            "price": self.parse_price(item_data.get("price")),
            "condition": self.standardize_condition(item_data.get("condition", "")),
            "brand": item_data.get("brand", ""),
            "category": item_data.get("category", ""),
            "images": item_data.get("images", []),
            "url": item_data.get("url", ""),
            "scraped_at": item_data.get("scraped_at", ""),
        }
        
        return standardized
        
    def parse_price(self, price_value) -> Optional[float]:
        if price_value is None:
            return None
            
        if isinstance(price_value, (int, float)):
            return float(price_value)
            
        if isinstance(price_value, str):
            price_str = price_value.replace('$', '').replace(',', '').strip()
            try:
                return float(price_str)
            except ValueError:
                return None
        
        return None
        
    def standardize_condition(self, condition: str) -> str:
        if not condition:
            return "unknown"
            
        condition_lower = condition.lower()
        
        if "new" in condition_lower or "nwt" in condition_lower:
            return "new_with_tags"
        elif "like new" in condition_lower or "excellent" in condition_lower:
            return "like_new"
        elif "good" in condition_lower:
            return "good"
        elif "fair" in condition_lower:
            return "fair"
        elif "poor" in condition_lower or "wear" in condition_lower:
            return "poor"
        else:
            return "unknown"
            
    def calculate_price_stats(self, items: List[Dict]) -> Dict:
        prices = [item["price"] for item in items if item.get("price") is not None]
        
        if not prices:
            return {"count": 0}
        
        import statistics
        
        return {
            "count": len(prices),
            "min": min(prices),
            "max": max(prices),
            "mean": statistics.mean(prices),
            "median": statistics.median(prices),
            "stdev": statistics.stdev(prices) if len(prices) > 1 else 0
        }
        
    def generate_summary_report(self, output_file: Optional[str] = None) -> Dict:
        logger.info("Generating summary report...")
        report = {
            "timestamp": self.get_timestamp(),
            "data_sources": self.verify_data_sources(),
            "deepfashion": self.inventory_deepfashion(),
            "scraped": self.inventory_scraped_data(),
            "directory_structure": self.get_directory_tree(),
        }
        report["totals"] = {
            "deepfashion_images": report["deepfashion"]["original"]["count"] + report["deepfashion"]["synthetic"]["count"],
            "scraped_items": sum(inv["count"] for inv in report["scraped"].values()),
        }
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Summary report saved to: {output_path}")
        
        return report
        
    def get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
        
    def get_directory_tree(self) -> Dict:
        def get_tree(path: Path, max_depth: int = 3, current_depth: int = 0) -> Dict:
            if current_depth >= max_depth or not path.exists():
                return {}
            
            tree = {}
            try:
                for item in path.iterdir():
                    if item.is_dir():
                        tree[item.name] = get_tree(item, max_depth, current_depth + 1)
                    else:
                        tree[item.name] = "file"
            except PermissionError:
                pass
            
            return tree
        
        return get_tree(self.base_dir)
        
    def print_summary(self):
        logger.info("DATA ORGANIZATION SUMMARY")
        
        sources = self.verify_data_sources()
        logger.info("Data Sources:")
        for source, available in sources.items():
            status = "" if available else ""
            logger.info(f"{status} {source}")
        
        df_inv = self.inventory_deepfashion()
        logger.info("DeepFashion:")
        logger.info(f"Original: {df_inv['original']['count']:,} images")
        logger.info(f"Synthetic: {df_inv['synthetic']['count']:,} images")
        
        scraped_inv = self.inventory_scraped_data()
        logger.info("Scraped Data:")
        for platform, stats in scraped_inv.items():
            logger.info(f"{platform}: {stats['count']:,} items")
        
        total = (df_inv['original']['count'] + df_inv['synthetic']['count'] + 
                sum(s['count'] for s in scraped_inv.values()))
        logger.info(f"Total Items: {total:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize data from multiple sources into unified structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create directory structure
  python organize_data.py --create-dirs
  
  # Organize DeepFashion data from data/raw/deepfashion to data/deepfashion/original
  python organize_data.py --organize-deepfashion
  
  # Organize with copying instead of symbolic links
  python organize_data.py --organize-deepfashion --copy
  
  # Verify data sources
  python organize_data.py --verify
  
  # Unify scraped data
  python organize_data.py --unify --format both
  
  # Generate full summary report
  python organize_data.py --summary --output data/summary_report.json
  
  # Do everything
  python organize_data.py --all
        """
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data",
        help="Base directory for all data (default: data)"
    )
    
    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create the complete directory structure"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of directories (removes existing)"
    )
    
    parser.add_argument(
        "--organize-deepfashion",
        action="store_true",
        help="Organize DeepFashion data from data/raw/deepfashion to data/deepfashion/original"
    )
    
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symbolic links (requires more disk space)"
    )
    
    parser.add_argument(
        "--deepfashion-source",
        type=str,
        default=None,
        help="Custom source directory for DeepFashion data (default: data/raw/deepfashion)"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify which data sources are available"
    )
    
    parser.add_argument(
        "--unify",
        action="store_true",
        help="Unify scraped data from all platforms"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format for unified data (default: both)"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate summary report"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for summary report (JSON)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all operations (create dirs, unify, summary)"
    )
    
    args = parser.parse_args()
    
    organizer = DataOrganizer(base_dir=args.base_dir)
    if args.all:
        args.create_dirs = True
        args.organize_deepfashion = True
        args.verify = True
        args.unify = True
        args.summary = True
        if not args.output:
            args.output = "data/data_organization_report.json"
    
    if args.create_dirs:
        organizer.create_directory_structure(force=args.force)
    
    if args.organize_deepfashion:
        organizer.organize_deepfashion_data(
            use_symlinks=not args.copy,
            source_dir=args.deepfashion_source
        )
    
    if args.verify:
        organizer.verify_data_sources()
    
    if args.unify:
        organizer.unify_scraped_data(output_format=args.format)
    
    if args.summary:
        organizer.generate_summary_report(output_file=args.output)
        organizer.print_summary()
    
    if not any([args.create_dirs, args.organize_deepfashion, args.verify, args.unify, args.summary]):
        organizer.print_summary()


if __name__ == "__main__":
    main()
