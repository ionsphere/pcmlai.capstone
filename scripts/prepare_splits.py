#!/usr/bin/env python3

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Sequence, Optional, Dict, List, Tuple
from collections import defaultdict
import random
import hashlib


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DEFAULT_BASE_DIR = "data"
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 42
DEFAULT_STRATIFY = "condition"
DEFAULT_REPORT_PATH = "data/processed/split_report.json"


class DataSplitter:
    def __init__(
        self,
        base_dir: str = DEFAULT_BASE_DIR,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        val_ratio: float = DEFAULT_VAL_RATIO,
        test_ratio: float = DEFAULT_TEST_RATIO,
        seed: int = DEFAULT_SEED
    ):
        self.base_dir = Path(base_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        random.seed(seed)
        
        self.deepfashion_dir = self.base_dir / "deepfashion"
        self.scraped_dir = self.base_dir / "scraped"
        self.processed_dir = self.base_dir / "processed"
        
        self.train_dir = self.processed_dir / "train"
        self.val_dir = self.processed_dir / "val"
        self.test_dir = self.processed_dir / "test"
        
        self.stats = {
            "train": defaultdict(int),
            "val": defaultdict(int),
            "test": defaultdict(int)
        }
        
    def prepare_directories(self, clean: bool = False):
        logger.info("Preparing directories for splits...")
        
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            if clean and split_dir.exists():
                logger.warning(f"Removing existing directory: {split_dir}")
                shutil.rmtree(split_dir)
            
            split_dir.mkdir(parents=True, exist_ok=True)
            
            (split_dir / "images").mkdir(exist_ok=True)
            (split_dir / "metadata").mkdir(exist_ok=True)
            
        logger.info("Directories prepared successfully!")
        
    def split_deepfashion_data(
        self,
        include_synthetic: bool = True,
        stratify_by: str = "condition"
    ):
        logger.info("Splitting DeepFashion data...")
        
        metadata_dir = self.deepfashion_dir / "metadata"
        metadata_file = metadata_dir / "dataset_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        images = []
        
        original_dir = self.deepfashion_dir / "original"
        if original_dir.exists():
            for img_path in original_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        images.append({
                            "path": img_path,
                            "type": "original",
                            "condition": 10,
                            "category": self.extract_category_from_path(img_path)
                        })
        
        if include_synthetic:
            synthetic_dir = self.deepfashion_dir / "synthetic_degraded"
            if synthetic_dir.exists():
                for img_path in synthetic_dir.rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        condition = self.extract_condition_from_filename(img_path.stem)
                        images.append({
                            "path": img_path,
                            "type": "synthetic",
                            "condition": condition,
                            "category": self.extract_category_from_path(img_path)
                        })
        
        logger.info(f"Found {len(images)} images to split")
        
        if not images:
            logger.warning("No images found to split!")
            return
        
        if stratify_by != "none":
            train, val, test = self.stratified_split(images, stratify_by)
        else:
            train, val, test = self.random_split(images)
        
        self.copy_files(train, self.train_dir, "train")
        self.copy_files(val, self.val_dir, "val")
        self.copy_files(test, self.test_dir, "test")
        logger.info(f"DeepFashion split complete: {len(train)} train, {len(val)} val, {len(test)} test")
        
    def split_scraped_data(
        self,
        stratify_by: str = "condition"
    ):
        logger.info("Splitting scraped marketplace data...")
        unified_file = self.scraped_dir / "unified" / "unified_data.json"
        if not unified_file.exists():
            logger.warning(f"Unified data file not found: {unified_file}")
            logger.warning("Run organize_data.py --unify first!")
            return
        
        with open(unified_file, 'r', encoding='utf-8') as f:
            items = json.load(f)
        
        logger.info(f"Found {len(items)} items to split")
        if not items:
            logger.warning("No items found to split!")
            return
        
        for item in items:
            item["price_range"] = self.get_price_range(item.get("price"))
        
        if stratify_by != "none":
            train, val, test = self.stratified_split(items, stratify_by)
        else:
            train, val, test = self.random_split(items)
        
        self.save_scraped_split(train, self.train_dir, "train")
        self.save_scraped_split(val, self.val_dir, "val")
        self.save_scraped_split(test, self.test_dir, "test")
        
        logger.info(f"Scraped data split complete: {len(train)} train, {len(val)} val, {len(test)} test")
        
    def stratified_split(
        self,
        items: List[Dict],
        stratify_key: str
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger.info(f"Performing stratified split by: {stratify_key}")
        groups = defaultdict(list)
        for item in items:
            key_value = item.get(stratify_key, "unknown")
            groups[key_value].append(item)
        
        logger.info(f"Found {len(groups)} groups for stratification")
        train, val, test = [], [], []
        for group_key, group_items in groups.items():
            random.shuffle(group_items)
            n = len(group_items)
            train_size = int(n * self.train_ratio)
            val_size = int(n * self.val_ratio)
            group_train = group_items[:train_size]
            group_val = group_items[train_size:train_size + val_size]
            group_test = group_items[train_size + val_size:]
            
            train.extend(group_train)
            val.extend(group_val)
            test.extend(group_test)
            
            logger.debug(f"Group '{group_key}': {len(group_train)} train, {len(group_val)} val, {len(group_test)} test")
        
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        
        return train, val, test
        
    def random_split(
        self,
        items: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger.info("Performing random split (no stratification)")
        items_copy = items.copy()
        random.shuffle(items_copy)
        n = len(items_copy)
        train_size = int(n * self.train_ratio)
        val_size = int(n * self.val_ratio)
        
        train = items_copy[:train_size]
        val = items_copy[train_size:train_size + val_size]
        test = items_copy[train_size + val_size:]
        
        return train, val, test
        
    def copy_files(self, items: List[Dict], target_dir: Path, split_name: str):
        logger.info(f"Copying {len(items)} files to {split_name} set...")
        images_dir = target_dir / "images"
        metadata_dir = target_dir / "metadata"
        metadata_list = []
        for idx, item in enumerate(items):
            source_path = item["path"]
            file_hash = hashlib.md5(str(source_path).encode()).hexdigest()[:8]
            new_filename = f"{split_name}_{idx:06d}_{file_hash}{source_path.suffix}"
            target_path = images_dir / new_filename
            try:
                shutil.copy2(source_path, target_path)
                metadata_list.append({
                    "id": idx,
                    "filename": new_filename,
                    "original_path": str(source_path),
                    "type": item.get("type"),
                    "condition": item.get("condition"),
                    "category": item.get("category"),
                })
                
                self.stats[split_name]["images"] += 1
                
            except Exception as e:
                logger.error(f"Error copying {source_path}: {e}")
        
        metadata_file = metadata_dir / "deepfashion_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2)
        
        logger.info(f"Copied {len(metadata_list)} images to {split_name} set")
        
    def save_scraped_split(self, items: List[Dict], target_dir: Path, split_name: str):
        logger.info(f"Saving {len(items)} scraped items to {split_name} set...")
        metadata_dir = target_dir / "metadata"
        metadata_file = metadata_dir / "scraped_data.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        
        self.stats[split_name]["scraped_items"] += len(items)
        logger.info(f"Saved {len(items)} scraped items to {split_name} set")
        
    def extract_category_from_path(self, path: Path) -> str:
        parts = path.parts
        categories = ["tops", "bottoms", "dresses", "outerwear", "shoes", "accessories"]
        for part in parts:
            part_lower = part.lower()
            for category in categories:
                if category in part_lower:
                    return category
        
        return "unknown"
        
    def extract_condition_from_filename(self, filename: str) -> int:
        if "condition" in filename.lower():
            try:
                parts = filename.lower().split("condition_")
                if len(parts) > 1:
                    condition_str = parts[1].split("_")[0]
                    return int(condition_str)
            except (IndexError, ValueError):
                pass
        
        return 10
        
    def get_price_range(self, price: Optional[float]) -> str:
        if price is None:
            return "unknown"
        
        if price < 20:
            return "under_20"
        elif price < 50:
            return "20_to_50"
        elif price < 100:
            return "50_to_100"
        elif price < 200:
            return "100_to_200"
        else:
            return "over_200"
            
    def generate_split_report(self, output_file: Optional[str] = None) -> Dict:
        logger.info("Generating split report...")
        report = {
            "split_ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio
            },
            "seed": self.seed,
            "statistics": dict(self.stats)
        }
        
        for split_name in ["train", "val", "test"]:
            split_dir = getattr(self, f"{split_name}_dir")
            metadata_dir = split_dir / "metadata"
            
            split_summary = {
                "images": 0,
                "scraped_items": 0,
                "metadata_files": []
            }
            
            images_dir = split_dir / "images"
            if images_dir.exists():
                split_summary["images"] = len(list(images_dir.glob("*")))
            
            if metadata_dir.exists():
                for meta_file in metadata_dir.glob("*.json"):
                    split_summary["metadata_files"].append(meta_file.name)
                    if meta_file.name == "scraped_data.json":
                        with open(meta_file, 'r') as f:
                            data = json.load(f)
                            split_summary["scraped_items"] = len(data)
            
            report[f"{split_name}_summary"] = split_summary
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Split report saved to: {output_path}")
        
        return report
        
    def print_summary(self):
        logger.info("DATA SPLIT SUMMARY")
        
        logger.info(f"Split Ratios:")
        logger.info(f"Train: {self.train_ratio * 100:.1f}%")
        logger.info(f"Val: {self.val_ratio * 100:.1f}%")
        logger.info(f"Test: {self.test_ratio * 100:.1f}%")
        logger.info(f"Seed: {self.seed}")
        
        for split_name in ["train", "val", "test"]:
            split_dir = getattr(self, f"{split_name}_dir")
            images_dir = split_dir / "images"
            metadata_dir = split_dir / "metadata"
            
            n_images = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
            n_scraped = 0
            
            scraped_file = metadata_dir / "scraped_data.json"
            if scraped_file.exists():
                with open(scraped_file, 'r') as f:
                    n_scraped = len(json.load(f))
            
            logger.info(f"{split_name.upper()} Set:")
            logger.info(f"Images: {n_images:,}")
            logger.info(f"Scraped Items: {n_scraped:,}")
            logger.info(f"Total: {n_images + n_scraped:,}")
        


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Prepare notebook train/val/test splits")
    parser.add_argument(
        "--deepfashion",
        action="store_true",
        help="Split DeepFashion data"
    )
    parser.add_argument(
        "--scraped",
        action="store_true",
        help="Split scraped marketplace data"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Split both DeepFashion and scraped data"
    )
    parser.add_argument(
        "--stratify",
        choices=["condition", "category", "price_range", "platform", "none"],
        default=DEFAULT_STRATIFY,
        help="Stratification key to use when splitting"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing splits before creating new ones"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help="Path for the split report JSON"
    )
    
    args = parser.parse_args(argv)

    if not any([args.deepfashion, args.scraped, args.all]):
        args.all = True
    
    splitter = DataSplitter()
    splitter.prepare_directories(clean=args.clean)
    
    if args.all:
        args.deepfashion = True
        args.scraped = True
    
    if args.deepfashion:
        splitter.split_deepfashion_data(
            include_synthetic=True,
            stratify_by=args.stratify if args.stratify in ["condition", "category", "none"] else "condition"
        )
    
    if args.scraped:
        splitter.split_scraped_data(
            stratify_by=args.stratify if args.stratify in ["condition", "price_range", "platform", "none"] else "condition"
        )
    
    splitter.generate_split_report(output_file=args.report)
    splitter.print_summary()


if __name__ == "__main__":
    main()
