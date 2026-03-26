#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence, Optional, Dict, List, Tuple

import pandas as pd
import yaml
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DeepFashionProcessor:
    def __init__(self, config_path: Path):
        self.config = self.load_config(config_path)
        self.master_inventory = []
        
    def load_config(self, config_path: Path) -> Dict:
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    
    def process_all(self, input_dir: Path, output_dir: Path):
        logger.info("Starting DeepFashion dataset processing...")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        thushan_dir = input_dir / "kaggle_thushan"
        if thushan_dir.exists():
            logger.info("Processing Kaggle Thushan (DeepFashion2) dataset...")
            self.process_kaggle_thushan(thushan_dir, output_dir)
        
        vishal_dir = input_dir / "kaggle_vishal"
        if vishal_dir.exists():
            logger.info("Processing Kaggle Vishal (DeepFashion-1) dataset...")
            self.process_kaggle_vishal(vishal_dir, output_dir)
        
        self.save_master_inventory(output_dir)
        logger.info("Processing complete!")
    
    def process_kaggle_thushan(self, input_dir: Path, output_dir: Path):
        logger.info("Looking for CSV dataframes...")
        train_csv = self.find_file(input_dir, ["train_df.csv", "train.csv"])
        val_csv = self.find_file(input_dir, ["validation_df.csv", "val.csv", "valid.csv"])
        if not train_csv:
            logger.warning("train_df.csv not found in Thushan dataset")
            return
        
        logger.info(f"Processing training data from: {train_csv.name}")
        train_df = pd.read_csv(train_csv)
        train_df["split"] = "train"
        train_df["source_dataset"] = "kaggle_thushan"
        train_df["subset"] = "deepfashion2"
        logger.info(f"Loaded {len(train_df)} training samples")
        logger.info(f"Columns: {list(train_df.columns)}")
        
        if val_csv:
            logger.info(f"Processing validation data from: {val_csv.name}")
            val_df = pd.read_csv(val_csv)
            val_df["split"] = "val"
            val_df["source_dataset"] = "kaggle_thushan"
            val_df["subset"] = "deepfashion2"
            logger.info(f"Loaded {len(val_df)} validation samples")
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            logger.warning("validation_df.csv not found")
            combined_df = train_df
        
        combined_df = self.standardize_columns(combined_df)
        output_path = output_dir / "deepfashion2_processed.csv"
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to: {output_path}")
        self.add_to_inventory(combined_df, "kaggle_thushan", "deepfashion2")
        if "category_name" in combined_df.columns or "category" in combined_df.columns:
            self.save_category_breakdown(combined_df, output_dir, "deepfashion2")
    
    def process_kaggle_vishal(self, input_dir: Path, output_dir: Path):
        logger.info("Looking for DeepFashion-1 structure...")
        anno_dir = self.find_dir(input_dir, ["Anno", "annotations", "anno"])
        if not anno_dir:
            logger.warning("Annotation directory not found")
            return
        
        logger.info(f"Found annotation directory: {anno_dir}")
        category_file = self.find_file(anno_dir, [
            "list_category_img.txt",
            "list_category_cloth.txt"
        ])
        
        if category_file:
            logger.info(f"Processing categories from: {category_file.name}")
            df = self.parse_deepfashion_annotations(category_file, anno_dir)
            if df is not None:
                df["source_dataset"] = "kaggle_vishal"
                df["subset"] = "category_attribute"
                output_path = output_dir / "deepfashion1_category_attribute.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"Saved processed data to: {output_path}")
                logger.info(f"Total samples: {len(df)}")
                self.add_to_inventory(df, "kaggle_vishal", "category_attribute")
    
    def parse_deepfashion_annotations(
        self,
        category_file: Path,
        anno_dir: Path
    ) -> Optional[pd.DataFrame]:
        try:
            with open(category_file, "r") as f:
                lines = f.readlines()
            
            data_lines = [l.strip() for l in lines[2:] if l.strip()]
            
            records = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 2:
                    image_path = parts[0]
                    category_id = int(parts[1])
                    records.append({
                        "image_path": image_path,
                        "category_id": category_id,
                    })
            
            df = pd.DataFrame(records)
            logger.info(f"Parsed {len(df)} annotations")
            
            partition_file = self.find_file(anno_dir.parent, [
                "Eval/list_eval_partition.txt",
                "list_eval_partition.txt"
            ])
            
            if partition_file:
                logger.info(f"Loading split information from: {partition_file.name}")
                df = self.add_split_info(df, partition_file)
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing annotations: {e}")
            return None
    
    def add_split_info(self, df: pd.DataFrame, partition_file: Path) -> pd.DataFrame:
        try:
            with open(partition_file, "r") as f:
                lines = f.readlines()
            
            split_map = {}
            for line in lines[2:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_path = parts[0]
                    split_code = parts[1]
                    split_name = {"train": "train", "val": "val", "test": "test"}.get(
                        split_code, split_code
                    )
                    split_map[image_path] = split_name
            
            df["split"] = df["image_path"].map(split_map)
            logger.info(f"Added split info for {df['split'].notna().sum()} samples")
            return df
        except Exception as e:
            logger.error(f"Error loading split info: {e}")
            return df
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        column_map = {
            "image_name": "image_path",
            "image": "image_path",
            "category": "category_name",
            "cat": "category_name",
        }
        
        df = df.rename(columns=column_map)
        
        if "image_id" not in df.columns and "image_path" in df.columns:
            df["image_id"] = df["image_path"].apply(
                lambda x: Path(x).stem if isinstance(x, str) else None
            )
        
        return df
    
    def add_to_inventory(self, df: pd.DataFrame, source: str, subset: str):
        for _, row in df.iterrows():
            self.master_inventory.append({
                "source_dataset": source,
                "subset": subset,
                "split": row.get("split", "unknown"),
                "image_path": row.get("image_path", ""),
                "image_id": row.get("image_id", ""),
            })
    
    def save_master_inventory(self, output_dir: Path):
        if not self.master_inventory:
            logger.warning("No inventory data to save")
            return
        
        df = pd.DataFrame(self.master_inventory)
        output_path = output_dir / "deepfashion_inventory.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved master inventory to: {output_path}")
    
    def save_category_breakdown(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        dataset_name: str
    ):
        category_col = "category_name" if "category_name" in df.columns else "category"
        
        if category_col not in df.columns:
            return
        
        category_stats = df.groupby(category_col).size().reset_index(name="count")
        category_stats = category_stats.sort_values("count", ascending=False)
        
        output_path = output_dir / f"{dataset_name}_category_stats.csv"
        category_stats.to_csv(output_path, index=False)
        
        logger.info(f"Saved category statistics to: {output_path}")
    
    def find_file(self, search_dir: Path, filenames: List[str]) -> Optional[Path]:
        for filename in filenames:
            file_path = search_dir / filename
            if file_path.exists():
                return file_path
            
            matches = list(search_dir.rglob(filename))
            if matches:
                return matches[0]
        
        return None
    
    def find_dir(self, search_dir: Path, dirnames: List[str]) -> Optional[Path]:
        for dirname in dirnames:
            dir_path = search_dir / dirname
            if dir_path.exists() and dir_path.is_dir():
                return dir_path
            
            matches = list(search_dir.rglob(dirname))
            matches = [m for m in matches if m.is_dir()]
            if matches:
                return matches[0]
        
        return None


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Process DeepFashion datasets to CSV format",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing downloaded datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for processed CSV files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deepfashion_sources.yaml"),
        help="Configuration file path",
    )
    
    args = parser.parse_args(argv)
    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        return 1
    
    processor = DeepFashionProcessor(args.config)
    processor.process_all(args.input, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
