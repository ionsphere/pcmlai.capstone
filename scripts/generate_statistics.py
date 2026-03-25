#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StatisticsGenerator:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        self.stats = []
        
    def generate_all_statistics(self, output_path: Path):
        logger.info("Generating DeepFashion Dataset Statistics")
        logger.info(f"Processed data directory: {self.processed_dir}")
        
        csv_files = list(self.processed_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files\n")
        for csv_file in csv_files:
            logger.info(f"Processing: {csv_file.name}")
            self.process_csv(csv_file)
            logger.info("")
        
        if self.stats:
            self.save_statistics(output_path)
            self.print_summary()
        else:
            logger.warning("No statistics generated")
    
    def process_csv(self, csv_path: Path):
        try:
            df = pd.read_csv(csv_path)
            dataset_name = csv_path.stem
            logger.info(f"Total samples: {len(df)}")
            logger.info(f"Columns: {list(df.columns)}")
            self.stats.append({
                "metric_name": "total_samples",
                "metric_value": len(df),
                "dataset": dataset_name,
                "subset": "all",
                "split": "all"
            })
            
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            self.stats.append({
                "metric_name": "memory_usage_mb",
                "metric_value": round(memory_mb, 2),
                "dataset": dataset_name,
                "subset": "all",
                "split": "all"
            })
            self.stats.append({
                "metric_name": "column_count",
                "metric_value": len(df.columns),
                "dataset": dataset_name,
                "subset": "all",
                "split": "all"
            })

            if "split" in df.columns:
                logger.info(f"Split distribution:")
                for split, count in df["split"].value_counts().items():
                    logger.info(f" - {split}: {count}")
                    self.stats.append({
                        "metric_name": "sample_count",
                        "metric_value": count,
                        "dataset": dataset_name,
                        "subset": "all",
                        "split": str(split)
                    })
            
            if "subset" in df.columns:
                logger.info(f"Subset distribution:")
                for subset, count in df["subset"].value_counts().items():
                    logger.info(f" - {subset}: {count}")
                    self.stats.append({
                        "metric_name": "sample_count",
                        "metric_value": count,
                        "dataset": dataset_name,
                        "subset": str(subset),
                        "split": "all"
                    })
            
            if "source_dataset" in df.columns:
                logger.info(f"Source distribution:")
                for source, count in df["source_dataset"].value_counts().items():
                    logger.info(f" - {source}: {count}")
            
            category_col = None
            for col in ["category_name", "category", "category_id"]:
                if col in df.columns:
                    category_col = col
                    break
            
            if category_col:
                unique_categories = df[category_col].nunique()
                logger.info(f"Unique categories: {unique_categories}")
                self.stats.append({
                    "metric_name": "unique_categories",
                    "metric_value": unique_categories,
                    "dataset": dataset_name,
                    "subset": "all",
                    "split": "all"
                })
                
                top_5 = df[category_col].value_counts().head(5)
                logger.info(f"Top 5 categories:")
                for cat, count in top_5.items():
                    logger.info(f" - {cat}: {count}")
            
            missing = df.isnull().sum()
            if missing.sum() > 0:
                logger.info(f"Missing values:")
                for col, count in missing[missing > 0].items():
                    pct = (count / len(df)) * 100
                    logger.info(f" - {col}: {count} ({pct:.1f}%)")
                    self.stats.append({
                        "metric_name": f"missing_{col}",
                        "metric_value": count,
                        "dataset": dataset_name,
                        "subset": "all",
                        "split": "all"
                    })
            
            if "image_path" in df.columns:
                valid_paths = df["image_path"].notna().sum()
                logger.info(f"Valid image paths: {valid_paths}")
                self.stats.append({
                    "metric_name": "valid_image_paths",
                    "metric_value": valid_paths,
                    "dataset": dataset_name,
                    "subset": "all",
                    "split": "all"
                })
            
        except Exception as e:
            logger.error(f"Error processing {csv_path.name}: {e}")
    
    def save_statistics(self, output_path: Path):
        df = pd.DataFrame(self.stats)
        df = df.sort_values(["dataset", "metric_name", "split"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Statistics saved to: {output_path}")
    
    def print_summary(self):
        df = pd.DataFrame(self.stats)
        logger.info("STATISTICS SUMMARY")
        total_samples = df[
            (df["metric_name"] == "total_samples") &
            (df["split"] == "all")
        ]["metric_value"].sum()
        
        logger.info(f"Total samples across all datasets: {int(total_samples)}")
        logger.info("Dataset breakdown:")
        for dataset in df["dataset"].unique():
            dataset_df = df[
                (df["dataset"] == dataset) &
                (df["metric_name"] == "total_samples") &
                (df["split"] == "all")
            ]
            if not dataset_df.empty:
                count = int(dataset_df["metric_value"].values[0])
                logger.info(f" - {dataset}: {count}")
        
        split_df = df[
            (df["metric_name"] == "sample_count") &
            (df["split"] != "all")
        ]
        if not split_df.empty:
            logger.info("Split breakdown:")
            for split in split_df["split"].unique():
                count = split_df[split_df["split"] == split]["metric_value"].sum()
                logger.info(f" - {split}: {int(count)}")
        
        logger.info(f"Total metrics generated: {len(self.stats)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate statistics for processed DeepFashion datasets",
    )
    
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory containing processed CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for statistics CSV",
    )
    
    args = parser.parse_args()
    
    if not args.processed_dir.exists():
        logger.error(f"Processed directory does not exist: {args.processed_dir}")
        return 1
    
    generator = StatisticsGenerator(args.processed_dir)
    generator.generate_all_statistics(args.output)
    logger.info("Statistics generation complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
