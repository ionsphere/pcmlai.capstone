import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from PIL import Image
import random
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic import SyntheticDegradationPipeline


def create_mock_condition_dataset(output_dir: str, num_samples: int = 500) -> Dict[str, Path]:
    print(f"Creating mock condition dataset...")
    print(f"Output: {output_dir}")
    print(f"Samples per condition: {num_samples}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    data_records = []
    
    print(f"Generating synthetic images...")
    
    print(f"Generating pristine images (condition 10)...")
    for i in tqdm(range(num_samples), desc="Pristine"):
        color = tuple(np.random.randint(100, 256, 3))
        img = Image.new('RGB', (380, 380), color)
        
        img_path = images_dir / f"pristine_{i:04d}.jpg"
        img.save(img_path, quality=95)
        
        data_records.append({
            'image_path': str(img_path.relative_to(output_path)),
            'condition_score': 10.0,
            'condition_label': 'pristine',
            'degradation_type': 'none',
            'source': 'mock_pristine'
        })
    
    pipeline = SyntheticDegradationPipeline()
    for condition in range(1, 10):
        print(f"Generating degraded images (condition {condition})...")
        for i in tqdm(range(num_samples), desc=f"Condition {condition}"):
            color = tuple(np.random.randint(100, 256, 3))
            img = Image.new('RGB', (380, 380), color)
            
            severity = (10 - condition) / 9
            degradation_types = random.sample(
                ['fading', 'stains', 'tears', 'pilling', 'wrinkles', 'discoloration', 'fraying'],
                k=random.randint(1, 3)
            )
            
            degraded_img, metadata = pipeline.apply_random_degradation(
                img,
                condition_score=condition,
                num_effects=len(degradation_types)
            )
            
            img_path = images_dir / f"degraded_c{condition}_{i:04d}.jpg"
            degraded_img.save(img_path, quality=95)
            
            if condition <= 3:
                label = 'poor'
            elif condition <= 6:
                label = 'fair'
            elif condition <= 8:
                label = 'good'
            else:
                label = 'excellent'
            
            data_records.append({
                'image_path': str(img_path.relative_to(output_path)),
                'condition_score': float(condition),
                'condition_label': label,
                'degradation_type': ','.join(degradation_types),
                'source': 'mock_degraded'
            })
    
    print(f"Generated {len(data_records)} images")
    
    df = pd.DataFrame(data_records)
    print(f"Creating stratified splits...")
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['condition_score'].astype(int)
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['condition_score'].astype(int)
    )
    
    print(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    splits = {}
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_path = output_path / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        splits[split_name] = split_path
        print(f"Saved: {split_path}")
    
    stats = {
        'total_samples': len(df),
        'num_conditions': 10,
        'condition_distribution': df['condition_score'].value_counts().sort_index().to_dict(),
        'label_distribution': df['condition_label'].value_counts().to_dict(),
        'splits': {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df)
        }
    }
    
    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")
    
    return splits


def prepare_deepfashion_condition_dataset(
    deepfashion_dir: str,
    synthetic_dir: str,
    output_dir: str,
    pristine_label: float = 10.0
) -> Dict[str, Path]:
    print(f"Preparing DeepFashion condition dataset...")
    print(f"DeepFashion: {deepfashion_dir}")
    print(f"Synthetic: {synthetic_dir}")
    print(f"Output: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_records = []
    
    df_dir = Path(deepfashion_dir)
    if df_dir.exists():
        print(f"Processing original images (condition {pristine_label})...")
        image_files = list(df_dir.glob("**/*.jpg")) + list(df_dir.glob("**/*.png"))
        for img_path in tqdm(image_files[:10000], desc="Original"):  # Limit for memory
            data_records.append({
                'image_path': str(img_path),
                'condition_score': pristine_label,
                'condition_label': 'pristine',
                'degradation_type': 'none',
                'source': 'deepfashion_original'
            })
    
    syn_dir = Path(synthetic_dir)
    if syn_dir.exists():
        print(f"Processing synthetic degraded images...")
        metadata_path = syn_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            for item in tqdm(metadata, desc="Synthetic"):
                data_records.append({
                    'image_path': item['output_path'],
                    'condition_score': float(item['condition_score']),
                    'condition_label': item.get('condition_label', 'unknown'),
                    'degradation_type': ','.join(item.get('degradation_types', [])),
                    'source': 'synthetic_degraded'
                })
        else:
            print(f"Warning: No metadata.json found in {syn_dir}")
    
    if not data_records:
        print(f"No data found. Use --mock to create test dataset.")
        return {}
    
    print(f"Collected {len(data_records)} samples")
    
    df = pd.DataFrame(data_records)
    print(f"Creating stratified splits...")
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, 
        stratify=df['condition_score'].astype(int)
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42,
        stratify=temp_df['condition_score'].astype(int)
    )
    
    print(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    splits = {}
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_path = output_path / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        splits[split_name] = split_path
        print(f"Saved: {split_path}")
    
    stats = {
        'total_samples': len(df),
        'num_conditions': int(df['condition_score'].nunique()),
        'condition_distribution': df['condition_score'].value_counts().sort_index().to_dict(),
        'label_distribution': df['condition_label'].value_counts().to_dict(),
        'source_distribution': df['source'].value_counts().to_dict(),
        'splits': {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df)
        }
    }
    
    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")
    
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Prepare condition assessment training dataset"
    )
    
    parser.add_argument(
        '--deepfashion-dir',
        type=str,
        default='data/deepfashion/original',
        help='DeepFashion original images directory'
    )
    parser.add_argument(
        '--synthetic-dir',
        type=str,
        default='data/deepfashion/synthetic_degraded',
        help='Synthetic degraded images directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/condition_assessment',
        help='Output directory for prepared dataset'
    )
    
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Create mock dataset for testing'
    )
    parser.add_argument(
        '--mock-samples',
        type=int,
        default=100,
        help='Number of samples per condition for mock dataset'
    )
    
    parser.add_argument(
        '--pristine-score',
        type=float,
        default=10.0,
        help='Condition score for pristine images'
    )
    
    args = parser.parse_args()
    print("CONDITION ASSESSMENT DATASET PREPARATION")
    if args.mock:
        splits = create_mock_condition_dataset(
            args.output_dir,
            num_samples=args.mock_samples
        )
    else:
        splits = prepare_deepfashion_condition_dataset(
            args.deepfashion_dir,
            args.synthetic_dir,
            args.output_dir,
            args.pristine_score
        )
    
    if splits:
        print(" DATASET PREPARATION COMPLETE")
        print(f"Dataset saved to: {args.output_dir}")
    else:
        print(f"No dataset prepared. Use --mock for testing.")
        sys.exit(1)


if __name__ == '__main__':
    main()
