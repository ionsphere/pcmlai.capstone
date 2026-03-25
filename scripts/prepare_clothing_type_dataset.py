import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import shutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DEEPFASHION_TO_STANDARD = {
    "Tee": "t-shirt",
    "Tank": "t-shirt",
    "Shirt": "shirt",
    "Blouse": "shirt",
    "Sweater": "sweater",
    "Cardigan": "sweater",
    "Hoodie": "hoodie",
    "Jacket": "jacket",
    "Blazer": "jacket",
    "Coat": "coat",
    "Cape": "coat",
    
    "Dress": "dress",
    "Romper": "dress",
    "Jumpsuit": "dress",
    
    "Skirt": "skirt",
    "Jeans": "jeans",
    "Pants": "pants",
    "Leggings": "leggings",
    "Shorts": "shorts",
    "Joggers": "pants",
    "Sweatpants": "pants",
    
    "Shoes": "shoes",
    "Sandal": "shoes",
    "Boot": "boots",
    "Sneakers": "sneakers",
    
    "Bag": "bag",
    "Purse": "bag",
    "Backpack": "bag",
    "Hat": "hat",
    "Cap": "hat",
    "Scarf": "scarf",
    "Gloves": "gloves",
    "Belt": "accessories",
    "Sunglasses": "accessories",
    "Jewelry": "accessories",
}

STANDARD_CATEGORIES = [
    "t-shirt", "shirt", "sweater", "hoodie", "jacket", "coat",
    "dress", "skirt", "jeans", "pants", "shorts", "leggings",
    "shoes", "boots", "sneakers",
    "bag", "hat", "scarf", "gloves", "accessories"
]


class ClothingTypeDataset(Dataset):
    def __init__(self, 
                 data: pd.DataFrame,
                 root_dir: Path,
                 transform=None,
                 category_to_idx: Optional[Dict[str, int]] = None):
        self.data = data.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        if category_to_idx is None:
            unique_categories = sorted(self.data['category'].unique())
            self.category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        else:
            self.category_to_idx = category_to_idx
        
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
        self.num_classes = len(self.category_to_idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.root_dir / row['image_path']
        image = Image.open(img_path).convert('RGB')
        category = row['category']
        label = self.category_to_idx[category]
        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)
    
    def get_class_distribution(self) -> Dict[str, int]:
        return dict(self.data['category'].value_counts())
    
    def get_class_weights(self) -> torch.Tensor:
        class_counts = self.data['category'].value_counts()
        total = len(self.data)
        weights = []
        
        for cat in sorted(self.category_to_idx.keys()):
            count = class_counts.get(cat, 0)
            if count > 0:
                weight = total / (len(self.category_to_idx) * count)
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.FloatTensor(weights)


def resolve_deepfashion_root(deepfashion_root: Path) -> Optional[Path]:
    candidates = [
        Path(deepfashion_root),
        Path(deepfashion_root) / "original",
    ]

    for candidate in candidates:
        if (candidate / "Anno").exists() and (candidate / "img").exists():
            return candidate

    return None


def load_category_lookup(category_file: Path) -> Dict[int, str]:
    with open(category_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    start_idx = 0
    for i, line in enumerate(lines):
        if line.lower().startswith('category_name'):
            start_idx = i + 1
            break

    category_lookup = {}
    category_id = 1
    for line in lines[start_idx:]:
        parts = line.split()
        if parts:
            category_lookup[category_id] = parts[0]
            category_id += 1

    return category_lookup


def load_classification_benchmark_annotations(deepfashion_root: Path) -> pd.DataFrame:
    anno_dir = deepfashion_root / "Anno"
    category_lookup = load_category_lookup(anno_dir / "list_category_cloth.txt")
    data = []

    print(f"Loading DeepFashion classification benchmark from: {anno_dir}")

    for split_name in ["train", "val", "test"]:
        image_list_file = anno_dir / f"{split_name}.txt"
        category_list_file = anno_dir / f"{split_name}_cate.txt"

        with open(image_list_file, 'r', encoding='utf-8') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        with open(category_list_file, 'r', encoding='utf-8') as f:
            category_ids = [line.strip() for line in f if line.strip()]

        if len(image_paths) != len(category_ids):
            raise ValueError(
                f"Mismatched DeepFashion benchmark files for {split_name}: "
                f"{len(image_paths)} images vs {len(category_ids)} category ids"
            )

        for image_path, category_id_str in zip(image_paths, category_ids):
            category_name = category_lookup.get(int(category_id_str))
            if category_name is None:
                continue

            data.append({
                'image_path': image_path,
                'category_original': category_name,
                'split': split_name,
            })

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} benchmark annotations")
    return df


def load_deepfashion_annotations(deepfashion_root: Path) -> Optional[pd.DataFrame]:
    deepfashion_root = Path(deepfashion_root)
    resolved_root = resolve_deepfashion_root(deepfashion_root)

    if resolved_root is None:
        possible_anno_files = [
            deepfashion_root / "Anno" / "list_category_cloth.txt",
            deepfashion_root / "Anno" / "list_category_img.txt",
            deepfashion_root / "original" / "Anno" / "list_category_cloth.txt",
            deepfashion_root / "original" / "Anno" / "list_category_img.txt",
            deepfashion_root / "list_category_cloth.txt",
            deepfashion_root / "list_category_img.txt",
        ]

        print("Could not find DeepFashion annotation files")
        print("Expected locations:")
        for f in possible_anno_files:
            print(f"  - {f}")
        return None

    anno_dir = resolved_root / "Anno"
    benchmark_files = [
        anno_dir / "train.txt",
        anno_dir / "train_cate.txt",
        anno_dir / "val.txt",
        anno_dir / "val_cate.txt",
        anno_dir / "test.txt",
        anno_dir / "test_cate.txt",
        anno_dir / "list_category_cloth.txt",
    ]

    if all(path.exists() for path in benchmark_files):
        return load_classification_benchmark_annotations(resolved_root)

    possible_anno_files = [
        anno_dir / "list_category_img.txt",
        resolved_root / "list_category_img.txt",
    ]

    anno_file = next((f for f in possible_anno_files if f.exists()), None)

    if anno_file is None:
        print("Found DeepFashion root, but no supported image-level annotation files")
        print(f"Resolved dataset root: {resolved_root}")
        print("Supported formats:")
        print("  - Classification benchmark: train.txt + train_cate.txt + list_category_cloth.txt")
        print("  - Image-level list: list_category_img.txt")
        return None

    print(f"Loading annotations from: {anno_file}")

    data = []
    with open(anno_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith('#') and not line.lower().startswith('category'):
            start_idx = i
            break

    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            image_path = parts[0]
            category = ' '.join(parts[1:])
            data.append({
                'image_path': image_path,
                'category_original': category
            })

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} annotations")

    return df


def map_categories_to_standard(df: pd.DataFrame) -> pd.DataFrame:
    def map_category(orig_cat: str) -> Optional[str]:
        if orig_cat in DEEPFASHION_TO_STANDARD:
            return DEEPFASHION_TO_STANDARD[orig_cat]
        
        orig_lower = orig_cat.lower()
        for key, value in DEEPFASHION_TO_STANDARD.items():
            if key.lower() in orig_lower or orig_lower in key.lower():
                return value
        
        return None
    
    df['category'] = df['category_original'].apply(map_category)
    before = len(df)
    df = df.dropna(subset=['category'])
    after = len(df)
    if before > after:
        print(f"Removed {before - after} items with unmapped categories")
    
    return df


def create_mock_dataset(output_dir: Path, num_samples: int = 1000) -> pd.DataFrame:
    print(f"Creating mock dataset with {num_samples} samples...")
    output_dir = Path(output_dir)
    mock_images_dir = output_dir / "mock_images"
    mock_images_dir.mkdir(parents=True, exist_ok=True)
    data = []
    for i in tqdm(range(num_samples), desc="Creating mock images"):
        category = np.random.choice(STANDARD_CATEGORIES)
        img = Image.new('RGB', (224, 224), 
                       color=(np.random.randint(0, 256),
                              np.random.randint(0, 256),
                              np.random.randint(0, 256)))
        
        img_filename = f"mock_{category}_{i:05d}.jpg"
        img_path = mock_images_dir / img_filename
        img.save(img_path)
        
        data.append({
            'image_path': str(img_path.relative_to(output_dir)),
            'category': category,
            'category_original': category
        })
    
    df = pd.DataFrame(data)
    print(f"Created {len(df)} mock images")
    
    return df


def create_splits(df: pd.DataFrame, 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['category'],
        random_state=random_state
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df['category'],
        random_state=random_state
    )
    
    print(f"Dataset splits:")
    print(f"Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def generate_statistics(df: pd.DataFrame, split_name: str = "dataset") -> Dict:
    category_distribution = {
        str(cat): int(count) for cat, count in df['category'].value_counts().items()
    }
    category_percentages = {
        str(cat): float(pct) for cat, pct in (df['category'].value_counts(normalize=True) * 100).items()
    }

    stats = {
        'split_name': split_name,
        'total_samples': int(len(df)),
        'num_categories': int(df['category'].nunique()),
        'category_distribution': category_distribution,
        'category_percentages': category_percentages,
    }
    
    return stats


def save_dataset_info(output_dir: Path,
                     train_df: pd.DataFrame,
                     val_df: pd.DataFrame,
                     test_df: pd.DataFrame,
                     category_to_idx: Dict[str, int],
                     image_root: Optional[Path] = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Saved CSV files to {output_dir}")
    
    mapping_file = output_dir / "category_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump({
            'category_to_idx': category_to_idx,
            'idx_to_category': {v: k for k, v in category_to_idx.items()},
            'num_classes': len(category_to_idx),
            'categories': sorted(category_to_idx.keys()),
            'image_root': str(image_root) if image_root else None,
        }, f, indent=2)
    print(f"Saved category mapping to {mapping_file}")
    
    all_stats = {
        'train': generate_statistics(train_df, 'train'),
        'val': generate_statistics(val_df, 'val'),
        'test': generate_statistics(test_df, 'test'),
        'total': generate_statistics(
            pd.concat([train_df, val_df, test_df]), 'total'
        )
    }
    
    stats_file = output_dir / "dataset_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"Saved statistics to {stats_file}")
    
    print("DATASET SUMMARY")
    for split_name, stats in all_stats.items():
        if split_name != 'total':
            print(f"{split_name.upper()}:")
            print(f"Total samples: {stats['total_samples']:,}")
            print(f"Categories: {stats['num_categories']}")
            print(f"Top 5 categories:")
            for cat, count in sorted(stats['category_distribution'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                pct = stats['category_percentages'][cat]
                print(f" - {cat}: {count:,} ({pct:.1f}%)")


def verify_dataset(output_dir: Path):
    output_dir = Path(output_dir)
    print("Verifying dataset...")
    required_files = [
        'train.csv',
        'val.csv', 
        'test.csv',
        'category_mapping.json',
        'dataset_statistics.json'
    ]
    
    for fname in required_files:
        fpath = output_dir / fname
        if not fpath.exists():
            print(f"Missing: {fpath}")
            return False
        else:
            print(f"Found: {fname}")
    
    train_df = pd.read_csv(output_dir / 'train.csv')
    val_df = pd.read_csv(output_dir / 'val.csv')
    test_df = pd.read_csv(output_dir / 'test.csv')
    
    print(f"Train set: {len(train_df):,} samples")
    print(f"Val set: {len(val_df):,} samples")
    print(f"Test set: {len(test_df):,} samples")
    
    train_cats = set(train_df['category'].unique())
    val_cats = set(val_df['category'].unique())
    test_cats = set(test_df['category'].unique())
    
    all_cats = train_cats | val_cats | test_cats
    print(f"Total categories: {len(all_cats)}")
    
    missing_in_val = train_cats - val_cats
    missing_in_test = train_cats - test_cats
    
    if missing_in_val:
        print(f"Categories in train but not in val: {missing_in_val}")
    if missing_in_test:
        print(f"Categories in train but not in test: {missing_in_test}")
    
    print("Dataset verification complete!")
    return True


def filter_missing_images(df: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    image_root = Path(image_root)
    exists_mask = df['image_path'].apply(lambda rel_path: (image_root / rel_path).exists())
    missing_count = int((~exists_mask).sum())

    if missing_count:
        print(f"Removed {missing_count} annotations whose images were missing under {image_root}")

    return df.loc[exists_mask].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare clothing type classification dataset"
    )
    parser.add_argument(
        '--deepfashion-root',
        type=str,
        default='data/deepfashion',
        help='Root directory of DeepFashion dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/clothing_type',
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Create mock dataset for testing (when DeepFashion not available)'
    )
    parser.add_argument(
        '--mock-samples',
        type=int,
        default=1000,
        help='Number of mock samples to create'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify existing dataset only'
    )
    
    args = parser.parse_args()
    
    deepfashion_root = Path(args.deepfashion_root)
    output_dir = Path(args.output_dir)
    
    if args.verify:
        verify_dataset(output_dir)
        return
    
    print("CLOTHING TYPE DATASET PREPARATION")
    resolved_root = None

    if args.mock or not deepfashion_root.exists():
        if not args.mock:
            print(f"DeepFashion root not found: {deepfashion_root}")
            print("Creating mock dataset instead...")
        df = create_mock_dataset(output_dir, args.mock_samples)
        image_root = output_dir
    else:
        df = load_deepfashion_annotations(deepfashion_root)
        
        if df is None or len(df) == 0:
            print(" Failed to load DeepFashion dataset")
            print("Use --mock flag to create a mock dataset for testing")
            return
        
        df = map_categories_to_standard(df)
        resolved_root = resolve_deepfashion_root(deepfashion_root)
        if resolved_root is not None:
            df = filter_missing_images(df, resolved_root)
        image_root = resolved_root
    
    if len(df) == 0:
        print("No data available after processing")
        return
    
    print(f"Total samples: {len(df):,}")
    print(f"Unique categories: {df['category'].nunique()}")
    
    categories = sorted(df['category'].unique())
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    if 'split' in df.columns and {'train', 'val', 'test'}.issubset(set(df['split'].unique())):
        train_df = df[df['split'] == 'train'].drop(columns=['split']).copy()
        val_df = df[df['split'] == 'val'].drop(columns=['split']).copy()
        test_df = df[df['split'] == 'test'].drop(columns=['split']).copy()
        print("Using official DeepFashion train/val/test benchmark splits")
        print(f"Train: {len(train_df):,} samples")
        print(f"Val: {len(val_df):,} samples")
        print(f"Test: {len(test_df):,} samples")
    else:
        train_df, val_df, test_df = create_splits(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_seed
        )
    
    save_dataset_info(output_dir, train_df, val_df, test_df, category_to_idx, image_root=image_root)
    verify_dataset(output_dir)
    
    print(" DATASET PREPARATION COMPLETE!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
