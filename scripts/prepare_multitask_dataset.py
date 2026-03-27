#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Sequence, Optional, Dict

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DEEPFASHION2_CSV = 'data/processed/deepfashion2_processed.csv'
DEFAULT_SYNTHETIC_ROOT = 'data/deepfashion/synthetic_degraded'
DEFAULT_OUTPUT_DIR = 'data/processed/multitask'
DEFAULT_RANDOM_SEED = 42
DEMO_MAX_PRISTINE_ROWS = 300
DEMO_MAX_SYNTHETIC_ROWS = 300
DEEPFASHION2_TO_STANDARD = {
    "short sleeve top": "t-shirt",
    "long sleeve top": "shirt",
    "short sleeve outwear": "jacket",
    "long sleeve outwear": "coat",
    "vest": "shirt",
    "sling": "shirt",
    "shorts": "shorts",
    "trousers": "pants",
    "skirt": "skirt",
    "short sleeve dress": "dress",
    "long sleeve dress": "dress",
    "vest dress": "dress",
    "sling dress": "dress",
}


def normalize_deepfashion2_path(path_value: str) -> str:
    path_str = str(path_value).replace("/", "\\").strip()
    kaggle_prefix = "\\kaggle\\input\\deepfashion2-original-with-dataframes\\DeepFashion2\\"
    if path_str.lower().startswith(kaggle_prefix.lower()):
        suffix = path_str[len(kaggle_prefix):].lstrip("\\")
        return str(Path("data") / "deepfashion" / "original" / "kaggle_thushan" / "DeepFashion2" / suffix)
    return path_str


def bbox_area(bbox_value: str) -> float:
    if not isinstance(bbox_value, str):
        return 0.0
    cleaned = bbox_value.strip().strip("[]")
    parts = [p.strip() for p in cleaned.split(",")]
    if len(parts) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(p) for p in parts]
    except ValueError:
        return 0.0
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def build_pristine_records(deepfashion2_csv: Path, demo: bool = False) -> pd.DataFrame:
    df = pd.read_csv(deepfashion2_csv)
    df = df[['path', 'category_name', 'split', 'b_box']].copy()
    df['image_path'] = df['path'].apply(normalize_deepfashion2_path)
    df['category'] = df['category_name'].str.lower().map(DEEPFASHION2_TO_STANDARD)
    df = df.dropna(subset=['category'])
    df['bbox_area'] = df['b_box'].apply(bbox_area)
    df = (
        df.sort_values(['image_path', 'bbox_area'], ascending=[True, False])
        .drop_duplicates(subset=['image_path'], keep='first')
        .copy()
    )
    df['condition_score'] = 10.0
    df['source'] = 'deepfashion2_pristine'
    df = df[['image_path', 'category', 'condition_score', 'split', 'source']]
    if demo:
        df = df.head(DEMO_MAX_PRISTINE_ROWS).copy()
    return df


def load_synthetic_metadata(synthetic_root: Path, demo: bool = False) -> pd.DataFrame:
    metadata_files = list(synthetic_root.rglob('generation_metadata.json'))
    records = []
    for metadata_file in metadata_files:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        for item in payload.get('statistics', {}).get('metadata', []):
            records.append({
                'source_image': str(Path(str(item['source_image']))),
                'image_path': str(Path(str(item['output_path']))),
                'condition_score': float(item['condition_level']),
                'source': 'synthetic_degraded',
            })

    synthetic_df = pd.DataFrame(records)
    if demo:
        synthetic_df = synthetic_df.head(DEMO_MAX_SYNTHETIC_ROWS).copy()
    return synthetic_df


def build_multitask_dataset(deepfashion2_csv: Path, synthetic_root: Path, demo: bool = False) -> pd.DataFrame:
    pristine_df = build_pristine_records(deepfashion2_csv, demo=demo)
    source_lookup = pristine_df[['image_path', 'category', 'split']].copy()
    source_lookup = source_lookup.rename(columns={'image_path': 'source_image'})
    synthetic_df = load_synthetic_metadata(synthetic_root, demo=demo)
    if synthetic_df.empty:
        return pristine_df

    synthetic_df['source_image'] = synthetic_df['source_image'].apply(lambda p: str(Path(p)))
    merged_synthetic = synthetic_df.merge(source_lookup, on='source_image', how='inner')
    merged_synthetic = merged_synthetic[['image_path', 'category', 'condition_score', 'split', 'source']]
    multitask_df = pd.concat([pristine_df, merged_synthetic], ignore_index=True)
    multitask_df = multitask_df[multitask_df['image_path'].apply(lambda p: (PROJECT_ROOT / Path(str(p))).exists())].copy()
    return multitask_df


def save_splits(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = df['split'].value_counts().to_dict() if 'split' in df.columns else {}
    has_source_splits = all(split_counts.get(split, 0) > 0 for split in ['train', 'val', 'test'])
    if not has_source_splits:
        stratify_col = df['category'] if df['category'].value_counts().min() >= 2 else None
        train_df, temp_df = train_test_split(
            df.drop(columns=['split'], errors='ignore'),
            test_size=0.30,
            random_state=DEFAULT_RANDOM_SEED,
            stratify=stratify_col,
        )
        temp_stratify = temp_df['category'] if stratify_col is not None else None
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=DEFAULT_RANDOM_SEED,
            stratify=temp_stratify,
        )
        split_frames = {'train': train_df, 'val': val_df, 'test': test_df}
    else:
        split_frames = {
            split: df[df['split'] == split].drop(columns=['split']).copy()
            for split in ['train', 'val', 'test']
        }

    split_map = {
        'train': output_dir / 'train.csv',
        'val': output_dir / 'val.csv',
        'test': output_dir / 'test.csv',
    }

    for split_name, output_path in split_map.items():
        split_df = split_frames[split_name]
        split_df.to_csv(output_path, index=False)

    stats = {
        'total_samples': int(len(df)),
        'split_counts': {split: int(len(split_frames[split])) for split in ['train', 'val', 'test']},
        'category_counts': {k: int(v) for k, v in df['category'].value_counts().to_dict().items()},
        'condition_min': float(df['condition_score'].min()),
        'condition_max': float(df['condition_score'].max()),
        'used_source_splits': has_source_splits,
    }
    with open(output_dir / 'dataset_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Prepare multitask training dataset")
    parser.add_argument(
        '--deepfashion2-csv',
        default=DEFAULT_DEEPFASHION2_CSV,
        help='Path to processed DeepFashion2 CSV'
    )
    parser.add_argument(
        '--synthetic-root',
        default=DEFAULT_SYNTHETIC_ROOT,
        help='Root directory containing synthetic degradation outputs'
    )
    parser.add_argument(
        '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for multitask CSV splits'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run a reduced demo dataset preparation pass'
    )
    args = parser.parse_args(argv)

    deepfashion2_csv = PROJECT_ROOT / args.deepfashion2_csv
    synthetic_root = PROJECT_ROOT / args.synthetic_root
    output_dir = PROJECT_ROOT / args.output_dir

    if not deepfashion2_csv.exists():
        raise FileNotFoundError(f"DeepFashion2 processed CSV not found: {deepfashion2_csv}")

    if args.demo:
        print(
            f"Demo mode enabled: using up to {DEMO_MAX_PRISTINE_ROWS} pristine and "
            f"{DEMO_MAX_SYNTHETIC_ROWS} synthetic rows"
        )

    df = build_multitask_dataset(deepfashion2_csv, synthetic_root, demo=args.demo)
    if df.empty:
        raise RuntimeError("Multitask dataset generation produced no rows.")

    save_splits(df, output_dir)
    print(f"Saved multitask dataset to {output_dir}")
    print(f"Rows: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")


if __name__ == '__main__':
    main()
