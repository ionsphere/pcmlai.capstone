import argparse
from pathlib import Path
from typing import Sequence, Optional, Tuple
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.price import PriceRangeBinner


DEFAULT_BINNING_STRATEGY = 'quantile'
DEFAULT_BIN_COUNT = 5
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_RANDOM_SEED = 42


def make_json_serializable(value):
    if isinstance(value, dict):
        return {str(k): make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [make_json_serializable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_features(features_dir: Path) -> np.ndarray:
    print(f"Loading features from {features_dir}...")
    features = np.load(features_dir / 'features.npy')
    print(f"Loaded features: {features.shape}")
    return features


def analyze_price_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    category_col: str = 'category'
):
    print("Analyzing price distribution...")
    
    prices = df['price'].values
    
    stats = {
        'count': int(len(prices)),
        'mean': float(np.mean(prices)),
        'median': float(np.median(prices)),
        'std': float(np.std(prices)),
        'min': float(np.min(prices)),
        'max': float(np.max(prices)),
        'q25': float(np.percentile(prices, 25)),
        'q75': float(np.percentile(prices, 75))
    }
    
    print(f"Price Statistics:")
    print(f"Count: ${stats['count']}")
    print(f"Mean: ${stats['mean']:.2f}")
    print(f"Median: ${stats['median']:.2f}")
    print(f"Std Dev: ${stats['std']:.2f}")
    print(f"Min: ${stats['min']:.2f}")
    print(f"Max: ${stats['max']:.2f}")
    print(f"Q25: ${stats['q25']:.2f}")
    print(f"Q75: ${stats['q75']:.2f}")
    
    if category_col in df.columns:
        category_stats = df.groupby(category_col)['price'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        print(f"Price by Category:")
        print(category_stats)
        
        category_stats.to_csv(output_dir / 'price_by_category.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].hist(prices, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Price ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Price Distribution')
    axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: ${stats['mean']:.2f}")
    axes[0, 0].axvline(stats['median'], color='green', linestyle='--', label=f"Median: ${stats['median']:.2f}")
    axes[0, 0].legend()
    
    axes[0, 1].hist(prices, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Price ($)')
    axes[0, 1].set_ylabel('Frequency (log scale)')
    axes[0, 1].set_title('Price Distribution (Log Scale)')
    axes[0, 1].set_yscale('log')
    
    axes[1, 0].boxplot(prices, vert=True)
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Price Box Plot')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    if category_col in df.columns:
        categories = df[category_col].unique()[:10]  # Limit to 10 categories for readability
        category_prices = [df[df[category_col] == cat]['price'].values for cat in categories]
        axes[1, 1].boxplot(category_prices, labels=categories, vert=True)
        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('Price ($)')
        axes[1, 1].set_title('Price by Category (Top 10)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'Category data not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'price_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved price distribution plot to {output_dir / 'price_distribution.png'}")
    
    # Save statistics
    with open(output_dir / 'price_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def create_price_bins(
    df: pd.DataFrame,
    output_dir: Path,
    strategy: str = DEFAULT_BINNING_STRATEGY,
    n_bins: int = DEFAULT_BIN_COUNT,
) -> Tuple[PriceRangeBinner, np.ndarray, np.ndarray]:
    print(f"Creating price bins (strategy: {strategy}, n_bins: {n_bins})...")
    
    prices = df['price'].values
    categories = None
    
    binner = PriceRangeBinner(
        strategy=strategy,
        n_bins=n_bins,
        custom_bins=None,
        category_specific=False
    )
    
    bin_indices, bin_labels = binner.fit_transform(prices, categories)
    
    df['price_bin'] = bin_indices
    df['price_bin_label'] = bin_labels
    
    print(f"Price Bins:")
    bin_info = binner.get_bin_info()
    for cat, bins in bin_info['bins'].items():
        labels = bin_info['labels'][cat]
        print(f"Category: {cat}")
        for i, (label, bin_start, bin_end) in enumerate(zip(labels, bins[:-1], bins[1:])):
            count = np.sum((bin_indices == i) & ((categories == cat) if categories is not None else True))
            print(f"Bin {i}: {label} (n={count})")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bin_counts = pd.Series(bin_labels).value_counts().sort_index()
    ax.bar(range(len(bin_counts)), bin_counts.values, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Price Bin')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Distribution of Samples Across Price Bins')
    ax.set_xticks(range(len(bin_counts)))
    ax.set_xticklabels(bin_counts.index, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(bin_counts.values):
        ax.text(i, v + max(bin_counts.values) * 0.01, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bin_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved bin distribution plot to {output_dir / 'bin_distribution.png'}")
    
    serializable_bin_info = make_json_serializable(bin_info)
    with open(output_dir / 'price_binner.json', 'w') as f:
        json.dump(serializable_bin_info, f, indent=2)
    
    print(f"Saved binner configuration to {output_dir / 'price_binner.json'}")
    
    return binner, bin_indices, bin_labels


def create_splits(
    features: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    output_dir: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED
):
    print(f"Creating splits (train={train_ratio}, val={val_ratio}, test={test_ratio})...")
    
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    
    X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
        features, labels, np.arange(len(labels)),
        test_size=test_ratio,
        stratify=labels,
        random_state=random_seed
    )
    
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_trainval, y_trainval, idx_trainval,
        test_size=val_size,
        stratify=y_trainval,
        random_state=random_seed
    )
    
    splits = {
        'train': (X_train, y_train, idx_train),
        'val': (X_val, y_val, idx_val),
        'test': (X_test, y_test, idx_test)
    }
    
    for split_name, (X, y, idx) in splits.items():
        np.save(output_dir / f'{split_name}_features.npy', X)
        np.save(output_dir / f'{split_name}_labels.npy', y)
        np.save(output_dir / f'{split_name}_indices.npy', idx)
        
        split_df = df.iloc[idx].copy()
        split_df.to_csv(output_dir / f'{split_name}_data.csv', index=False)
        
        print(f"{split_name.capitalize()} split: {len(X)} samples")
        
        unique, counts = np.unique(y, return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")
    
    metadata = {
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'random_seed': random_seed,
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'n_test': int(len(X_test)),
        'n_features': int(features.shape[1]),
        'n_classes': int(len(np.unique(labels)))
    }
    
    with open(output_dir / 'split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved split metadata to {output_dir / 'split_metadata.json'}")


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description='Prepare dataset for price classification')
    parser.add_argument('--features-dir', type=str, required=True,
                        help='Directory with extracted features')
    parser.add_argument('--data-csv', type=str, required=True,
                        help='Path to original data CSV')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for prepared dataset')
    parser.add_argument('--strategy', type=str, default=DEFAULT_BINNING_STRATEGY,
                        choices=['quantile', 'uniform'],
                        help='Binning strategy')
    parser.add_argument('--n-bins', type=int, default=DEFAULT_BIN_COUNT,
                        help='Number of price bins')
    
    args = parser.parse_args(argv)
    
    features_dir = Path(args.features_dir)
    data_csv = Path(args.data_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    if not data_csv.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_csv}")
    
    features = load_features(features_dir)
    
    print(f"Loading data from {data_csv}...")
    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} samples")
    
    if 'price' not in df.columns:
        raise ValueError("'price' column not found in data CSV")
    
    analyze_price_distribution(df, output_dir)
    
    binner, bin_indices, bin_labels = create_price_bins(
        df,
        output_dir,
        strategy=args.strategy,
        n_bins=args.n_bins,
    )
    
    create_splits(
        features,
        bin_indices,
        df,
        output_dir,
    )
    
    print("Price Dataset Preparation Summary")
    print(f"Input features: {features.shape}")
    print(f"Price bins: {len(np.unique(bin_indices))}")
    print(f"Binning strategy: {args.strategy}")
    print(f"Train samples: {int(len(features) * DEFAULT_TRAIN_RATIO)}")
    print(f"Val samples: {int(len(features) * DEFAULT_VAL_RATIO)}")
    print(f"Test samples: {int(len(features) * DEFAULT_TEST_RATIO)}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
