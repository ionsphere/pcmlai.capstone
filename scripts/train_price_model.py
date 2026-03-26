import argparse
import sys
from pathlib import Path
from typing import Sequence, Optional
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.price import PriceClassifier


def load_prepared_data(features_dir: Path) -> Dict[str, Any]:
    print(f"Loading data from {features_dir}...")
    
    X_train = np.load(features_dir / 'train_features.npy')
    y_train = np.load(features_dir / 'train_labels.npy')
    X_val = np.load(features_dir / 'val_features.npy')
    y_val = np.load(features_dir / 'val_labels.npy')
    X_test = np.load(features_dir / 'test_features.npy')
    y_test = np.load(features_dir / 'test_labels.npy')
    
    with open(features_dir / 'price_binner.json', 'r') as f:
        binner_info = json.load(f)
    
    with open(features_dir / 'split_metadata.json', 'r') as f:
        split_info = json.load(f)
    
    feature_names = None
    feature_names_path = features_dir.parent / 'features' / 'feature_names.json'
    if feature_names_path.exists():
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
    
    bins_dict = binner_info.get('bins', {})
    labels_dict = binner_info.get('labels', {})
    
    if 'global' in labels_dict:
        class_names = labels_dict['global']
    else:
        first_cat = list(labels_dict.keys())[0]
        class_names = labels_dict[first_cat]
    
    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Val: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Classes: {len(class_names)} price ranges")
    print(f"Class names: {class_names}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'class_names': class_names,
        'feature_names': feature_names,
        'n_classes': len(class_names),
        'binner_info': binner_info,
        'split_info': split_info
    }


def generate_mock_data(n_samples: int = 1000, n_features: int = 100, n_classes: int = 5):
    print(f"Generating mock data...")
    print(f"Samples: {n_samples}")
    print(f"Features: {n_features}")
    print(f"Classes: {n_classes}")
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    label_signal = X[:, :10].mean(axis=1)
    y = pd.qcut(label_signal, q=n_classes, labels=False)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]
    
    class_names = [f"$0-${(i+1)*20}" for i in range(n_classes)]
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Val: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'class_names': class_names,
        'feature_names': [f'feature_{i}' for i in range(n_features)],
        'n_classes': n_classes,
        'binner_info': {'n_bins': n_classes, 'strategy': 'mock'},
        'split_info': {'train_size': n_train, 'val_size': n_val, 'test_size': len(y_test)}
    }


def train_model(
    model_type: str,
    data: Dict[str, Any],
    model_params: Dict[str, Any] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    if model_params is None:
        model_params = {}
    
    print(f"Training {model_type.upper()} model...")
    
    model = PriceClassifier(
        model_type=model_type,
        n_classes=data['n_classes'],
        **model_params
    )
    
    start_time = time.time()
    model.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        verbose=verbose
    )
    train_time = time.time() - start_time
    
    print(f"Training completed in {train_time:.2f}s")
    
    print(f"Evaluating on validation set...")
    val_metrics = model.evaluate(
        data['X_val'],
        data['y_val'],
        class_names=data['class_names']
    )
    
    print(f"Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"F1-score (weighted): {val_metrics['f1_weighted']:.4f}")
    print(f"Within +/-1 bracket: {val_metrics['within_1_accuracy']:.4f}")
    print(f"Evaluating on test set...")
    test_metrics = model.evaluate(
        data['X_test'],
        data['y_test'],
        class_names=data['class_names']
    )
    
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1-score (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"Within +/-1 bracket: {test_metrics['within_1_accuracy']:.4f}")
    
    return {
        'model': model,
        'model_type': model_type,
        'train_time': train_time,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }


def compare_models(data: Dict[str, Any], output_dir: Path):
    print("COMPARING MODELS")
    
    models_config = {
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
    }
    
    results = []
    
    for model_type, params in models_config.items():
        try:
            result = train_model(model_type, data, params, verbose=False)
            results.append(result)
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            continue
    
    print("MODEL COMPARISON RESULTS")
    
    comparison_data = []
    for result in results:
        comparison_data.append({
            'Model': result['model_type'].upper(),
            'Train Time (s)': f"{result['train_time']:.2f}",
            'Val Accuracy': f"{result['val_metrics']['accuracy']:.4f}",
            'Val F1': f"{result['val_metrics']['f1_weighted']:.4f}",
            'Val +/-1': f"{result['val_metrics']['within_1_accuracy']:.4f}",
            'Test Accuracy': f"{result['test_metrics']['accuracy']:.4f}",
            'Test F1': f"{result['test_metrics']['f1_weighted']:.4f}",
            'Test +/-1': f"{result['test_metrics']['within_1_accuracy']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"Saved comparison to {output_dir / 'model_comparison.csv'}")
    
    best_result = max(results, key=lambda x: x['test_metrics']['accuracy'])
    print(f"Best Model: {best_result['model_type'].upper()}")
    print(f"Test Accuracy: {best_result['test_metrics']['accuracy']:.4f}")
    print(f"Test F1: {best_result['test_metrics']['f1_weighted']:.4f}")
    print(f"Within +/-1 bracket: {best_result['test_metrics']['within_1_accuracy']:.4f}")
    
    return results, best_result


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_feature_importance(model: PriceClassifier, feature_names: List[str], output_path: Path, top_k: int = 20):
    try:
        importance_df = model.get_feature_importance(feature_names, top_k=top_k)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_k} Feature Importances')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved feature importance to {output_path}")
        
        csv_path = output_path.with_suffix('.csv')
        importance_df.to_csv(csv_path, index=False)
        print(f"Saved feature importance CSV to {csv_path}")
        
    except Exception as e:
        print(f"Could not generate feature importance: {e}")


def save_results(result: Dict[str, Any], data: Dict[str, Any], output_dir: Path):
    print(f"Saving results to {output_dir}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = result['model']
    model_type = result['model_type']
    model_path = output_dir / f'{model_type}_model.pkl'
    model.save(model_path)
    print(f"Saved model to {model_path}")
    
    metrics = {
        'model_type': model_type,
        'train_time': result['train_time'],
        'timestamp': datetime.now().isoformat(),
        'validation_metrics': {
            'accuracy': float(result['val_metrics']['accuracy']),
            'f1_weighted': float(result['val_metrics']['f1_weighted']),
            'f1_macro': float(result['val_metrics']['f1_macro']),
            'within_1_accuracy': float(result['val_metrics']['within_1_accuracy'])
        },
        'test_metrics': {
            'accuracy': float(result['test_metrics']['accuracy']),
            'f1_weighted': float(result['test_metrics']['f1_weighted']),
            'f1_macro': float(result['test_metrics']['f1_macro']),
            'within_1_accuracy': float(result['test_metrics']['within_1_accuracy'])
        },
        'data_info': {
            'n_train': int(data['X_train'].shape[0]),
            'n_val': int(data['X_val'].shape[0]),
            'n_test': int(data['X_test'].shape[0]),
            'n_features': int(data['X_train'].shape[1]),
            'n_classes': int(data['n_classes']),
            'class_names': data['class_names']
        }
    }
    
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("VALIDATION SET\n")
        f.write("=" * 80 + "\n")
        f.write(classification_report(
            data['y_val'],
            result['val_metrics']['predictions'],
            target_names=data['class_names'],
            zero_division=0
        ))
        f.write("\n\nTEST SET\n")
        f.write("=" * 80 + "\n")
        f.write(classification_report(
            data['y_test'],
            result['test_metrics']['predictions'],
            target_names=data['class_names'],
            zero_division=0
        ))
    print(f"Saved classification report to {report_path}")
    
    cm = np.array(result['test_metrics']['confusion_matrix'])
    plot_confusion_matrix(cm, data['class_names'], output_dir / 'confusion_matrix.png')
    
    if data['feature_names']:
        plot_feature_importance(
            model,
            data['feature_names'],
            output_dir / 'feature_importance.png',
            top_k=20
        )


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description='Train Price Classification Model')
    parser.add_argument('--features-dir', type=str, default='data/price_classification',
                        help='Directory with prepared features')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with mock data')
    parser.add_argument('--mock-samples', type=int, default=1000,
                        help='Number of mock samples for quick test')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'random_forest'],
                        help='Model type to train')
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare all model types')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of estimators (trees)')
    parser.add_argument('--max-depth', type=int, default=8,
                        help='Maximum tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization (slow)')
    parser.add_argument('--output-dir', type=str, default='models/price_model',
                        help='Output directory for model and results')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose training output')
    
    args = parser.parse_args(argv)
    
    print("PRICE CLASSIFICATION MODEL TRAINING")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.quick_test:
        print("Quick Test Mode")
        data = generate_mock_data(n_samples=args.mock_samples)
    else:
        features_dir = Path(args.features_dir)
        if not features_dir.exists():
            print(f"Error: Features directory not found: {features_dir}")
            return 1
        
        data = load_prepared_data(features_dir)
    
    output_dir = Path(args.output_dir)
    
    if args.compare_models:
        results, best_result = compare_models(data, output_dir)
        save_results(best_result, data, output_dir / 'best_model')
    else:
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'random_state': 42
        }
        
        if args.model == 'lightgbm':
            model_params['verbose'] = -1
        elif args.model == 'random_forest':
            model_params['n_jobs'] = -1
            del model_params['learning_rate']
        
        result = train_model(args.model, data, model_params, verbose=args.verbose)
        save_results(result, data, output_dir)
    
    print(" TRAINING COMPLETED")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
