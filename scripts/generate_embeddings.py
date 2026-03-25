import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional
import json
from datetime import datetime
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.embeddings import (
    VisionEmbeddingExtractor,
    TextEmbeddingExtractor,
    MultiModalEmbedding,
    EmbeddingPipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate embeddings for similarity search')
    parser.add_argument('--data-csv', type=str, default=None,
                      help='Path to CSV file with image paths and text data')
    parser.add_argument('--image-col', type=str, default='image_path',
                      help='Column name for image paths')
    parser.add_argument('--text-col', type=str, default='description',
                      help='Column name for text data (can specify multiple with comma)')
    parser.add_argument('--title-col', type=str, default='title',
                      help='Column name for title (optional, will be combined with description)')
    parser.add_argument('--vision-model', type=str, default=None,
                      help='Path to trained vision model checkpoint')
    parser.add_argument('--text-model', type=str, default='all-MiniLM-L6-v2',
                      help='Sentence Transformer model name')
    parser.add_argument('--mode', type=str, default='multimodal',
                      choices=['vision', 'text', 'multimodal'],
                      help='Embedding mode')
    parser.add_argument('--fusion-method', type=str, default='concat',
                      choices=['concat', 'weighted'],
                      help='Method to fuse vision and text embeddings')
    parser.add_argument('--reduce-dim', action='store_true',
                      help='Apply dimensionality reduction')
    parser.add_argument('--reduction-method', type=str, default='pca',
                      choices=['pca', 'umap'],
                      help='Dimensionality reduction method')
    parser.add_argument('--target-dim', type=int, default=128,
                      help='Target dimension after reduction')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for processing')
    parser.add_argument('--normalize', action='store_true', default=True,
                      help='Normalize embeddings (L2 norm)')
    parser.add_argument('--max-samples', type=int, default=None,
                      help='Maximum number of samples to process (for testing)')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save embeddings')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--mock', action='store_true',
                      help='Generate mock data for testing')
    parser.add_argument('--mock-samples', type=int, default=100,
                      help='Number of mock samples to generate')
    
    return parser.parse_args()


def generate_mock_data(num_samples: int, output_dir: str) -> str:
    from PIL import Image
    import os
    
    print(f"Generating {num_samples} mock samples...")
    images_dir = Path(output_dir) / 'mock_images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    categories = ['shirt', 'jeans', 'dress', 'jacket', 'shoes']
    conditions = ['excellent', 'good', 'fair', 'worn']
    for i in range(num_samples):
        color = tuple(np.random.randint(0, 256, 3).tolist())
        img = Image.new('RGB', (380, 380), color)
        
        img_path = images_dir / f'item_{i:05d}.jpg'
        img.save(img_path)
        
        category = np.random.choice(categories)
        condition = np.random.choice(conditions)
        price = np.random.uniform(10, 200)
        
        data.append({
            'item_id': f'MOCK_{i:05d}',
            'image_path': str(img_path),
            'title': f"Mock {category.title()} Item {i}",
            'description': f"A {condition} condition {category} in great shape. Perfect for daily wear.",
            'category': category,
            'condition': condition,
            'price': round(price, 2),
        })
    
    df = pd.DataFrame(data)
    csv_path = Path(output_dir) / 'mock_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Mock data saved to {csv_path}")
    
    return str(csv_path)


def load_data(args) -> pd.DataFrame:
    if args.mock:
        csv_path = generate_mock_data(args.mock_samples, args.output_dir)
        df = pd.read_csv(csv_path)
    else:
        print(f"Loading data from {args.data_csv}")
        df = pd.read_csv(args.data_csv)
    
    print(f"Loaded {len(df)} samples")
    
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Limited to {len(df)} samples")
    
    return df


def prepare_data(df: pd.DataFrame, args) -> tuple:
    images = None
    texts = None
    
    if args.mode in ['vision', 'multimodal']:
        if args.image_col not in df.columns:
            print(f"Warning: Image column '{args.image_col}' not found in CSV")
            if args.mode == 'vision':
                raise ValueError("Vision mode requires image column")
        else:
            images = df[args.image_col].tolist()
            print(f"Prepared {len(images)} images")
    
    if args.mode in ['text', 'multimodal']:
        text_parts = []
        
        if args.title_col in df.columns:
            titles = df[args.title_col].fillna('').astype(str)
            text_parts.append(titles)
        
        if args.text_col in df.columns:
            descriptions = df[args.text_col].fillna('').astype(str)
            text_parts.append(descriptions)
        else:
            print(f"Warning: Text column '{args.text_col}' not found in CSV")
            if args.mode == 'text':
                raise ValueError("Text mode requires text column")
        
        if text_parts:
            texts = [' '.join(parts) for parts in zip(*text_parts)]
            print(f"Prepared {len(texts)} texts")
    
    return images, texts


def main():
    args = parse_args()
    
    if not args.mock and args.data_csv is None:
        print("Error: Either --mock or --data-csv must be provided")
        sys.exit(1)
    
    print("EMBEDDING GENERATION")
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {args.device}")
    
    df = load_data(args)
    images, texts = prepare_data(df, args)
    if args.mode == 'vision' and args.vision_model is None:
        raise ValueError("Vision mode requires --vision-model argument")
    if args.mode == 'text' and args.text_model is None:
        raise ValueError("Text mode requires --text-model argument")
    
    print("INITIALIZING EMBEDDING PIPELINE")
    pipeline = EmbeddingPipeline(
        vision_model_path=args.vision_model if args.mode in ['vision', 'multimodal'] else None,
        text_model_name=args.text_model if args.mode in ['text', 'multimodal'] else None,
        fusion_method=args.fusion_method,
        reduce_dim=args.reduce_dim,
        reduction_method=args.reduction_method,
        target_dim=args.target_dim,
        device=args.device,
    )
    
    print(f"Embedding dimension: {pipeline.multi_modal.embedding_dim}")
    if args.reduce_dim:
        print(f"Will be reduced to: {args.target_dim}")
    
    print("GENERATING EMBEDDINGS")
    
    embeddings = pipeline.generate(
        images=images,
        texts=texts,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    metadata = {
        'mode': args.mode,
        'num_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'vision_model': args.vision_model,
        'text_model': args.text_model,
        'fusion_method': args.fusion_method,
        'reduced': args.reduce_dim,
        'reduction_method': args.reduction_method if args.reduce_dim else None,
        'target_dim': args.target_dim if args.reduce_dim else None,
        'normalized': args.normalize,
        'generated_at': datetime.now().isoformat(),
        'data_source': args.data_csv,
        'item_ids': df.get('item_id', df.index).tolist(),
    }
    
    print("SAVING EMBEDDINGS")
    
    pipeline.save(args.output_dir, embeddings, metadata)
    
    df_path = Path(args.output_dir) / 'items_data.csv'
    df.to_csv(df_path, index=False)
    print(f"Item data saved to {df_path}")
    
    print("EMBEDDING GENERATION COMPLETE")
    print(f"Embeddings saved to: {args.output_dir}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print("Embedding Statistics:")
    print(f"Mean: {np.mean(embeddings):.6f}")
    print(f"Std: {np.std(embeddings):.6f}")
    print(f"Min: {np.min(embeddings):.6f}")
    print(f"Max: {np.max(embeddings):.6f}")
    
    if args.normalize:
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"L2 Norms (should be ~1.0 if normalized):")
        print(f"Mean: {np.mean(norms):.6f}")
        print(f"Min: {np.min(norms):.6f}")
        print(f"Max: {np.max(norms):.6f}")


if __name__ == '__main__':
    main()
