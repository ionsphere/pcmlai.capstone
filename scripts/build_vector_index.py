import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence, Optional
import json
import sys
import time


sys.path.append(str(Path(__file__).parent.parent))

from src.vector_search import FAISSIndex, SimilaritySearch


def load_embeddings(embeddings_dir: str):
    embeddings_dir = Path(embeddings_dir)
    embeddings_path = embeddings_dir / 'embeddings.npy'
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    embeddings = np.load(str(embeddings_path))
    print(f"Loaded embeddings: shape={embeddings.shape}")
    
    metadata_path = embeddings_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        item_ids = metadata.get('item_ids', [])
    else:
        item_ids = list(range(len(embeddings)))
    
    items_csv = embeddings_dir / 'items_data.csv'
    if items_csv.exists():
        items_data = pd.read_csv(items_csv)
        print(f"Loaded items data: {len(items_data)} items")
    else:
        items_data = pd.DataFrame({'item_id': item_ids})
        print("Warning: items_data.csv not found, using minimal item IDs")
    
    index_metadata = [{'item_id': item_id} for item_id in item_ids]
    
    return embeddings, index_metadata, items_data


def build_index(
    embeddings: np.ndarray,
    metadata: list,
    index_type: str = 'flat',
    metric: str = 'cosine',
    nlist: int = 100,
    nprobe: int = 10,
    hnsw_m: int = 32,
    use_gpu: bool = False,
):
    print(f"Building {index_type.upper()} index...")
    print(f"Metric: {metric}")
    print(f"Embeddings: {embeddings.shape}")
    
    start_time = time.time()
    
    index = FAISSIndex(
        index_type=index_type,
        dimension=embeddings.shape[1],
        metric=metric,
        nlist=nlist,
        nprobe=nprobe,
        hnsw_m=hnsw_m,
        use_gpu=use_gpu,
    )
    index.build(embeddings, metadata)
    
    build_time = time.time() - start_time
    print(f"Index built in {build_time:.2f} seconds")
    
    stats = index.get_stats()
    print("Index Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return index


def test_search(index: FAISSIndex, embeddings: np.ndarray, items_data: pd.DataFrame):
    print("Testing Search")
    search = SimilaritySearch(index=index, items_data=items_data)
    n_test = min(3, len(embeddings))
    for i in range(n_test):
        print(f"Query {i+1}/{n_test}:")
        print(f"Item ID: {items_data.iloc[i].get('item_id', i)}")
        
        start_time = time.time()
        results = search.search(embeddings[i], k=5)
        search_time = (time.time() - start_time) * 1000
        
        print(f"Search time: {search_time:.2f} ms")
        print(f"Top 5 similar items:")
        
        for j, result in enumerate(results, 1):
            item_id = result.get('item_id', result['index'])
            similarity = result['similarity']
            price = result.get('price', 'N/A')
            category = result.get('category', 'N/A')
            print(f"{j}. ID={item_id}, Similarity={similarity:.4f}, Price={price}, Category={category}")
    
    print("Testing Pricing Context")
    query_idx = 0
    print(f"Query Item ID: {items_data.iloc[query_idx].get('item_id', query_idx)}")
    
    start_time = time.time()
    context = search.get_pricing_context(embeddings[query_idx], k=10)
    context_time = (time.time() - start_time) * 1000
    
    print(f"Pricing context generated in {context_time:.2f} ms")
    print(f"Pricing Context:")
    for key, value in context.items():
        if key != 'similar_items':
            print(f"{key}: {value}")
    
    if 'similar_items' in context:
        print(f"Top 3 similar items:")
        for i, item in enumerate(context['similar_items'][:3], 1):
            print(f"{i}. Price={item['price']}, Similarity={item['similarity']:.4f}")


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Build vector index from embeddings")
    parser.add_argument('--embeddings-dir', type=str, default='data/embeddings',
                        help='Directory containing embeddings and metadata')
    parser.add_argument('--output-dir', type=str, default='data/vector_index',
                        help='Output directory for index')
    parser.add_argument('--output-name', type=str, default='clothing_index',
                        help='Output index name (without extension)')
    parser.add_argument('--index-type', type=str, default='flat',
                        choices=['flat', 'ivf', 'hnsw'],
                        help='Type of FAISS index to build')
    parser.add_argument('--metric', type=str, default='cosine',
                        choices=['cosine', 'l2'],
                        help='Distance metric to use')
    parser.add_argument('--nlist', type=int, default=100,
                        help='Number of clusters for IVF index')
    parser.add_argument('--nprobe', type=int, default=10,
                        help='Number of clusters to search in IVF')
    parser.add_argument('--hnsw-m', type=int, default=32,
                        help='Number of connections for HNSW')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for indexing (requires faiss-gpu)')
    parser.add_argument('--test-search', action='store_true',
                        help='Test search after building index')
    parser.add_argument('--skip-save', action='store_true',
                        help='Skip saving index (for testing)')
    
    args = parser.parse_args(argv)
    
    print("Build Vector Index")
    print(f"Configuration:")
    print(f"Embeddings dir: {args.embeddings_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Index type: {args.index_type}")
    print(f"Metric: {args.metric}")
    
    print("Loading Embeddings")
    
    embeddings, metadata, items_data = load_embeddings(args.embeddings_dir)
    index = build_index(
        embeddings=embeddings,
        metadata=metadata,
        index_type=args.index_type,
        metric=args.metric,
        nlist=args.nlist,
        nprobe=args.nprobe,
        hnsw_m=args.hnsw_m,
        use_gpu=args.use_gpu,
    )
    
    if args.test_search:
        test_search(index, embeddings, items_data)
    
    if not args.skip_save:
        print("Saving Index")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / args.output_name
        index.save(str(output_path))
        items_csv_path = output_dir / 'items_data.csv'
        items_data.to_csv(items_csv_path, index=False)
        print(f"Saved items data to {items_csv_path}")
        print(f"Index saved to {output_path}")
    
    print("Done!")


if __name__ == '__main__':
    main()
