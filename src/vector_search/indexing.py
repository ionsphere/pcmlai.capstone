"""
Vector Index Module

Provides vector indexing capabilities using FAISS and other backends.
Supports efficient similarity search for large-scale clothing item databases.
"""

import numpy as np
import faiss
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import pickle
from abc import ABC, abstractmethod


class VectorIndex(ABC):
    """Abstract base class for vector indices."""
    
    @abstractmethod
    def build(self, embeddings: np.ndarray, metadata: Optional[List[Dict]] = None):
        """Build index from embeddings."""
        pass
    
    @abstractmethod
    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load index from disk."""
        pass


class FAISSIndex(VectorIndex):
    """
    FAISS-based vector index for fast similarity search.
    
    Supports multiple index types:
    - Flat (exact search, best quality, slower for large datasets)
    - IVF (inverted file index, good balance)
    - HNSW (hierarchical navigable small world, fast approximate search)
    """
    
    def __init__(
        self,
        index_type: str = 'flat',
        dimension: Optional[int] = None,
        metric: str = 'cosine',  # 'cosine' or 'l2'
        nlist: int = 100,  # Number of clusters for IVF
        nprobe: int = 10,  # Number of clusters to visit for IVF search
        hnsw_m: int = 32,  # Number of connections for HNSW
        use_gpu: bool = False,
    ):
        """
        Initialize FAISS index.
        
        Args:
            index_type: Type of index ('flat', 'ivf', 'hnsw')
            dimension: Embedding dimension (required for building index)
            metric: Distance metric ('cosine' or 'l2')
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search in IVF
            hnsw_m: Number of connections per layer for HNSW
            use_gpu: Whether to use GPU for indexing (requires faiss-gpu)
        """
        self.index_type = index_type.lower()
        self.dimension = dimension
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        self.use_gpu = use_gpu
        
        self.index = None
        self.metadata = None
        self.n_items = 0
        
        # Validate index type
        valid_types = ['flat', 'ivf', 'hnsw']
        if self.index_type not in valid_types:
            raise ValueError(f"Invalid index_type: {index_type}. Must be one of {valid_types}")
    
    def build(self, embeddings: np.ndarray, metadata: Optional[List[Dict]] = None):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of shape (n_items, dimension)
            metadata: Optional list of metadata dicts for each item
        """
        # Validate embeddings
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
        
        self.n_items, self.dimension = embeddings.shape
        
        # Normalize embeddings for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
        
        # Create index based on type
        if self.index_type == 'flat':
            self.index = self._create_flat_index()
        elif self.index_type == 'ivf':
            self.index = self._create_ivf_index(embeddings)
        elif self.index_type == 'hnsw':
            self.index = self._create_hnsw_index()
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                print(f"Warning: Could not move index to GPU: {e}")
                print("Falling back to CPU indexing")
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        if metadata is not None:
            if len(metadata) != self.n_items:
                raise ValueError(f"Metadata length ({len(metadata)}) must match embeddings ({self.n_items})")
            self.metadata = metadata
        else:
            # Create default metadata with item IDs
            self.metadata = [{'item_id': i} for i in range(self.n_items)]
        
        print(f"Built {self.index_type.upper()} index with {self.n_items} items (dim={self.dimension})")
    
    def _create_flat_index(self) -> faiss.Index:
        """Create flat (exact) index."""
        if self.metric == 'cosine':
            # Cosine similarity is equivalent to inner product after L2 normalization
            index = faiss.IndexFlatIP(self.dimension)
        else:
            # L2 distance
            index = faiss.IndexFlatL2(self.dimension)
        return index
    
    def _create_ivf_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create IVF (inverted file) index with clustering."""
        # Create quantizer (flat index for cluster centroids)
        if self.metric == 'cosine':
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_L2)
        
        # Train the index with embeddings
        print(f"Training IVF index with {self.nlist} clusters...")
        index.train(embeddings.astype(np.float32))
        index.nprobe = self.nprobe  # Search nprobe clusters
        
        return index
    
    def _create_hnsw_index(self) -> faiss.Index:
        """Create HNSW (hierarchical navigable small world) index."""
        if self.metric == 'cosine':
            index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_L2)
        
        # HNSW parameters
        index.hnsw.efConstruction = 40  # Construction time parameter (higher = better quality, slower build)
        index.hnsw.efSearch = 16  # Search time parameter (higher = better recall, slower search)
        
        return index
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        return_metadata: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[Dict]]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query embedding(s) of shape (dimension,) or (n_queries, dimension)
            k: Number of neighbors to return
            return_metadata: Whether to return metadata for results
        
        Returns:
            distances: Array of distances of shape (n_queries, k)
            indices: Array of indices of shape (n_queries, k)
            metadata: List of metadata dicts (if return_metadata=True)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Ensure query is 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Validate query dimension
        if query.shape[1] != self.dimension:
            raise ValueError(f"Query dimension ({query.shape[1]}) must match index dimension ({self.dimension})")
        
        # Normalize query for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(query)
        
        # Search
        distances, indices = self.index.search(query.astype(np.float32), k)
        
        # Convert cosine inner product to distance (1 - similarity)
        if self.metric == 'cosine':
            distances = 1.0 - distances
        
        if return_metadata:
            # Get metadata for results
            result_metadata = []
            for idx_row in indices:
                row_metadata = []
                for idx in idx_row:
                    if idx >= 0 and idx < len(self.metadata):
                        row_metadata.append(self.metadata[idx])
                    else:
                        row_metadata.append({'item_id': -1, 'error': 'invalid_index'})
                result_metadata.append(row_metadata)
            
            return distances, indices, result_metadata
        else:
            return distances, indices
    
    def add(self, embeddings: np.ndarray, metadata: Optional[List[Dict]] = None):
        """
        Add new embeddings to existing index.
        
        Args:
            embeddings: Numpy array of shape (n_new_items, dimension)
            metadata: Optional list of metadata dicts for new items
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Validate embeddings
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) must match index dimension ({self.dimension})")
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        
        # Update metadata
        n_new = embeddings.shape[0]
        if metadata is not None:
            if len(metadata) != n_new:
                raise ValueError(f"Metadata length ({len(metadata)}) must match new embeddings ({n_new})")
            self.metadata.extend(metadata)
        else:
            # Create default metadata
            new_metadata = [{'item_id': self.n_items + i} for i in range(n_new)]
            self.metadata.extend(new_metadata)
        
        self.n_items += n_new
        print(f"Added {n_new} items to index. Total: {self.n_items}")
    
    def remove(self, indices: List[int]):
        """
        Remove items from index by their indices.
        
        Note: FAISS doesn't support efficient removal. This creates a new index without the removed items.
        For large-scale removals, consider rebuilding the entire index.
        """
        raise NotImplementedError("FAISS doesn't support efficient removal. Rebuild index instead.")
    
    def save(self, path: str):
        """
        Save index to disk.
        
        Args:
            path: Base path for saving (will create .index and .metadata files)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = str(path.with_suffix('.index'))
        
        # Move to CPU before saving if on GPU
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        # Save metadata and config
        metadata_path = str(path.with_suffix('.metadata'))
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        config_path = str(path.with_suffix('.config.json'))
        config = {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'metric': self.metric,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'hnsw_m': self.hnsw_m,
            'n_items': self.n_items,
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved index to {path}")
    
    def load(self, path: str):
        """
        Load index from disk.
        
        Args:
            path: Base path for loading (without extension)
        """
        path = Path(path)
        
        # Load config
        config_path = str(path.with_suffix('.config.json'))
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.index_type = config['index_type']
        self.dimension = config['dimension']
        self.metric = config['metric']
        self.nlist = config.get('nlist', 100)
        self.nprobe = config.get('nprobe', 10)
        self.hnsw_m = config.get('hnsw_m', 32)
        self.n_items = config['n_items']
        
        # Load FAISS index
        index_path = str(path.with_suffix('.index'))
        self.index = faiss.read_index(index_path)
        
        # Set nprobe for IVF index
        if self.index_type == 'ivf':
            self.index.nprobe = self.nprobe
        
        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                print(f"Warning: Could not move index to GPU: {e}")
        
        # Load metadata
        metadata_path = str(path.with_suffix('.metadata'))
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded {self.index_type.upper()} index with {self.n_items} items from {path}")
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        if self.index is None:
            return {'status': 'not_built'}
        
        stats = {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'metric': self.metric,
            'n_items': self.n_items,
            'is_trained': self.index.is_trained,
        }
        
        if self.index_type == 'ivf':
            stats.update({
                'nlist': self.nlist,
                'nprobe': self.nprobe,
            })
        elif self.index_type == 'hnsw':
            stats.update({
                'hnsw_m': self.hnsw_m,
            })
        
        return stats
