"""
Similarity Search Module

Provides high-level similarity search functionality with filtering and pricing context.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import pandas as pd
import json

from .indexing import VectorIndex, FAISSIndex


class SimilaritySearch:
    """
    High-level similarity search for clothing items.
    
    Features:
    - k-NN search with filtering
    - Category-based filtering
    - Price range filtering
    - Pricing context generation from similar items
    - Confidence scoring
    """
    
    def __init__(
        self,
        index: VectorIndex,
        items_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize similarity search.
        
        Args:
            index: Vector index (FAISS, Annoy, etc.)
            items_data: DataFrame with item metadata (id, category, price, etc.)
        """
        self.index = index
        self.items_data = items_data
        
        # Build lookup tables for filtering
        if items_data is not None:
            self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build lookup tables for efficient filtering."""
        # Index by item ID
        if 'item_id' in self.items_data.columns:
            self.id_to_idx = {
                item_id: idx
                for idx, item_id in enumerate(self.items_data['item_id'])
            }
        else:
            self.id_to_idx = {idx: idx for idx in range(len(self.items_data))}
        
        # Index by category
        if 'category' in self.items_data.columns:
            self.category_indices = {}
            for idx, category in enumerate(self.items_data['category']):
                if category not in self.category_indices:
                    self.category_indices[category] = []
                self.category_indices[category].append(idx)
        else:
            self.category_indices = {}
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        category: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None,
        condition_range: Optional[Tuple[float, float]] = None,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search for similar items with filtering.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            category: Filter by clothing category (optional)
            price_range: Filter by price range (min, max) (optional)
            condition_range: Filter by condition score (min, max) (optional)
            filters: Additional custom filters (optional)
        
        Returns:
            List of result dictionaries with metadata and scores
        """
        # Search in index (fetch more than k to allow for filtering)
        search_k = min(k * 5, self.index.n_items)
        distances, indices, metadata = self.index.search(
            query_embedding,
            k=search_k,
            return_metadata=True
        )
        
        # Flatten results (handle batch queries)
        if distances.ndim == 2:
            distances = distances[0]
            indices = indices[0]
            metadata = metadata[0]
        
        # Build results with metadata
        results = []
        for dist, idx, meta in zip(distances, indices, metadata):
            # Skip invalid indices
            if idx < 0 or idx >= len(self.items_data):
                continue
            
            # Get item data
            item_data = self.items_data.iloc[idx].to_dict()
            
            # Apply filters
            if not self._passes_filters(item_data, category, price_range, condition_range, filters):
                continue
            
            # Build result
            result = {
                'index': int(idx),
                'distance': float(dist),
                'similarity': float(1.0 - dist),  # Convert distance to similarity
                **item_data,
                **meta,
            }
            results.append(result)
            
            # Stop when we have enough results
            if len(results) >= k:
                break
        
        return results
    
    def _passes_filters(
        self,
        item: Dict,
        category: Optional[str],
        price_range: Optional[Tuple[float, float]],
        condition_range: Optional[Tuple[float, float]],
        filters: Optional[Dict],
    ) -> bool:
        """Check if item passes all filters."""
        # Category filter
        if category is not None:
            if item.get('category') != category:
                return False
        
        # Price range filter
        if price_range is not None:
            price = item.get('price')
            if price is None:
                return False
            min_price, max_price = price_range
            if price < min_price or price > max_price:
                return False
        
        # Condition range filter
        if condition_range is not None:
            condition = item.get('condition_score')
            if condition is None:
                return False
            min_cond, max_cond = condition_range
            if condition < min_cond or condition > max_cond:
                return False
        
        # Custom filters
        if filters is not None:
            for key, value in filters.items():
                if callable(value):
                    # Custom filter function
                    if not value(item):
                        return False
                else:
                    # Exact match
                    if item.get(key) != value:
                        return False
        
        return True
    
    def get_pricing_context(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        category: Optional[str] = None,
        condition_range: Optional[Tuple[float, float]] = None,
    ) -> Dict:
        """
        Get pricing context from similar items.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar items to analyze
            category: Filter by category (optional)
            condition_range: Filter by condition (optional)
        
        Returns:
            Dictionary with pricing statistics and context
        """
        # Search for similar items
        results = self.search(
            query_embedding=query_embedding,
            k=k,
            category=category,
            condition_range=condition_range,
        )
        
        if not results:
            return {
                'n_similar': 0,
                'error': 'No similar items found',
            }
        
        # Extract prices
        prices = [r['price'] for r in results if 'price' in r and r['price'] is not None]
        
        if not prices:
            return {
                'n_similar': len(results),
                'error': 'No price information available',
            }
        
        prices = np.array(prices)
        
        # Calculate statistics
        context = {
            'n_similar': len(results),
            'n_with_prices': len(prices),
            'mean_price': float(np.mean(prices)),
            'median_price': float(np.median(prices)),
            'std_price': float(np.std(prices)),
            'min_price': float(np.min(prices)),
            'max_price': float(np.max(prices)),
            'q25_price': float(np.percentile(prices, 25)),
            'q75_price': float(np.percentile(prices, 75)),
            'iqr_price': float(np.percentile(prices, 75) - np.percentile(prices, 25)),
        }
        
        # Identify outliers (beyond 1.5 * IQR)
        q1, q3 = np.percentile(prices, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
        
        context['n_outliers'] = len(outliers)
        context['outlier_ratio'] = len(outliers) / len(prices) if len(prices) > 0 else 0.0
        
        # Calculate confidence score
        # Higher confidence when:
        # 1. More similar items found
        # 2. Lower standard deviation
        # 3. Fewer outliers
        confidence = self._calculate_confidence(
            n_similar=len(results),
            std=context['std_price'],
            mean=context['mean_price'],
            outlier_ratio=context['outlier_ratio'],
        )
        context['confidence'] = confidence
        
        # Add similar items details
        context['similar_items'] = [
            {
                'item_id': r.get('item_id'),
                'price': r.get('price'),
                'category': r.get('category'),
                'condition_score': r.get('condition_score'),
                'similarity': r.get('similarity'),
            }
            for r in results[:5]  # Top 5 most similar
        ]
        
        return context
    
    def _calculate_confidence(
        self,
        n_similar: int,
        std: float,
        mean: float,
        outlier_ratio: float,
    ) -> float:
        """
        Calculate confidence score for pricing context.
        
        Score from 0 to 1, where:
        - 1.0 = high confidence (many similar items, low variance, no outliers)
        - 0.0 = low confidence (few items, high variance, many outliers)
        """
        # Component 1: Number of similar items (more is better)
        # Scale: 5 items = 0.5, 10+ items = 1.0
        n_score = min(n_similar / 10.0, 1.0)
        
        # Component 2: Coefficient of variation (lower is better)
        # CV = std / mean. Lower CV means more consistent prices
        cv = std / mean if mean > 0 else 1.0
        cv_score = 1.0 / (1.0 + cv)  # Convert to 0-1 range (higher is better)
        
        # Component 3: Outlier ratio (fewer is better)
        outlier_score = 1.0 - outlier_ratio
        
        # Weighted average
        confidence = (
            0.4 * n_score +
            0.4 * cv_score +
            0.2 * outlier_score
        )
        
        return float(confidence)
    
    def get_price_distribution(
        self,
        query_embedding: np.ndarray,
        k: int = 50,
        category: Optional[str] = None,
    ) -> Dict:
        """
        Get price distribution from similar items for visualization.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar items to analyze
            category: Filter by category (optional)
        
        Returns:
            Dictionary with distribution data
        """
        # Get similar items
        results = self.search(
            query_embedding=query_embedding,
            k=k,
            category=category,
        )
        
        if not results:
            return {'error': 'No similar items found'}
        
        # Extract prices and similarities
        prices = []
        similarities = []
        for r in results:
            if 'price' in r and r['price'] is not None:
                prices.append(r['price'])
                similarities.append(r.get('similarity', 0.0))
        
        if not prices:
            return {'error': 'No price information available'}
        
        prices = np.array(prices)
        similarities = np.array(similarities)
        
        # Calculate histogram
        n_bins = min(10, len(prices) // 3)  # Adaptive number of bins
        hist, bin_edges = np.histogram(prices, bins=n_bins)
        
        return {
            'prices': prices.tolist(),
            'similarities': similarities.tolist(),
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
            },
            'n_items': len(prices),
        }
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        **kwargs
    ) -> List[List[Dict]]:
        """
        Perform batch similarity search.
        
        Args:
            query_embeddings: Batch of query embeddings (n_queries, dimension)
            k: Number of results per query
            **kwargs: Additional search parameters
        
        Returns:
            List of result lists (one per query)
        """
        results = []
        for query in query_embeddings:
            query_results = self.search(query, k=k, **kwargs)
            results.append(query_results)
        
        return results
    
    def save_config(self, path: str):
        """Save similarity search configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            'index_type': self.index.__class__.__name__,
            'index_stats': self.index.get_stats(),
            'n_items': len(self.items_data) if self.items_data is not None else 0,
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved similarity search config to {path}")
