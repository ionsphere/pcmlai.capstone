"""
Vector Search Module

Provides vector similarity search capabilities for clothing items using various backends.
"""

from .indexing import VectorIndex, FAISSIndex
from .similarity import SimilaritySearch

__all__ = ['VectorIndex', 'FAISSIndex', 'SimilaritySearch']
