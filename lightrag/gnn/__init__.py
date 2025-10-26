"""
Graph Neural Network module for LightRAG.

This module provides FastRP embeddings and query-time Personalized PageRank:
- FastRP embeddings for structural similarity patterns
- Query-aware Personalized PageRank computed dynamically at query time
- Enhanced retrieval integration with multi-perspective expansion
- Path extraction and reasoning capabilities
"""

from .node_embedding import NodeEmbeddingEnhancer, NodeEmbeddingConfig

__all__ = [
    "NodeEmbeddingEnhancer",
    "NodeEmbeddingConfig"
]