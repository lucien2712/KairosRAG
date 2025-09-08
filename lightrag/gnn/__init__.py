"""
Graph Neural Network module for LightRAG.

This module provides FastRP + Personalized PageRank enhanced embeddings including:
- FastRP embeddings for structural patterns
- Personalized PageRank for entity importance 
- Enhanced retrieval integration
- Path extraction and reasoning
"""

from .node_embedding import NodeEmbeddingEnhancer, NodeEmbeddingConfig

__all__ = [
    "NodeEmbeddingEnhancer",
    "NodeEmbeddingConfig"
]