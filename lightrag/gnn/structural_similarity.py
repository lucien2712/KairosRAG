"""
FastRP Embedding Similarity Calculator
Uses precomputed FastRP embeddings for true structural similarity computation
"""

import numpy as np
from typing import List, Dict
from ..utils import logger


def compute_adaptive_fastrp_similarity(target_entity: str, seed_entities: List[str], node_embedding) -> float:
    """
    True FastRP similarity calculation using precomputed 128-dimensional embeddings
    Computes cosine similarity between FastRP embedding vectors

    Args:
        target_entity: Target entity name
        seed_entities: List of seed entity names
        node_embedding: NodeEmbeddingEnhancer instance with fastrp_embeddings

    Returns:
        Average cosine similarity between target and seed entity embeddings
    """
    try:
        if not node_embedding or not hasattr(node_embedding, 'fastrp_embeddings'):
            logger.warning("FastRP embeddings not available")
            return 0.0

        fastrp_embeddings = node_embedding.fastrp_embeddings
        if not fastrp_embeddings:
            logger.warning("FastRP embeddings dictionary is empty")
            return 0.0

        # Get target entity embedding
        if target_entity not in fastrp_embeddings:
            logger.debug(f"Target entity '{target_entity}' not found in FastRP embeddings")
            return 0.0

        target_embedding = fastrp_embeddings[target_entity]
        similarities = []

        for seed_entity in seed_entities:
            if seed_entity not in fastrp_embeddings:
                logger.debug(f"Seed entity '{seed_entity}' not found in FastRP embeddings")
                continue

            seed_embedding = fastrp_embeddings[seed_entity]

            # Compute cosine similarity between embeddings
            similarity = cosine_similarity(target_embedding, seed_embedding)
            similarities.append(similarity)

            logger.debug(f"FastRP similarity between '{target_entity}' and '{seed_entity}': {similarity:.4f}")

        # Return average similarity across all seed entities
        result = float(np.mean(similarities)) if similarities else 0.0
        logger.debug(f"Average FastRP similarity for '{target_entity}': {result:.4f} (from {len(similarities)} seed entities)")

        return result

    except Exception as e:
        logger.error(f"Error computing FastRP embedding similarity: {e}")
        return 0.0


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between 0 and 1
    """
    try:
        # Normalize vectors to unit length
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)  

    except Exception as e:
        logger.error(f"Error computing cosine similarity: {e}")
        return 0.0