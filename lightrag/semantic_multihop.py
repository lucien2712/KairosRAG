from __future__ import annotations

import asyncio
import numpy as np
from typing import Any
from collections import defaultdict

from .base import (
    BaseGraphStorage,
    BaseVectorStorage,
    QueryParam,
)
from .utils import (
    logger,
    compute_mdhash_id,
)
from .gnn.node_embedding import (
    compute_embeddings_in_batches as _compute_embeddings_in_batches,
    resolve_positive_int as _resolve_positive_int,
)


# ============================================================================
# Vector Retrieval Functions
# ============================================================================

async def _get_entity_vectors_batch(
    entity_names: list[str],
    entities_vdb: BaseVectorStorage
) -> dict[str, list[float]]:
    """Get entity vectors in batch for multi-hop relevance calculation."""
    try:
        # Generate vector database IDs for all entities
        from .utils import compute_mdhash_id
        entity_vdb_ids = [compute_mdhash_id(name, prefix="ent-") for name in entity_names]

        # Check if entities_vdb supports direct vector access (NanoVectorDB)
        if hasattr(entities_vdb, '_get_client'):
            # Direct access for NanoVectorDB
            client = await entities_vdb._get_client()
            vdb_results = client.get(entity_vdb_ids)

            entity_vectors = {}
            for i, vdb_data in enumerate(vdb_results):
                if vdb_data and 'vector' in vdb_data:
                    entity_name = entity_names[i]
                    try:
                        # Decode NanoVectorDB format: base64 + zlib + float16
                        import base64
                        import numpy as np
                        import zlib

                        compressed_vector = base64.b64decode(vdb_data['vector'])
                        vector_bytes = zlib.decompress(compressed_vector)
                        vector_array = np.frombuffer(vector_bytes, dtype=np.float16).astype(np.float32)
                        entity_vectors[entity_name] = vector_array.tolist()
                    except Exception as e:
                        logger.debug(f"Error decoding vector for {entity_name}: {e}")

            logger.info(f"Multi-hop: Retrieved {len(entity_vectors)}/{len(entity_names)} entity vectors from VDB")
            return entity_vectors
        else:
            # Fallback: use regular get_by_ids (without vectors) - will still compute embeddings
            logger.debug("Vector storage doesn't support direct vector access, will compute embeddings")
            return {}

    except Exception as e:
        logger.debug(f"Error getting entity vectors batch: {e}")
        return {}

async def _get_relationship_vectors_batch(
    relationships: list[dict],
    relationships_vdb: BaseVectorStorage
) -> dict[str, list[float]]:
    """Get relationship vectors in batch for multi-hop relevance calculation."""
    try:
        # Generate vector database IDs for all relationships
        from .utils import compute_mdhash_id
        rel_vdb_ids = []
        rel_keys = []

        for rel in relationships:
            src_id = rel.get('src_id', '')
            tgt_id = rel.get('tgt_id', '')
            if src_id and tgt_id:
                # IMPORTANT: Must match insert format (line 2131) - no separator between src and tgt
                # For undirected graphs, try BOTH directions (src+tgt and tgt+src)
                rel_key_1 = src_id + tgt_id
                rel_key_2 = tgt_id + src_id
                vdb_id_1 = compute_mdhash_id(rel_key_1, prefix="rel-")
                vdb_id_2 = compute_mdhash_id(rel_key_2, prefix="rel-")

                # Add both possible IDs (VDB lookup will filter to existing ones)
                rel_vdb_ids.append(vdb_id_1)
                if vdb_id_2 != vdb_id_1:  # Avoid duplicates
                    rel_vdb_ids.append(vdb_id_2)

                # Store normalized key for later lookup (lexicographic order)
                if src_id > tgt_id:
                    normalized_key = f"{tgt_id}-{src_id}"
                else:
                    normalized_key = f"{src_id}-{tgt_id}"
                rel_keys.append(normalized_key)

        # Check if relationships_vdb supports direct vector access (NanoVectorDB)
        if hasattr(relationships_vdb, '_get_client'):
            # Direct access for NanoVectorDB
            client = await relationships_vdb._get_client()
            storage = getattr(client, '_NanoVectorDB__storage', None)

            if storage is None:
                logger.warning("Cannot access NanoVectorDB storage directly, using client.get()")
                vdb_results = client.get(rel_vdb_ids)
            else:
                # Build ID lookup map from storage
                storage_map = {item['__id__']: item for item in storage.get('data', [])}
                # Get results for requested IDs
                vdb_results = [storage_map.get(vdb_id) for vdb_id in rel_vdb_ids]

            relationship_vectors = {}
            for vdb_data in vdb_results:
                if vdb_data and 'vector' in vdb_data:
                    # Get src_id and tgt_id from VDB data
                    src_id = vdb_data.get('src_id')
                    tgt_id = vdb_data.get('tgt_id')

                    if not src_id or not tgt_id:
                        continue

                    # Normalize key using lexicographic order
                    if src_id > tgt_id:
                        normalized_key = f"{tgt_id}-{src_id}"
                    else:
                        normalized_key = f"{src_id}-{tgt_id}"

                    # Skip if already decoded (avoid duplicate decoding for bidirectional edges)
                    if normalized_key in relationship_vectors:
                        continue

                    try:
                        # Decode NanoVectorDB format: base64 + zlib + float16
                        import base64
                        import numpy as np
                        import zlib

                        compressed_vector = base64.b64decode(vdb_data['vector'])
                        vector_bytes = zlib.decompress(compressed_vector)
                        vector_array = np.frombuffer(vector_bytes, dtype=np.float16).astype(np.float32)
                        relationship_vectors[normalized_key] = vector_array.tolist()
                    except Exception as e:
                        logger.debug(f"Error decoding vector for {normalized_key}: {e}")

            logger.info(f"Multi-hop: Retrieved {len(relationship_vectors)}/{len(rel_keys)} relationship vectors from VDB")
            return relationship_vectors
        else:
            # Fallback: use regular get_by_ids (without vectors) - will still compute embeddings
            logger.debug("Vector storage doesn't support direct vector access, will compute embeddings")
            return {}

    except Exception as e:
        logger.debug(f"Error getting relationship vectors batch: {e}")
        return {}


# ============================================================================
# Scoring Functions
# ============================================================================

def _calculate_relevance_scores_vectorized(
    candidates: list[dict],
    nodes_data: list[dict],
    edges_data: list[dict],
    current_hop: int,
    ll_keywords: str,
    hl_keywords: str,
    entity_vectors_cache: dict[str, list[float]],
    relationship_vectors_cache: dict[str, list[float]],
    query_ll_embedding: list[float],
    query_hl_embedding: list[float],
) -> list[float]:
    """Calculate relevance scores for all candidates using vectorized numpy operations."""
    import numpy as np

    n_candidates = len(candidates)
    if n_candidates == 0:
        return []

    # Prepare matrices for vectorized computation
    ll_similarities = np.zeros(n_candidates)
    hl_similarities = np.zeros(n_candidates)

    # 1. Vectorized Low-level similarity calculation (entities)
    if query_ll_embedding is not None and entity_vectors_cache:
        query_ll_vec = np.array(query_ll_embedding)
        query_ll_norm = np.linalg.norm(query_ll_vec)

        if query_ll_norm > 0:
            entity_vectors = []
            valid_ll_indices = []

            for i, (candidate, node_data) in enumerate(zip(candidates, nodes_data)):
                entity_name = candidate.get("entity_name", "")
                if entity_name in entity_vectors_cache:
                    entity_vectors.append(entity_vectors_cache[entity_name])
                    valid_ll_indices.append(i)

            if entity_vectors:
                # Stack entity vectors into matrix: (n_valid_entities, embed_dim)
                entity_matrix = np.array(entity_vectors)

                # Vectorized cosine similarity computation
                entity_norms = np.linalg.norm(entity_matrix, axis=1)
                valid_entity_mask = entity_norms > 0

                if np.any(valid_entity_mask):
                    # Compute dot products for all entities at once
                    dot_products = np.dot(entity_matrix, query_ll_vec)

                    # Vectorized cosine similarity
                    cosine_sims = np.zeros(len(entity_vectors))
                    cosine_sims[valid_entity_mask] = dot_products[valid_entity_mask] / (
                        entity_norms[valid_entity_mask] * query_ll_norm
                    )
                    cosine_sims = np.maximum(cosine_sims, 0.0)  # Ensure non-negative

                    # Map back to original candidate indices
                    for j, original_idx in enumerate(valid_ll_indices):
                        ll_similarities[original_idx] = cosine_sims[j]

    # 2. Vectorized High-level similarity calculation (relationships)
    if query_hl_embedding is not None and relationship_vectors_cache:
        query_hl_vec = np.array(query_hl_embedding)
        query_hl_norm = np.linalg.norm(query_hl_vec)

        if query_hl_norm > 0:
            relation_vectors = []
            valid_hl_indices = []

            for i, (candidate, edge_data) in enumerate(zip(candidates, edges_data)):
                edge = candidate.get("edge", {})
                edge_src = edge.get('src_id', '')
                edge_tgt = edge.get('tgt_id', '')

                # Normalize key using lexicographic order (matching _get_relationship_vectors_batch)
                if edge_src > edge_tgt:
                    edge_key = f"{edge_tgt}-{edge_src}"
                else:
                    edge_key = f"{edge_src}-{edge_tgt}"

                if edge_key in relationship_vectors_cache:
                    relation_vectors.append(relationship_vectors_cache[edge_key])
                    valid_hl_indices.append(i)

            if relation_vectors:
                # Stack relation vectors into matrix: (n_valid_relations, embed_dim)
                relation_matrix = np.array(relation_vectors)

                # Vectorized cosine similarity computation
                relation_norms = np.linalg.norm(relation_matrix, axis=1)
                valid_relation_mask = relation_norms > 0

                if np.any(valid_relation_mask):
                    # Compute dot products for all relations at once
                    dot_products = np.dot(relation_matrix, query_hl_vec)

                    # Vectorized cosine similarity
                    cosine_sims = np.zeros(len(relation_vectors))
                    cosine_sims[valid_relation_mask] = dot_products[valid_relation_mask] / (
                        relation_norms[valid_relation_mask] * query_hl_norm
                    )
                    cosine_sims = np.maximum(cosine_sims, 0.0)  # Ensure non-negative

                    # Map back to original candidate indices
                    for j, original_idx in enumerate(valid_hl_indices):
                        hl_similarities[original_idx] = cosine_sims[j]

    # 3. Vectorized distance decay calculation
    decay = 0.8 ** (current_hop - 1)

    # 4. Vectorized final score computation
    final_scores = (
        0.4 * ll_similarities +
        0.5 * hl_similarities +
        0.1 * decay
    )
    """
    0.35 * ll_similarities +
        0.45 * hl_similarities +
        0.2 * decay

      0.4 * ll_similarities +
        0.4 * hl_similarities +
        0.2 * decay
    """

    return final_scores.tolist()


# ============================================================================
# Relation Finding Functions
# ============================================================================

async def _find_relations_for_entities(
    entities: list[dict],
    relationships_vdb: BaseVectorStorage,
    knowledge_graph_inst: BaseGraphStorage | None = None,
    max_concurrent: int = 30,
) -> list[dict]:
    """Find relations connecting given entities with parallel queries.

    Note: This uses vector database queries (not embedding API calls),
    so we can safely use higher concurrency without hitting API rate limits.
    Default: 30 (suitable for most VDB backends)
    """
    if not entities:
        return []

    try:
        entity_names = [e.get("entity_name", "") for e in entities]
        entity_names_set = set(entity_names)  # Convert to set for faster lookups

        # Fast path: use knowledge graph adjacency to gather relations
        if knowledge_graph_inst and entity_names_set:
            try:
                nodes_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(list(entity_names_set))

                edge_pairs: list[dict[str, str]] = []
                seen_edge_pairs: set[tuple[str, str]] = set()

                for node_name, edges in nodes_edges_dict.items():
                    for edge in edges or []:
                        src_id, tgt_id = edge[0], edge[1]
                        if src_id in entity_names_set and tgt_id in entity_names_set:
                            canonical_pair = tuple(sorted((src_id, tgt_id)))
                            if canonical_pair not in seen_edge_pairs:
                                edge_pairs.append({"src": src_id, "tgt": tgt_id})
                                seen_edge_pairs.add(canonical_pair)

                if edge_pairs:
                    edge_data = await knowledge_graph_inst.get_edges_batch(edge_pairs)
                    relations = []
                    for edge_pair in edge_pairs:
                        src_id = edge_pair["src"]
                        tgt_id = edge_pair["tgt"]
                        edge_props = (
                            edge_data.get((src_id, tgt_id))
                            or edge_data.get((tgt_id, src_id))
                        )
                        if not edge_props:
                            continue
                        relation_info = {
                            **edge_props,
                            "src_id": edge_props.get("src_id", src_id),
                            "tgt_id": edge_props.get("tgt_id", tgt_id),
                            "description": edge_props.get("description", ""),
                            "keywords": edge_props.get("keywords", ""),
                            "rank": edge_props.get("rank", 0),
                            "discovery_method": "entity_relation",
                        }
                        relations.append(relation_info)

                    if relations:
                        logger.debug(
                            f"Found {len(relations)} relations via knowledge graph adjacency for {len(entity_names_set)} entities"
                        )
                        return relations
            except Exception as kg_error:
                logger.debug(
                    f"Knowledge graph relation lookup failed, falling back to vector search: {kg_error}"
                )

        # Fallback: vector database queries (embedding + search)
        # Pre-compute query embeddings once if possible
        query_embeddings: dict[str, list[float]] = {}
        embedding_func = getattr(relationships_vdb, "embedding_func", None)
        if embedding_func and getattr(embedding_func, "func", None):
            unique_entity_names = list(dict.fromkeys(entity_names))
            batch_fallback = 16  # vector DB queries tolerate higher batch size
            batch_size = _resolve_positive_int(
                getattr(relationships_vdb, "global_config", {}).get("embedding_batch_size", None),
                batch_fallback,
            )
            try:
                embeddings = await _compute_embeddings_in_batches(
                    unique_entity_names, embedding_func, batch_size
                )
                for name, embedding in zip(unique_entity_names, embeddings):
                    query_embeddings[name] = embedding
            except Exception as exc:
                logger.debug(f"Relation embedding pre-compute failed, will fallback to inline embedding: {exc}")
                query_embeddings = {}

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def query_entity_relations(entity_name: str) -> list[dict]:
            """Query relations for a single entity with semaphore control."""
            async with semaphore:
                try:
                    relation_results = await relationships_vdb.query(
                        entity_name,
                        top_k=10,
                        query_embedding=query_embeddings.get(entity_name),
                    )
                    return relation_results
                except Exception as e:
                    logger.debug(f"Error finding relations for {entity_name}: {e}")
                    return []

        # Parallel query all entity relations (deduplicated)
        unique_entity_order = list(dict.fromkeys(entity_names))
        tasks = [query_entity_relations(entity_name) for entity_name in unique_entity_order]
        unique_relation_results = await asyncio.gather(*tasks)
        relation_lookup = dict(zip(unique_entity_order, unique_relation_results))

        # Process results and deduplicate
        seen_relations = set()
        relations = []

        for entity_name in entity_names:
            relation_results = relation_lookup.get(entity_name, [])
            for relation_data in relation_results:
                src_id = relation_data.get("src_id", "")
                tgt_id = relation_data.get("tgt_id", "")

                # Keep relations that connect our entities
                if src_id in entity_names_set and tgt_id in entity_names_set:
                    relation_key = f"{src_id}|{tgt_id}"

                    if relation_key not in seen_relations:
                        relation_info = {
                            "src_id": src_id,
                            "tgt_id": tgt_id,
                            "description": relation_data.get("description", ""),
                            "keywords": relation_data.get("keywords", ""),
                            "discovery_method": "entity_relation",
                            "rank": relation_data.get("rank", 0),
                            **relation_data
                        }
                        relations.append(relation_info)
                        seen_relations.add(relation_key)

        logger.debug(f"Found {len(relations)} relations for {len(entity_names)} entities via vector search fallback")
        return relations

    except Exception as e:
        logger.error(f"Error finding relations for entities: {e}")
        return []


# ============================================================================
# Core Multi-hop Expansion Functions
# ============================================================================

async def _original_multi_hop_expand(
    seed_nodes: list[dict],
    ll_keywords: str,
    hl_keywords: str,
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    query_ll_embedding: list[float] | None = None,
    query_hl_embedding: list[float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Original multi-hop expansion logic (semantic path only).

    Args:
        query_ll_embedding: Pre-computed ll_keywords embedding (optional, for caching)
        query_hl_embedding: Pre-computed hl_keywords embedding (optional, for caching)
    """

    logger.info(f"Starting multi-hop expansion: max_hop={query_param.max_hop}, top_neighbors={query_param.top_neighbors}, threshold={query_param.multi_hop_relevance_threshold}")

    # Use pre-computed embeddings if provided, otherwise compute them
    # Note: Use 'is None' check to avoid ambiguous truth value errors with arrays
    need_compute = (query_ll_embedding is None) or (query_hl_embedding is None)
    if need_compute:
        try:
            # Pre-compute LL query embedding if not provided
            if query_ll_embedding is None and ll_keywords and entities_vdb.embedding_func and entities_vdb.embedding_func.func:
                query_ll_result = await entities_vdb.embedding_func.func([ll_keywords])
                query_ll_embedding = query_ll_result[0]

            # Pre-compute HL query embedding if not provided
            if query_hl_embedding is None and hl_keywords and relationships_vdb.embedding_func and relationships_vdb.embedding_func.func:
                query_hl_result = await relationships_vdb.embedding_func.func([hl_keywords])
                query_hl_embedding = query_hl_result[0]

            logger.debug(f"Computed missing query embeddings: ll={'✓' if query_ll_embedding is not None else '✗'}, hl={'✓' if query_hl_embedding is not None else '✗'}")
        except Exception as e:
            logger.debug(f"Error computing query embeddings: {e}")
    else:
        logger.debug(f"Using pre-computed query embeddings: ll={'✓' if query_ll_embedding is not None else '✗'}, hl={'✓' if query_hl_embedding is not None else '✗'}")

    all_expanded_entities = []
    all_expanded_relations = []
    current_nodes = seed_nodes.copy()
    visited_nodes = set(node["entity_name"] for node in seed_nodes)
    visited_relations = set()  # Track visited (src_id, tgt_id) pairs to prevent duplicates and cycles

    # BFS expansion for all hops
    # Track prefetch task for next hop
    prefetch_task = None

    for hop in range(1, query_param.max_hop + 1):
        logger.info(f"Multi-hop: Starting hop {hop} with {len(current_nodes)} current nodes")
        if not current_nodes:
            logger.info(f"Multi-hop: No current nodes at hop {hop}, stopping expansion")
            # Cancel any pending prefetch
            if prefetch_task and not prefetch_task.done():
                prefetch_task.cancel()
            break

        logger.debug(f"Processing hop {hop} with {len(current_nodes)} nodes")
        next_nodes = []
        hop_relations = []

        # Get neighbors for current nodes (use prefetch if available)
        node_names = [node["entity_name"] for node in current_nodes]

        # If prefetch task exists from previous hop, await it
        if prefetch_task is not None:
            try:
                batch_edges_dict = await prefetch_task
                logger.debug(f"Hop {hop}: Using prefetched edge data ({len(batch_edges_dict)} entries)")
            except asyncio.CancelledError:
                logger.debug(f"Hop {hop}: Prefetch cancelled, fetching normally")
                batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)
            except Exception as e:
                logger.warning(f"Hop {hop}: Prefetch failed ({e}), fetching normally")
                batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)
            finally:
                prefetch_task = None
        else:
            # No prefetch available, fetch normally
            batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)

        # Collect all neighboring nodes and edges
        neighbor_candidates = []
        total_edges = 0

        # First pass: collect all unique edges for batch retrieval
        all_edge_pairs = []
        edge_to_node_mapping = {}  # Maps edge tuple to (node_name, neighbor_name)

        for node_name in node_names:
            edges = batch_edges_dict.get(node_name, [])
            total_edges += len(edges)
            logger.debug(f"Node '{node_name}' has {len(edges)} edges")

            for edge in edges:
                edge_key = (edge[0], edge[1])
                neighbor_name = edge[1] if edge[0] == node_name else edge[0]

                if neighbor_name not in visited_nodes:
                    # Add to batch if not already added
                    if edge_key not in edge_to_node_mapping:
                        all_edge_pairs.append({"src": edge[0], "tgt": edge[1]})
                        edge_to_node_mapping[edge_key] = []
                    edge_to_node_mapping[edge_key].append((node_name, neighbor_name))

        # Prepare all batch requests in parallel
        import asyncio

        # Prepare edge batch request
        async def get_edges_task():
            if all_edge_pairs:
                return await knowledge_graph_inst.get_edges_batch(all_edge_pairs)
            return {}

        # Prepare unique collections for parallel requests (optimized single pass)
        unique_entity_names_set = set()
        for edge_key, node_neighbor_pairs in edge_to_node_mapping.items():
            for node_name, neighbor_name in node_neighbor_pairs:
                unique_entity_names_set.add(neighbor_name)

        unique_entity_names = list(unique_entity_names_set)
        unique_edge_keys = list(edge_to_node_mapping.keys())
        unique_relationships = [{"src_id": edge_key[0], "tgt_id": edge_key[1]} for edge_key in unique_edge_keys]

        # Prepare all async tasks
        async def get_entity_vectors_task():
            if unique_entity_names:
                try:
                    return await _get_entity_vectors_batch(unique_entity_names, entities_vdb)
                except Exception as e:
                    logger.debug(f"Error fetching entity vectors batch: {e}")
                    return {}
            return {}

        async def get_relationship_vectors_task():
            if unique_relationships:
                try:
                    return await _get_relationship_vectors_batch(unique_relationships, relationships_vdb)
                except Exception as e:
                    logger.debug(f"Error fetching relationship vectors batch: {e}")
                    return {}
            return {}

        async def get_nodes_task():
            if unique_entity_names:
                return await knowledge_graph_inst.get_nodes_batch(unique_entity_names)
            return {}

        # Execute all batch requests in parallel
        logger.debug(f"Multi-hop parallel optimization: Fetching {len(all_edge_pairs)} edges, {len(unique_entity_names)} entities, {len(unique_relationships)} relationships, and {len(unique_entity_names)} nodes in parallel")

        all_edge_data_dict, entity_vectors_cache, relationship_vectors_cache, all_nodes_dict = await asyncio.gather(
            get_edges_task(),
            get_entity_vectors_task(),
            get_relationship_vectors_task(),
            get_nodes_task(),
            return_exceptions=True
        )

        # Handle any exceptions from parallel execution
        if isinstance(all_edge_data_dict, Exception):
            logger.warning(f"Error in edge batch retrieval: {all_edge_data_dict}")
            all_edge_data_dict = {}
        if isinstance(entity_vectors_cache, Exception):
            logger.warning(f"Error in entity vectors batch retrieval: {entity_vectors_cache}")
            entity_vectors_cache = {}
        if isinstance(relationship_vectors_cache, Exception):
            logger.warning(f"Error in relationship vectors batch retrieval: {relationship_vectors_cache}")
            relationship_vectors_cache = {}
        if isinstance(all_nodes_dict, Exception):
            logger.warning(f"Error in nodes batch retrieval: {all_nodes_dict}")
            all_nodes_dict = {}

        logger.debug(f"Parallel retrieval completed: {len(all_edge_data_dict)} edges, {len(entity_vectors_cache)} entity vectors, {len(relationship_vectors_cache)} relationship vectors, {len(all_nodes_dict)} nodes")

        # Build neighbor candidates using batch results
        neighbor_candidates = []
        for edge_key, node_neighbor_pairs in edge_to_node_mapping.items():
            edge_data = all_edge_data_dict.get(edge_key)

            if edge_data:
                edge_info = {
                    "src_id": edge_key[0],
                    "tgt_id": edge_key[1],
                    **edge_data
                }

                for node_name, neighbor_name in node_neighbor_pairs:
                    neighbor_candidates.append({
                        "seed_node": node_name,  # Track which seed node this neighbor came from
                        "entity_name": neighbor_name,
                        "edge": edge_info
                    })

        # Score and select best neighbors using vectorized computation
        scored_neighbors = []

        if neighbor_candidates:
            # Prepare data structures for vectorized scoring
            candidates_with_node_data = []
            nodes_data_list = []
            edges_data_list = []

            for candidate in neighbor_candidates:
                node_data = all_nodes_dict.get(candidate["entity_name"])
                if node_data:
                    # Add entity_name to node_data
                    full_node_data = {
                        **node_data,
                        "entity_name": candidate["entity_name"],
                        "rank": node_data.get("degree", 0)
                    }
                    candidates_with_node_data.append(candidate)
                    nodes_data_list.append(full_node_data)
                    edges_data_list.append(candidate["edge"])

            if candidates_with_node_data:
                logger.debug(f"Multi-hop vectorized scoring: Processing {len(candidates_with_node_data)} candidates with numpy vectorization")

                # Vectorized relevance scoring for all candidates at once
                scores = _calculate_relevance_scores_vectorized(
                    candidates_with_node_data,
                    nodes_data_list,
                    edges_data_list,
                    hop,
                    ll_keywords,
                    hl_keywords,
                    entity_vectors_cache,
                    relationship_vectors_cache,
                    query_ll_embedding,
                    query_hl_embedding
                )

                # Apply threshold and build scored neighbors (with seed node tracking)
                for i, (candidate, node_data, score) in enumerate(zip(candidates_with_node_data, nodes_data_list, scores)):
                    if score >= query_param.multi_hop_relevance_threshold:
                        scored_neighbors.append((candidate["seed_node"], node_data, candidate["edge"], score))
                    else:
                        logger.debug(f"Node {candidate['entity_name']} rejected: score {score:.3f} < threshold {query_param.multi_hop_relevance_threshold}")

                # Log performance metrics
                logger.debug(f"Vectorized scoring completed: {len(scored_neighbors)} qualified out of {len(candidates_with_node_data)} candidates")

        logger.info(f"Multi-hop: Hop {hop} found {len(scored_neighbors)} qualified nodes")

        # Group neighbors by seed node and select top N per seed
        from collections import defaultdict
        neighbors_by_seed = defaultdict(list)
        for seed_node, node_data, edge_data, score in scored_neighbors:
            neighbors_by_seed[seed_node].append((node_data, edge_data, score))

        # Select top_neighbors for each seed node
        selected = []
        for seed_node, neighbors in neighbors_by_seed.items():
            # Sort by score and select top N for this seed
            neighbors.sort(key=lambda x: x[2], reverse=True)
            top_n = neighbors[:query_param.top_neighbors]
            selected.extend(top_n)
            logger.debug(f"Seed node '{seed_node}': selected {len(top_n)} neighbors from {len(neighbors)} candidates")

        logger.info(f"Multi-hop: Selected {len(selected)} total neighbors from {len(neighbors_by_seed)} seed nodes (max {query_param.top_neighbors} per seed)")

        if not selected:
            logger.debug(f"No relevant neighbors found at hop {hop}, stopping expansion")
            break

        # Collect next hop candidates for prefetching
        next_hop_candidates = []
        for node_data, _, _ in selected:
            next_hop_candidates.append(node_data["entity_name"])

        # Start prefetching next hop edges asynchronously (if not last hop)
        if hop < query_param.max_hop and next_hop_candidates:
            async def prefetch_next_hop_edges():
                """Prefetch edges for next hop candidates."""
                try:
                    logger.debug(f"Prefetch: Starting edge fetch for {len(next_hop_candidates)} candidates (hop {hop+1})")
                    result = await knowledge_graph_inst.get_nodes_edges_batch(next_hop_candidates)
                    logger.debug(f"Prefetch: Completed edge fetch for hop {hop+1}")
                    return result
                except Exception as e:
                    logger.debug(f"Prefetch: Failed for hop {hop+1}: {e}")
                    raise

            # Create prefetch task (non-blocking)
            prefetch_task = asyncio.create_task(prefetch_next_hop_edges())
            logger.debug(f"Prefetch: Started background task for hop {hop+1} ({len(next_hop_candidates)} nodes)")

        # Add selected neighbors to results (with relation deduplication and cycle prevention)
        for node_data, edge_data, score in selected:
            entity_name = node_data["entity_name"]

            # Add entity if not visited
            if entity_name not in visited_nodes:
                next_nodes.append(node_data)
                all_expanded_entities.append(node_data)
                visited_nodes.add(entity_name)

                # Only add relation if target node is new (prevents cycles like A→B→A)
                edge_key = (edge_data["src_id"], edge_data["tgt_id"])
                if edge_key not in visited_relations:
                    hop_relations.append(edge_data)
                    visited_relations.add(edge_key)

        all_expanded_relations.extend(hop_relations)
        current_nodes = next_nodes

    # Clean up any remaining prefetch task
    if prefetch_task and not prefetch_task.done():
        prefetch_task.cancel()
        logger.debug("Cleaned up unused prefetch task")

    logger.info(f"Multi-hop expansion completed: {len(all_expanded_entities)} entities, {len(all_expanded_relations)} relations")
    return all_expanded_entities, all_expanded_relations


# ============================================================================
# Core Multi-hop Expansion (Semantic Path Only)
# ============================================================================
# This module contains ONLY the semantic multi-hop expansion logic.
# Three-perspective coordination (multihop + PPR + FastRP) is handled in operate.py.
