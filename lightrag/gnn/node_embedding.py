from __future__ import annotations

"""Node embedding enhancer using FastRP and Personalized PageRank."""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
import os
import pickle
import json
from scipy import sparse

from ..utils import cosine_similarity
from .structural_similarity import compute_adaptive_fastrp_batch

logger = logging.getLogger(__name__)


def resolve_positive_int(value: int | str | None, default: int) -> int:
    """Return a positive integer from value or fallback to default."""
    try:
        candidate = int(value)  # type: ignore[arg-type]
        if candidate > 0:
            return candidate
    except (TypeError, ValueError):
        pass
    return default


async def compute_embeddings_in_batches(
    items: List[str],
    embedding_func,
    batch_size: int,
) -> List:
    """Compute embeddings for the given items in batches while preserving order."""
    if batch_size <= 0:
        batch_size = 1

    embeddings: List = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        if not batch:
            continue
        batch_embeddings = await embedding_func.func(batch)
        if isinstance(batch_embeddings, list):
            embeddings.extend(batch_embeddings)
        else:
            embeddings.append(batch_embeddings)
    return embeddings


@dataclass
class NodeEmbeddingConfig:
    """Configuration for FastRP and Personalized PageRank."""
    
    # FastRP parameters
    embedding_dimension: int = 256
    normalization_strength: float = -0.1
    iteration_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.5, 0.25])
    random_seed: Optional[int] = 42
    
    # Personalized PageRank parameters  
    pagerank_alpha: float = 0.85  # Damping parameter
    pagerank_max_iter: int = 100
    pagerank_tol: float = 1e-06


class NodeEmbeddingEnhancer:
    """Enhances entity embeddings using FastRP and Personalized PageRank."""
    
    def __init__(self, config: NodeEmbeddingConfig, working_dir: str):
        self.config = config
        self.working_dir = working_dir
        
        self.fastrp_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.pagerank_scores: Optional[Dict[str, float]] = None
        self.graph: Optional[nx.Graph] = None
        self._relations_cache: Optional[List[Dict]] = None  # Cache for all relations

        # File paths for persistence
        self.graph_path = os.path.join(working_dir, "node_embedding_graph.pkl")
        self.fastrp_path = os.path.join(working_dir, "fastrp_embeddings.pkl")
        self.pagerank_path = os.path.join(working_dir, "pagerank_scores.json")
        self.relations_cache_path = os.path.join(working_dir, "relations_cache.pkl")
        
        # Try to load existing data
        self._load_persisted_data()
        
    async def compute_node_embeddings(
        self, 
        entities: List[Dict],
        relations: List[Dict],
        text_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute enhanced embeddings during document insertion.
        
        Args:
            entities: List of entity dictionaries
            relations: List of relation dictionaries  
            text_embeddings: Dict mapping entity_id -> text_embedding
            
        Returns:
            Dict mapping entity_id -> enhanced_embedding
        """
        if not entities or not relations:
            logger.warning("No entities or relations to process for node embedding")
            return {}
            
        logger.info(f"Computing enhanced embeddings for {len(entities)} entities, {len(relations)} relations")
        
        # 1. Build/update NetworkX graph
        new_graph = self._build_networkx_graph(entities, relations)
        
        # Merge with existing graph if present
        if self.graph is not None:
            # Merge nodes and edges from new graph
            self.graph.add_nodes_from(new_graph.nodes(data=True))
            self.graph.add_edges_from(new_graph.edges(data=True))
            logger.info(f"Updated existing graph: now has {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
        else:
            self.graph = new_graph
        
        if len(self.graph.nodes()) == 0:
            logger.warning("Empty graph, skipping node embedding computation")
            return {}
            
        # 2. Compute FastRP embeddings
        self.fastrp_embeddings = await self._compute_fastrp_embeddings()
        
        # 3. Compute Personalized PageRank scores
        self.pagerank_scores = self._compute_pagerank_scores()
        
        # 4. Save computed data to disk for query-time reuse
        self._save_persisted_data()
        
        logger.info(f"FastRP + PageRank computation completed for {len(self.fastrp_embeddings)} entities")
        # Note: In dual-path approach, we don't return mixed embeddings
        # Instead, FastRP embeddings and PageRank scores are used separately
        return {}
        
    def _build_networkx_graph(self, entities: List[Dict], relations: List[Dict]) -> nx.Graph:
        """Build NetworkX graph from entities and relations."""
        graph = nx.Graph()
        
        # Add nodes
        for entity in entities:
            entity_id = str(entity.get('entity_name', entity.get('id', '')))
            if entity_id:
                graph.add_node(entity_id)
                
        # Add edges
        for relation in relations:
            src = str(relation.get('src_id', ''))
            tgt = str(relation.get('tgt_id', ''))
            
            if src and tgt and src in graph.nodes() and tgt in graph.nodes():
                # Use weight if available, otherwise default to 1.0
                weight = relation.get('weight', 1.0)
                graph.add_edge(src, tgt, weight=weight)
                
        logger.info(f"Built graph with {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        return graph
        
    async def _compute_fastrp_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute FastRP embeddings using proper degree normalization and self-loops."""
        if len(self.graph.nodes()) < 2:
            logger.warning("Graph too small for FastRP, using random embeddings")
            return {node: np.random.normal(0, 0.1, self.config.embedding_dimension) 
                   for node in self.graph.nodes()}
            
        try:
            # Set random seed for reproducibility
            if self.config.random_seed is not None:
                np.random.seed(self.config.random_seed)
            
            # Convert NetworkX graph to adjacency matrix
            node_list = list(self.graph.nodes())
            n_nodes = len(node_list)
            embedding_dim = self.config.embedding_dimension

            # Create adjacency matrix with edges and self-loops
            A = nx.adjacency_matrix(self.graph, nodelist=node_list, weight="weight").astype(np.float32)
            A = A + sparse.eye(A.shape[0], dtype=np.float32)  # Add self-loops
            
            # Degree normalization
            deg = np.asarray(A.sum(axis=1)).ravel()
            deg[deg == 0] = 1.0  # Avoid division by zero
            
            # Create normalized adjacency matrix S = D^{-1/2} (A+I) D^{-1/2}
            D_inv_sqrt = sparse.diags(1.0 / np.sqrt(deg))
            S = D_inv_sqrt @ A @ D_inv_sqrt
            
            # Degree-based normalization strength r
            r = self.config.normalization_strength  # e.g., -0.1

            # Initialize random projection matrix R (n x d)
            R = np.random.choice([-1.0, 1.0], size=(n_nodes, embedding_dim)).astype(np.float32)
            # Row normalize R
            R_norms = np.linalg.norm(R, axis=1, keepdims=True)
            R_norms = np.where(R_norms > 1e-9, R_norms, 1e-9)
            R = R / R_norms

            # FastRP multi-order aggregation: X = Σ(k=0 to K) w_k * D^r * S^k * R
            X = np.zeros((n_nodes, embedding_dim), dtype=np.float32)
            Z = R.copy()

            for k, weight in enumerate(self.config.iteration_weights):
                # Apply single-sided degree normalization: D^r * Z
                term = (deg ** r)[:, np.newaxis] * Z

                # Accumulate weighted term
                X += weight * term

                # Propagate for next iteration: Z = S @ Z
                if k < len(self.config.iteration_weights) - 1:  # Don't compute for last iteration
                    Z = S @ Z
                    if sparse.issparse(Z):
                        Z = Z.toarray()
            
            # Optional: final row normalization for stability
            final_norms = np.linalg.norm(X, axis=1, keepdims=True)
            final_norms = np.where(final_norms > 1e-9, final_norms, 1e-9)
            X = X / final_norms
            
            # Convert back to dictionary format
            embeddings = {}
            for idx, node in enumerate(node_list):
                embeddings[str(node)] = X[idx]
                    
            logger.info(f"FastRP embeddings computed for {len(embeddings)} nodes")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error computing FastRP embeddings: {e}")
            # Fallback to random embeddings
            return {node: np.random.normal(0, 0.1, self.config.embedding_dimension) 
                   for node in self.graph.nodes()}
                   
                   
    def _compute_pagerank_scores(self) -> Dict[str, float]:
        """Compute standard PageRank scores for all nodes."""
        try:
            pagerank_scores = nx.pagerank(
                self.graph,
                alpha=self.config.pagerank_alpha,
                max_iter=self.config.pagerank_max_iter,
                tol=self.config.pagerank_tol
            )
            logger.info(f"PageRank scores computed for {len(pagerank_scores)} nodes")
            return pagerank_scores
            
        except Exception as e:
            logger.error(f"Error computing PageRank scores: {e}")
            # Fallback to uniform scores
            num_nodes = len(self.graph.nodes())
            return {node: 1.0/num_nodes for node in self.graph.nodes()}
            
    async def get_all_relations_cached(self, knowledge_graph_inst) -> List[Dict]:
        """Get all relations with caching to avoid repeated full-graph queries."""
        if self._relations_cache is None:
            logger.debug("Relations cache miss, fetching from knowledge graph")
            self._relations_cache = await knowledge_graph_inst.get_all_relationships()
            # Optionally persist to disk
            try:
                os.makedirs(self.working_dir, exist_ok=True)
                with open(self.relations_cache_path, 'wb') as f:
                    pickle.dump(self._relations_cache, f)
                logger.debug(f"Saved {len(self._relations_cache)} relations to cache")
            except Exception as e:
                logger.debug(f"Failed to save relations cache: {e}")
        return self._relations_cache

    def invalidate_relations_cache(self):
        """Invalidate relations cache (call when graph structure changes)."""
        self._relations_cache = None
        if os.path.exists(self.relations_cache_path):
            try:
                os.remove(self.relations_cache_path)
                logger.debug("Relations cache invalidated")
            except Exception as e:
                logger.debug(f"Failed to remove relations cache file: {e}")

    def compute_personalized_pagerank(
        self,
        seed_entities: List[str],
        entity_similarities: Dict[str, float] = None,
        relation_similarities: Dict[str, float] = None,
        tau: float = 0.1,
        alpha: float = 0.3
    ) -> Dict[str, float]:
        """
        Compute personalized PageRank scores for query-time reasoning with simplified edge reweighting.

        Args:
            seed_entities: List of entity IDs to use as personalization seeds
            entity_similarities: Optional dict mapping entity_id -> similarity_score for query-aware seed weighting (Phase 1)
            relation_similarities: Optional dict mapping relation_keywords -> similarity_score for direct edge reweighting (Phase 2)
            tau: Temperature parameter for softmax weighting (lower = more focused)
            alpha: Not used in simplified implementation (kept for compatibility)

        Returns:
            Dict mapping entity_id -> personalized_pagerank_score
        """
        if not self.graph or not seed_entities:
            return {}
            
        try:
            # Create personalization vector
            personalization = {}
            valid_seeds = [e for e in seed_entities if e in self.graph.nodes()]

            if not valid_seeds:
                logger.warning("No valid seed entities found in graph")
                return {}

            # Compute query-aware seed weights if similarity scores provided
            if entity_similarities:
                seed_weights = self._compute_query_aware_weights(valid_seeds, entity_similarities, tau)
                logger.debug(f"Query-aware seed weights computed: {len(seed_weights)} weighted seeds")
            else:
                # Fallback to uniform weights
                seed_weights = {seed: 1.0/len(valid_seeds) for seed in valid_seeds}
                logger.debug(f"Using uniform seed weights for {len(valid_seeds)} seeds")

            # Build personalization vector for all nodes
            for node in self.graph.nodes():
                personalization[node] = seed_weights.get(node, 0.0)

            # Ensure normalization (should already be normalized, but double-check)
            total = sum(personalization.values())
            if total > 0:
                personalization = {k: v/total for k, v in personalization.items()}

            # Phase 2: Apply direct edge reweighting if relation similarities provided
            if relation_similarities and alpha > 0.0:
                # Temporarily set edge weights based on hl_keywords similarity
                for u, v, edge_data in self.graph.edges(data=True):
                    edge_keywords = edge_data.get('keywords', '')
                    if edge_keywords in relation_similarities:
                        # Use similarity as weight directly
                        similarity = relation_similarities[edge_keywords]
                        self.graph[u][v]['temp_weight'] = max(0.1, similarity)  # Min weight 0.1
                    else:
                        self.graph[u][v]['temp_weight'] = 0.1  # Default low weight

                # Compute PageRank with temporary weights
                ppr_scores = nx.pagerank(
                    self.graph,
                    alpha=self.config.pagerank_alpha,
                    personalization=personalization,
                    max_iter=self.config.pagerank_max_iter,
                    tol=self.config.pagerank_tol,
                    weight='temp_weight'
                )

                # Clean up temporary weights
                for u, v in self.graph.edges():
                    if 'temp_weight' in self.graph[u][v]:
                        del self.graph[u][v]['temp_weight']

                logger.debug(f"Applied direct edge reweighting with {len(relation_similarities)} relation similarities")
            else:
                # Compute PageRank with original weights
                ppr_scores = nx.pagerank(
                    self.graph,
                    alpha=self.config.pagerank_alpha,
                    personalization=personalization,
                    max_iter=self.config.pagerank_max_iter,
                    tol=self.config.pagerank_tol,
                    weight='weight'
                )
            
            logger.info(f"Personalized PageRank computed for {len(ppr_scores)} nodes with {len(valid_seeds)} seeds")
            return ppr_scores
            
        except Exception as e:
            logger.error(f"Error computing Personalized PageRank: {e}")
            return {}
        
    def extract_reasoning_paths(
        self, 
        source_entities: List[str], 
        target_entities: List[str],
        max_paths: int = 5,
        max_length: int = 4
    ) -> List[List[str]]:
        """Extract reasoning paths between source and target entities."""
        if not self.graph:
            return []
            
        paths = []
        
        for source in source_entities:
            for target in target_entities:
                if source == target:
                    continue
                    
                if source in self.graph.nodes() and target in self.graph.nodes():
                    try:
                        # Find shortest paths
                        all_paths = list(nx.all_simple_paths(
                            self.graph, source, target, cutoff=max_length
                        ))
                        
                        # Sort by length and take top paths
                        all_paths.sort(key=len)
                        paths.extend(all_paths[:max_paths])
                        
                    except nx.NetworkXNoPath:
                        continue
                        
        # Remove duplicates and sort by length  
        unique_paths = []
        for path in paths:
            if path not in unique_paths:
                unique_paths.append(path)
                
        unique_paths.sort(key=len)
        return unique_paths[:max_paths]

    async def compute_query_aware_ppr(
        self,
        seed_nodes: List[Dict],
        top_ppr_nodes: int,
        knowledge_graph_inst,
        global_config: Dict[str, Any],
        ll_keywords: str = "",
        hl_keywords: str = ""
    ) -> List[Dict]:
        """Compute top PPR entities with query-aware weighting."""
        if top_ppr_nodes <= 0:
            return []

        seed_entity_names = [
            node.get("entity_name", "")
            for node in seed_nodes
            if node.get("entity_name")
        ]
        if not seed_entity_names:
            logger.warning("No seed entities for PPR analysis")
            return []

        logger.info(f"PPR analysis from {len(seed_entity_names)} seed entities")

        entity_similarities: Dict[str, float] = {}
        if ll_keywords.strip():
            try:
                entities_vdb = global_config.get("entities_vdb")
                if entities_vdb and hasattr(entities_vdb, "embedding_func"):
                    embedding_func = entities_vdb.embedding_func
                    if embedding_func and embedding_func.func:
                        query_embedding = await embedding_func.func([ll_keywords])
                        query_vec = np.array(query_embedding[0])

                        entity_strs: List[str] = []
                        entity_order: List[str] = []

                        for seed_entity in seed_entity_names:
                            seed_info = next(
                                (node for node in seed_nodes if node.get("entity_name") == seed_entity),
                                None
                            )
                            if seed_info and seed_info.get("entity_description"):
                                entity_str = f"{seed_entity} {seed_info['entity_description']}"
                                entity_strs.append(entity_str)
                                entity_order.append(seed_entity)

                        if entity_strs:
                            batch_fallback = resolve_positive_int(global_config.get("llm_model_max_async", 4), 4)
                            batch_size = resolve_positive_int(
                                global_config.get("embedding_batch_size"),
                                batch_fallback
                            )
                            entity_embeddings = await compute_embeddings_in_batches(
                                entity_strs, embedding_func, batch_size
                            )

                            for idx, entity_embedding in enumerate(entity_embeddings):
                                entity_vec = np.array(entity_embedding)
                                similarity = cosine_similarity(query_vec, entity_vec)
                                entity_name = entity_order[idx]
                                entity_similarities[entity_name] = similarity

                        logger.debug(f"Computed query-aware similarities for {len(entity_similarities)} seed entities")

            except Exception as exc:
                logger.warning(f"Could not compute query-aware similarities: {exc}")

        relation_similarities: Dict[str, float] = {}
        if hl_keywords.strip():
            try:
                relationships_vdb = global_config.get("relationships_vdb")
                if relationships_vdb and hasattr(relationships_vdb, "embedding_func"):
                    embedding_func = relationships_vdb.embedding_func
                    if embedding_func and embedding_func.func:
                        hl_query_embedding = await embedding_func.func([hl_keywords])
                        hl_query_vec = np.array(hl_query_embedding[0])

                        all_relations = await self.get_all_relations_cached(knowledge_graph_inst)

                        rel_strs: List[str] = []
                        rel_keywords_order: List[str] = []

                        for relation in all_relations:
                            rel_keywords = relation.get("keywords", "")
                            if rel_keywords.strip():
                                rel_str = (
                                    f"{rel_keywords}\t{relation.get('source_id', '')}\n"
                                    f"{relation.get('target_id', '')}\n{relation.get('description', '')}"
                                )
                                rel_strs.append(rel_str)
                                rel_keywords_order.append(rel_keywords)

                        if rel_strs:
                            batch_fallback = resolve_positive_int(global_config.get("llm_model_max_async", 4), 4)
                            batch_size = resolve_positive_int(
                                global_config.get("embedding_batch_size"),
                                batch_fallback
                            )
                            rel_embeddings = await compute_embeddings_in_batches(
                                rel_strs, embedding_func, batch_size
                            )

                            for idx, rel_embedding in enumerate(rel_embeddings):
                                rel_vec = np.array(rel_embedding)
                                similarity = cosine_similarity(hl_query_vec, rel_vec)
                                rel_keywords = rel_keywords_order[idx]
                                relation_similarities[rel_keywords] = similarity

                        logger.debug(f"Computed query-aware relation similarities for {len(relation_similarities)} relations")

            except Exception as exc:
                logger.warning(f"Could not compute relation similarities: {exc}")

        ppr_scores = self.compute_personalized_pagerank(
            seed_entity_names,
            entity_similarities if entity_similarities else None,
            relation_similarities if relation_similarities else None,
            tau=0.1,
            alpha=0.3
        )

        if not ppr_scores:
            logger.warning("No Personalized PageRank scores computed")
            return []

        all_entities = [
            (entity_name, score)
            for entity_name, score in ppr_scores.items()
            if entity_name not in seed_entity_names
        ]
        if not all_entities:
            logger.warning("No entities found for PPR ranking")
            return []

        all_entities.sort(key=lambda x: x[1], reverse=True)
        top_entities = all_entities[:top_ppr_nodes]

        logger.info(f"PPR selected top {len(top_entities)} entities from {len(all_entities)} total entities")

        top_entity_names = [entity_name for entity_name, _ in top_entities]
        nodes_dict = await knowledge_graph_inst.get_nodes_batch(top_entity_names)

        ppr_entities: List[Dict] = []
        for entity_name, ppr_score in top_entities:
            entity_data = (nodes_dict or {}).get(entity_name)
            if entity_data:
                entity_copy = dict(entity_data)
                entity_copy.update({
                    "entity_name": entity_name,
                    "pagerank_score": ppr_score,
                    "discovery_method": "ppr_analysis",
                    "rank": entity_data.get("degree", 0),
                })
                ppr_entities.append(entity_copy)

        logger.info(
            "PPR analysis completed: %d entities with avg PageRank: %.4f",
            len(ppr_entities),
            (sum(e["pagerank_score"] for e in ppr_entities) / len(ppr_entities)) if ppr_entities else 0.0,
        )
        return ppr_entities

    async def compute_adaptive_fastrp(
        self,
        seed_nodes: List[Dict],
        top_fastrp_nodes: int,
        knowledge_graph_inst
    ) -> List[Dict]:
        """Compute top FastRP entities using vectorized similarities."""
        if top_fastrp_nodes <= 0:
            return []

        seed_entity_names = [
            node.get("entity_name", "")
            for node in seed_nodes
            if node.get("entity_name")
        ]
        if not seed_entity_names:
            logger.warning("No seed entities for FastRP analysis")
            return []

        if not hasattr(self, "fastrp_embeddings") or not self.fastrp_embeddings:
            logger.warning("FastRP embeddings not available for FastRP analysis")
            return []

        logger.info(f"FastRP analysis from {len(seed_entity_names)} seed entities")

        candidate_scores = compute_adaptive_fastrp_batch(
            seed_entity_names,
            self,
        )

        if not candidate_scores:
            logger.warning("No valid FastRP candidates found")
            return []

        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )
        top_candidates = sorted_candidates[:top_fastrp_nodes]

        top_entity_names = [entity_name for entity_name, _ in top_candidates]
        nodes_dict = await knowledge_graph_inst.get_nodes_batch(top_entity_names)

        top_smart_entities: List[Dict] = []
        for entity_name, similarity in top_candidates:
            entity_data = (nodes_dict or {}).get(entity_name)
            if entity_data:
                entity_copy = dict(entity_data)
                entity_copy.update({
                    "entity_name": entity_name,
                    "adaptive_fastrp_similarity": similarity,
                    "discovery_method": "fastrp_analysis",
                    "rank": entity_data.get("degree", 0),
                })
                top_smart_entities.append(entity_copy)

        logger.info(
            "Adaptive FastRP selected top %d entities with avg similarity: %.4f",
            len(top_smart_entities),
            (sum(e["adaptive_fastrp_similarity"] for e in top_smart_entities) / len(top_smart_entities))
            if top_smart_entities else 0.0,
        )

        return top_smart_entities
    
    def _save_persisted_data(self):
        """Save graph structure and embeddings to disk for query-time reuse."""
        try:
            # Ensure working directory exists
            os.makedirs(self.working_dir, exist_ok=True)
            
            # Save graph structure
            if self.graph:
                with open(self.graph_path, 'wb') as f:
                    pickle.dump(self.graph, f)
                logger.info(f"Saved graph with {len(self.graph.nodes())} nodes to {self.graph_path}")
            
            # Save FastRP embeddings
            if self.fastrp_embeddings:
                with open(self.fastrp_path, 'wb') as f:
                    pickle.dump(self.fastrp_embeddings, f)
                logger.info(f"Saved FastRP embeddings for {len(self.fastrp_embeddings)} entities")
            
            # Save PageRank scores (JSON for readability)
            if self.pagerank_scores:
                with open(self.pagerank_path, 'w') as f:
                    json.dump(self.pagerank_scores, f, indent=2)
                logger.info(f"Saved PageRank scores for {len(self.pagerank_scores)} entities")
                
        except Exception as e:
            logger.error(f"Error saving node embedding data: {e}")
    
    def _load_persisted_data(self):
        """Load previously saved graph structure and embeddings."""
        try:
            # Load graph structure
            if os.path.exists(self.graph_path):
                with open(self.graph_path, 'rb') as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded graph with {len(self.graph.nodes())} nodes from {self.graph_path}")

            # Load FastRP embeddings
            if os.path.exists(self.fastrp_path):
                with open(self.fastrp_path, 'rb') as f:
                    self.fastrp_embeddings = pickle.load(f)
                logger.info(f"Loaded FastRP embeddings for {len(self.fastrp_embeddings)} entities")

            # Load PageRank scores
            if os.path.exists(self.pagerank_path):
                with open(self.pagerank_path, 'r') as f:
                    self.pagerank_scores = json.load(f)
                logger.info(f"Loaded PageRank scores for {len(self.pagerank_scores)} entities")

            # Load Relations cache
            if os.path.exists(self.relations_cache_path):
                with open(self.relations_cache_path, 'rb') as f:
                    self._relations_cache = pickle.load(f)
                logger.info(f"Loaded relations cache with {len(self._relations_cache)} relations")

        except Exception as e:
            logger.warning(f"Could not load persisted node embedding data: {e}")
            # Reset to empty state
            self.graph = None
            self.fastrp_embeddings = None
            self.pagerank_scores = None
            self._relations_cache = None

    def _compute_query_aware_weights(
        self,
        valid_seeds: List[str],
        entity_similarities: Dict[str, float],
        tau: float = 0.01
    ) -> Dict[str, float]:
        """
        Compute query-aware seed weights using softmax on similarity scores.

        Args:
            valid_seeds: List of valid seed entities in the graph
            entity_similarities: Dict mapping entity_id -> similarity_score
            tau: Temperature parameter for softmax (lower = more focused)

        Returns:
            Dict mapping seed_entity -> normalized_weight
        """
        import numpy as np

        # Extract similarity scores for valid seeds
        scores = []
        seeds_with_scores = []

        for seed in valid_seeds:
            if seed in entity_similarities:
                scores.append(entity_similarities[seed])
                seeds_with_scores.append(seed)
            else:
                # Fallback score for seeds without similarity data
                scores.append(0.0)
                seeds_with_scores.append(seed)

        if not seeds_with_scores:
            logger.warning("No seeds with similarity scores, falling back to uniform weights")
            return {seed: 1.0/len(valid_seeds) for seed in valid_seeds}

        # Apply softmax with temperature
        scores = np.array(scores)
        exp_scores = np.exp(scores / tau)
        weights = exp_scores / np.sum(exp_scores)

        # Create weight dictionary
        seed_weights = {seed: float(weight) for seed, weight in zip(seeds_with_scores, weights)}

        logger.debug(f"Query-aware weights: max={max(weights):.3f}, min={min(weights):.3f}, "
                    f"std={np.std(weights):.3f}")

        return seed_weights

