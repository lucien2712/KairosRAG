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

logger = logging.getLogger(__name__)


@dataclass
class NodeEmbeddingConfig:
    """Configuration for FastRP and Personalized PageRank."""
    
    # FastRP parameters
    embedding_dimension: int = 128
    normalization_strength: float = -0.1
    iteration_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
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
        
        # File paths for persistence
        self.graph_path = os.path.join(working_dir, "node_embedding_graph.pkl")
        self.fastrp_path = os.path.join(working_dir, "fastrp_embeddings.pkl")
        self.pagerank_path = os.path.join(working_dir, "pagerank_scores.json")
        
        # Try to load existing data
        self._load_persisted_data()
        
    async def compute_embeddings_during_insert(
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
                # Add edge weight based on relation strength (optional)
                weight = 1.0  # Can be enhanced with relation confidence scores
                graph.add_edge(src, tgt, weight=weight)
                
        logger.info(f"Built graph with {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        return graph
        
    async def _compute_fastrp_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute FastRP embeddings for all nodes using Fast Random Projection."""
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
            node_to_idx = {node: idx for idx, node in enumerate(node_list)}
            n_nodes = len(node_list)
            
            # Create adjacency matrix
            adj_matrix = nx.adjacency_matrix(self.graph, nodelist=node_list)
            adj_matrix = adj_matrix.astype(np.float32)
            
            # Initialize random projection matrix
            # Using sparse random matrix for efficiency
            embedding_dim = self.config.embedding_dimension
            random_matrix = self._generate_sparse_random_matrix(n_nodes, embedding_dim)
            
            # FastRP algorithm: iterative matrix multiplication
            embeddings_matrix = random_matrix.copy()
            
            for i, weight in enumerate(self.config.iteration_weights):
                if i == 0:
                    # First iteration: use random matrix
                    continue
                else:
                    # Subsequent iterations: propagate through adjacency matrix
                    embeddings_matrix = adj_matrix @ embeddings_matrix
                    
                # Apply normalization
                if self.config.normalization_strength != 0:
                    row_norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
                    row_norms = np.power(row_norms, self.config.normalization_strength)
                    embeddings_matrix = embeddings_matrix / (row_norms + 1e-10)
                
                # Weight the iteration
                if i == 0:
                    final_embeddings = weight * embeddings_matrix
                else:
                    final_embeddings += weight * embeddings_matrix
            
            # Convert back to dictionary format
            embeddings = {}
            for idx, node in enumerate(node_list):
                embeddings[str(node)] = final_embeddings[idx]
                    
            logger.info(f"FastRP embeddings computed for {len(embeddings)} nodes")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error computing FastRP embeddings: {e}")
            # Fallback to random embeddings
            return {node: np.random.normal(0, 0.1, self.config.embedding_dimension) 
                   for node in self.graph.nodes()}
                   
    def _generate_sparse_random_matrix(self, n_nodes: int, embedding_dim: int) -> np.ndarray:
        """Generate sparse random projection matrix for FastRP."""
        # Create sparse random matrix following FastRP paper
        # Use {-1, 0, 1} values with specific probabilities
        matrix = np.zeros((n_nodes, embedding_dim), dtype=np.float32)
        
        # Fill matrix with sparse random values
        # Probability: 1/6 for -1, 2/3 for 0, 1/6 for 1
        for i in range(n_nodes):
            for j in range(embedding_dim):
                rand_val = np.random.random()
                if rand_val < 1/6:
                    matrix[i, j] = -1.0
                elif rand_val > 5/6:
                    matrix[i, j] = 1.0
                # else: remains 0.0
                
        return matrix
                   
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
            
    def compute_personalized_pagerank(self, seed_entities: List[str]) -> Dict[str, float]:
        """
        Compute personalized PageRank scores for query-time reasoning.
        
        Args:
            seed_entities: List of entity IDs to use as personalization seeds
            
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
                
            for node in self.graph.nodes():
                personalization[node] = 1.0 if node in valid_seeds else 0.0
                
            # Normalize personalization vector
            total = sum(personalization.values())
            if total > 0:
                personalization = {k: v/total for k, v in personalization.items()}
            
            # Compute personalized PageRank
            ppr_scores = nx.pagerank(
                self.graph,
                alpha=self.config.pagerank_alpha,
                personalization=personalization,
                max_iter=self.config.pagerank_max_iter,
                tol=self.config.pagerank_tol
            )
            
            logger.info(f"Personalized PageRank computed for {len(ppr_scores)} nodes with {len(valid_seeds)} seeds")
            return ppr_scores
            
        except Exception as e:
            logger.error(f"Error computing Personalized PageRank: {e}")
            return {}
            
    # DEPRECATED: Mixed embeddings are not used in the current dual-path approach
    # This method is kept for potential future use but is not currently called
    def _combine_embeddings_deprecated(self, text_embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        DEPRECATED: Combine text embeddings, FastRP embeddings, and PageRank scores.
        
        This method created mixed embeddings but is not used in the current dual-path approach.
        Instead, FastRP embeddings and PageRank scores are used separately in structural path expansion.
        """
        # Implementation kept for reference but not used
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
                
        except Exception as e:
            logger.warning(f"Could not load persisted node embedding data: {e}")
            # Reset to empty state
            self.graph = None
            self.fastrp_embeddings = None
            self.pagerank_scores = None