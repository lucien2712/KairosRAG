import numpy as np
import json
import asyncio
import os
import base64
import zlib
import hashlib
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from tqdm.asyncio import tqdm as atqdm
from .utils import logger
from .prompt import PROMPTS


async def single_pass_agentic_merging(rag_instance, threshold: float = 0.8, langchain_client=None, token_tracker=None, tool_llm_model_kwargs: dict = None) -> dict:
    """
    Asynchronously perform intelligent entity merging using vector similarity and LLM decision making.

    Args:
        rag_instance: LightRAG instance containing all necessary components
        threshold: Cosine similarity threshold for candidate pair filtering (default: 0.8)
        langchain_client: Optional shared LangChain ChatOpenAI client (if None, creates new one)
        token_tracker: Optional TokenTracker instance for tracking token usage

    Returns:
        dict: Statistics about the merging process
    """
    import time as time_module
    from .operate import _handle_entity_relation_summary
    from .utils import compute_mdhash_id
    from .kg.shared_storage import get_entity_lock_manager

    start_time = time_module.time()
    print(f"Starting agentic entity merging with threshold={threshold}")

    # Load entity comparison cache
    cache_file_path = os.path.join(rag_instance.working_dir, "entity_merge_cache.json")
    entity_pair_cache = {}
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                loaded_cache = json.load(f)

            # Migrate old cache format to new format if needed
            # Old format: key = "hash1_hash2" (based on description hashes)
            # New format: key = "entityA_entityB" (based on entity IDs)
            migrated_cache = {}
            old_format_count = 0

            for key, value in loaded_cache.items():
                # Check if this is new format (has entity_a_id and entity_b_id in value)
                if "entity_a_id" in value and "entity_b_id" in value:
                    # New format - check if key is entity-based or hash-based
                    entity_a_id = value["entity_a_id"]
                    entity_b_id = value["entity_b_id"]
                    entity_pair_id = f"{min(entity_a_id, entity_b_id)}||{max(entity_a_id, entity_b_id)}"

                    if key == entity_pair_id:
                        # Already in new format
                        migrated_cache[key] = value
                    else:
                        # Old hash-based key with entity IDs in value - migrate
                        old_format_count += 1
                        migrated_cache[entity_pair_id] = value
                else:
                    # Very old format without entity IDs - cannot migrate, skip
                    old_format_count += 1
                    logger.warning(f"Skipping old cache entry without entity IDs: {key}")

            entity_pair_cache = migrated_cache
            if old_format_count > 0:
                print(f"Migrated {old_format_count} cache entries from old format to new format")
            print(f"Loaded {len(entity_pair_cache)} cached entity pair comparisons")
        except Exception as e:
            print(f"Failed to load entity pair cache: {e}")
            entity_pair_cache = {}

    # Create LangChain callback for token tracking
    from langchain_core.callbacks.base import BaseCallbackHandler

    class TokenTrackingCallback(BaseCallbackHandler):
        """Callback to track token usage from LangChain LLM calls"""
        def __init__(self, token_tracker):
            self.token_tracker = token_tracker

        def on_llm_end(self, response, **kwargs):
            """Called when LLM finishes"""
            if self.token_tracker and hasattr(response, 'llm_output'):
                llm_output = response.llm_output
                if llm_output and 'token_usage' in llm_output:
                    usage = llm_output['token_usage']
                    token_counts = {
                        'prompt_tokens': usage.get('prompt_tokens', 0),
                        'completion_tokens': usage.get('completion_tokens', 0),
                        'total_tokens': usage.get('total_tokens', 0),
                    }
                    self.token_tracker.add_usage(token_counts)

    # Initialize callback
    callbacks = []
    if token_tracker:
        callbacks = [TokenTrackingCallback(token_tracker)]

    # Initialize LLM using rag_instance.tool_llm_model_name or use shared client
    if langchain_client is None:
        llm_params = {
            "model": rag_instance.tool_llm_model_name,
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
            "callbacks": callbacks  # Add callbacks
        }

        # Add tool_llm_model_kwargs if provided (e.g., {"reasoning_effort": "minimal"} for GPT-5)
        if tool_llm_model_kwargs:
            llm_params["model_kwargs"] = tool_llm_model_kwargs

        llm = ChatOpenAI(**llm_params)
        print(f"LLM initialized: {rag_instance.tool_llm_model_name}")
    else:
        llm = langchain_client
        if callbacks:
            llm.callbacks = callbacks
        print(f"Using shared LangChain client: {rag_instance.tool_llm_model_name}")

    # Build global_config for summarization
    global_config = {
        "tokenizer": rag_instance.tokenizer,
        "summary_context_size": rag_instance.summary_context_size,
        "summary_max_tokens": rag_instance.summary_max_tokens,
        "force_llm_summary_on_merge": rag_instance.force_llm_summary_on_merge,
        "summary_length_recommended": rag_instance.summary_length_recommended,
        "llm_model_func": rag_instance.llm_model_func,
        "addon_params": {
            "language": "English"  # Default language, can be made configurable
        }
    }

    # Statistics tracking for summarization
    summarization_stats = {
        "entities_summarized": 0,
        "relations_summarized": 0,
        "entity_llm_calls": 0,
        "relation_llm_calls": 0
    }

    # Get entity lock manager
    lock_mgr = get_entity_lock_manager()

    # Define the merge tool for LLM
    @tool
    async def merge_entities_tool(a_entity_id: str, b_entity_id: str) -> str:
        """
        If two entities represent the same real-world entity, merge b_entity_id into a_entity_id.
        """
        # Acquire locks for both entities to prevent concurrent modifications
        # Locks are acquired in sorted order to prevent deadlock
        async with await lock_mgr.acquire_locks(a_entity_id, b_entity_id):
            try:
                # print(f"TOOL CALLED: Merging {b_entity_id} -> {a_entity_id}")

                # CRITICAL: Validate entity existence after acquiring locks
                # This prevents TOCTOU race conditions where entities are deleted
                # between evaluation and execution
                entity_a = await rag_instance.chunk_entity_relation_graph.get_node(a_entity_id)
                entity_b = await rag_instance.chunk_entity_relation_graph.get_node(b_entity_id)

                if entity_a is None or entity_b is None:
                    missing = []
                    if entity_a is None:
                        missing.append(a_entity_id)
                    if entity_b is None:
                        missing.append(b_entity_id)
                    skip_msg = f"Merge skipped: Entities no longer exist: {', '.join(missing)}"
                    print(f"SKIPPED: {b_entity_id} -> {a_entity_id} (entities deleted by concurrent operation)")
                    return skip_msg

                # Step 1: Perform merge with concatenation
                await rag_instance.amerge_entities(
                    source_entities=[a_entity_id, b_entity_id],
                    target_entity=a_entity_id,
                    merge_strategy={
                        "created_at": "keep_last",
                        "description": "concatenate",
                        "entity_type": "keep_first",
                        "source_id": "join_unique",
                        "file_path": "join_unique",
                    },
                )

                # Step 2: Summarize merged entity description using same logic as normal insertion
                merged_node = await rag_instance.chunk_entity_relation_graph.get_node(a_entity_id)
                if merged_node and 'description' in merged_node and merged_node['description']:
                    description_list = merged_node['description'].split('\n\n')

                    # Use the same summarization logic as normal entity insertion
                    summarized_desc, llm_used = await _handle_entity_relation_summary(
                        description_type="Entity",
                        entity_or_relation_name=a_entity_id,
                        description_list=description_list,
                        global_config=global_config,
                        llm_response_cache=rag_instance.llm_response_cache,
                        seperator="\n\n"
                    )

                    # Update entity if description changed
                    if summarized_desc != merged_node['description']:
                        # Update graph storage
                        await rag_instance.chunk_entity_relation_graph.upsert_node(
                            a_entity_id,
                            {'description': summarized_desc}
                        )

                        # Update vector database
                        entity_vdb_id = compute_mdhash_id(a_entity_id, prefix="ent-")
                        entity_content = f"{a_entity_id}\n{summarized_desc}"

                        await rag_instance.entities_vdb.upsert(
                            {entity_vdb_id: {
                                "content": entity_content,
                                "entity_name": a_entity_id
                            }}
                        )

                        summarization_stats["entities_summarized"] += 1
                        if llm_used:
                            summarization_stats["entity_llm_calls"] += 1

                # Step 3: Summarize all related relationship descriptions
                # Get all edges connected to the merged entity
                edge_tuples = await rag_instance.chunk_entity_relation_graph.get_node_edges(a_entity_id)

                if edge_tuples:
                    # Batch fetch all edge data in parallel
                    edge_data_tasks = [
                        rag_instance.chunk_entity_relation_graph.get_edge(src_id, tgt_id)
                        for src_id, tgt_id in edge_tuples
                    ]
                    all_edges = await asyncio.gather(*edge_data_tasks)

                    # Prepare summarization tasks
                    async def process_relationship(edge, src_id, tgt_id):
                        if not edge or 'description' not in edge:
                            return None

                        description = edge['description']
                        description_list = description.split('\n\n')

                        # Use same summarization logic for relationships
                        summarized_desc, llm_used = await _handle_entity_relation_summary(
                            description_type="Relationship",
                            entity_or_relation_name=f"{src_id} -> {tgt_id}",
                            description_list=description_list,
                            global_config=global_config,
                            llm_response_cache=rag_instance.llm_response_cache,
                            seperator="\n\n"
                        )

                        # Only return update data if description changed
                        if summarized_desc != description:
                            return {
                                'src_id': src_id,
                                'tgt_id': tgt_id,
                                'edge': edge,
                                'summarized_desc': summarized_desc,
                                'llm_used': llm_used
                            }
                        return None

                    # Process all relationships in parallel
                    relationship_tasks = [
                        process_relationship(edge, src_id, tgt_id)
                        for edge, (src_id, tgt_id) in zip(all_edges, edge_tuples)
                    ]
                    relationship_results = await asyncio.gather(*relationship_tasks)

                    # Filter out None results and batch update
                    updates_to_apply = [r for r in relationship_results if r is not None]

                    if updates_to_apply:
                        # Batch update graph storage
                        graph_update_tasks = [
                            rag_instance.chunk_entity_relation_graph.upsert_edge(
                                update['src_id'],
                                update['tgt_id'],
                                {**update['edge'], 'description': update['summarized_desc']}
                            )
                            for update in updates_to_apply
                        ]

                        # Batch update vector database
                        vdb_update_tasks = [
                            rag_instance.relationships_vdb.upsert(
                                {compute_mdhash_id(update['src_id'] + update['tgt_id'], prefix="rel-"): {
                                    "src_id": update['src_id'],
                                    "tgt_id": update['tgt_id'],
                                    "content": f"{update['edge'].get('keywords', '')}\\t{update['src_id']}\\n{update['tgt_id']}\\n{update['summarized_desc']}"
                                }}
                            )
                            for update in updates_to_apply
                        ]

                        # Execute all updates in parallel
                        await asyncio.gather(*graph_update_tasks, *vdb_update_tasks)

                        # Update statistics
                        summarization_stats["relations_summarized"] += len(updates_to_apply)
                        summarization_stats["relation_llm_calls"] += sum(
                            1 for update in updates_to_apply if update['llm_used']
                        )

                # print(f"TOOL COMPLETED: Merge {b_entity_id} -> {a_entity_id}")
                return f"Merge successfully: {a_entity_id} <- {b_entity_id}"
            except Exception as e:
                print(f"TOOL FAILED: Merge {b_entity_id} -> {a_entity_id}, Error: {str(e)}")
                return f"Merge failed: {str(e)}"

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools([merge_entities_tool])

    # Step 1: Get all entities from graph storage and their vectors from entities_vdb
    print("Loading entities from graph storage...")
    all_nodes = await rag_instance.chunk_entity_relation_graph.get_all_nodes()
    print(f"Found {len(all_nodes)} nodes in graph storage")

    if not all_nodes:
        print("No entities found in graph storage")
        return {
            "total_entities": 0,
            "candidate_pairs": 0,
            "llm_evaluated_pairs": 0,
            "merged_pairs": 0,
            "remaining_entities": 0,
            "processing_time": time_module.time() - start_time,
            "similarity_threshold": threshold
        }

    # Generate vector database IDs for all entities and batch get vector data
    entity_vdb_ids = []
    entity_id_to_node = {}  # Map entity_id to node data

    for node in all_nodes:
        entity_id = node.get('entity_id')
        if entity_id:
            vdb_id = compute_mdhash_id(entity_id, prefix="ent-")
            entity_vdb_ids.append(vdb_id)
            entity_id_to_node[entity_id] = node

    # Access vector data directly from the storage client to get the vector field
    print(f"Batch loading vectors for {len(entity_vdb_ids)} entities...")
    entities_vdb_data = {}
    if entity_vdb_ids:
        # Access the internal client to get raw data including vectors
        client = await rag_instance.entities_vdb._get_client()
        vdb_results = client.get(entity_vdb_ids)
        print(f"Retrieved {len(vdb_results)} vector results")

        # Process results - create mapping from entity_id to vector data
        for vdb_data in vdb_results:
            if vdb_data and 'content' in vdb_data:
                # Extract entity_id from content (format: "entity_name\ndescription")
                content = vdb_data['content']
                entity_name = content.split('\n')[0] if '\n' in content else content
                entities_vdb_data[entity_name] = vdb_data

    print(f"Successfully loaded vectors for {len(entities_vdb_data)} entities")

    if not entities_vdb_data:
        print("No vector data found for entities")
        return {
            "total_entities": 0,
            "candidate_pairs": 0,
            "llm_evaluated_pairs": 0,
            "merged_pairs": 0,
            "remaining_entities": 0,
            "processing_time": time_module.time() - start_time,
            "similarity_threshold": threshold
        }

    # Step 2: Extract entity information and embeddings
    print(f"Extracting entity information and embeddings...")

    # Helper function for parallel vector decoding
    def decode_single_vector(vector_encoded: str) -> np.ndarray:
        """Decode base64 + zlib compressed vector (CPU-intensive operation)."""
        try:
            compressed_vector = base64.b64decode(vector_encoded)
            vector_bytes = zlib.decompress(compressed_vector)
            return np.frombuffer(vector_bytes, dtype=np.float16).astype(np.float32)
        except Exception as e:
            return None

    # Prepare data for parallel processing
    entity_names = []
    encoded_vectors = []
    descriptions = []

    for entity_name, entity_data in entities_vdb_data.items():
        if 'content' in entity_data and 'vector' in entity_data:
            node_data = entity_id_to_node.get(entity_name, {})
            description = node_data.get('description', '')

            entity_names.append(entity_name)
            encoded_vectors.append(entity_data['vector'])
            descriptions.append(description)

    # Parallel vector decoding using thread pool (asyncio-compatible)
    # Note: Using ThreadPoolExecutor since base64/zlib releases GIL
    from concurrent.futures import ThreadPoolExecutor
    import asyncio

    async def decode_vectors_parallel(encoded_vectors):
        """Decode all vectors in parallel using thread pool."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=min(32, len(encoded_vectors))) as executor:
            # Submit all decoding tasks
            futures = [
                loop.run_in_executor(executor, decode_single_vector, vec)
                for vec in encoded_vectors
            ]
            # Wait for all to complete
            return await asyncio.gather(*futures)

    # Decode all vectors in parallel
    if encoded_vectors:
        print(f"Decoding {len(encoded_vectors)} vectors in parallel...")
        decoded_vectors = await decode_vectors_parallel(encoded_vectors)
    else:
        decoded_vectors = []

    # Build entities and embeddings lists (keep as numpy arrays, not lists)
    entities = []
    entity_embeddings = []

    for entity_name, description, vector_array in zip(entity_names, descriptions, decoded_vectors):
        if vector_array is not None and len(vector_array) > 0:
            entities.append({
                'entity_id': entity_name,
                'description': description,
            })
            entity_embeddings.append(vector_array)  # Keep as numpy array (optimization #4)
        elif vector_array is None:
            print(f"    - Error decoding vector for: {entity_name}")
        else:
            print(f"    - Empty vector for: {entity_name}")

    print(f"Successfully extracted {len(entities)} entities with embeddings")

    total_entities = len(entities)
    if total_entities < 2:
        return {
            "total_entities": total_entities,
            "candidate_pairs": 0,
            "llm_evaluated_pairs": 0,
            "merged_pairs": 0,
            "remaining_entities": total_entities,
            "processing_time": time_module.time() - start_time,
            "similarity_threshold": threshold
        }

    # Step 3: Calculate similarity matrix using numpy (efficient!)
    print(f"Calculating similarity matrix for {len(entities)} entities...")

    if not entity_embeddings:
        print("No valid embeddings found")
        return {
            "total_entities": len(all_nodes),
            "candidate_pairs": 0,
            "llm_evaluated_pairs": 0,
            "merged_pairs": 0,
            "remaining_entities": len(all_nodes),
            "processing_time": time_module.time() - start_time,
            "similarity_threshold": threshold
        }

    # Check and standardize vector dimensions (vectorized approach)
    vector_dimensions = np.array([len(emb) for emb in entity_embeddings])
    max_dim = int(vector_dimensions.max())
    min_dim = int(vector_dimensions.min())

    # Warn if inconsistent dimensions detected
    unique_dims = np.unique(vector_dimensions)
    if len(unique_dims) > 1:
        logger.warning(
            f"Inconsistent vector dimensions detected: {unique_dims.tolist()}. "
            f"This may indicate different embedding models were used. "
            f"Padding/truncating to {max_dim} dimensions."
        )

    # Vectorized padding: pre-allocate array and fill (much faster than loop + list concatenation)
    try:
        embeddings = np.zeros((len(entity_embeddings), max_dim), dtype=np.float32)
        for i, emb in enumerate(entity_embeddings):
            emb_len = len(emb)
            if emb_len <= max_dim:
                embeddings[i, :emb_len] = emb  # Pad with zeros automatically
            else:
                embeddings[i, :] = emb[:max_dim]  # Truncate if longer

        # print(f"Created embeddings array: {embeddings.shape}")

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        # Calculate similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)
        # print(f"Similarity matrix calculated: {similarity_matrix.shape}")
    except Exception as e:
        print(f"Error creating similarity matrix: {e}")
        return {
            "total_entities": len(all_nodes),
            "candidate_pairs": 0,
            "llm_evaluated_pairs": 0,
            "merged_pairs": 0,
            "remaining_entities": len(all_nodes),
            "processing_time": time_module.time() - start_time,
            "similarity_threshold": threshold,
            "error": str(e)
        }

    # Step 4: Find candidate pairs above threshold (vectorized for performance)
    print(f"Filtering candidate pairs with similarity >= {threshold}...")

    # Vectorized extraction using numpy - much faster for large entity sets
    # Get upper triangle indices (i < j) to avoid duplicate pairs
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[upper_triangle_indices]

    # Filter by threshold using boolean indexing
    mask = similarities >= threshold
    filtered_indices = np.where(mask)[0]

    # Build candidate pairs list
    candidate_pairs = [
        (
            int(upper_triangle_indices[0][idx]),  # i index
            int(upper_triangle_indices[1][idx]),  # j index
            float(similarities[idx])              # similarity score
        )
        for idx in filtered_indices
    ]

    # Sort by similarity descending - process most similar pairs first
    # This improves merging efficiency: highly similar entities are more likely to merge,
    # and their merges may cascade to related entities
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"Found {len(candidate_pairs)} candidate pairs for LLM evaluation (sorted by similarity)")

    # Step 5: LLM decision making for candidate pairs
    print(f"Starting LLM evaluation of candidate pairs...")
    merged_pairs = 0
    llm_evaluated_pairs = 0
    cached_merges = 0
    cached_skips = 0
    skipped_already_merged = 0  # Track entities that were already merged
    active_entities = set(range(len(entities)))  # Track which entities are still active

    llm_concurrency_limit = max(1, getattr(rag_instance, "llm_model_max_async", 4))
    llm_semaphore = asyncio.Semaphore(llm_concurrency_limit)

    async def evaluate_pair_llm(
        entity_a_id: str,
        entity_b_id: str,
        desc_a: str,
        desc_b: str,
    ):
        """Call LLM to evaluate whether two entities should be merged."""
        user_prompt = PROMPTS["entity_merge_user"].format(
            a_entity_id=entity_a_id,
            a_description=desc_a,
            b_entity_id=entity_b_id,
            b_description=desc_b,
        )
        messages = [
            {"role": "system", "content": PROMPTS["entity_merge_system"]},
            {"role": "user", "content": PROMPTS["entity_merge_examples"]},
            {"role": "user", "content": user_prompt},
        ]
        async with llm_semaphore:
            return await llm_with_tools.ainvoke(messages)

    # Concurrent task management with LLM call tracking
    pending_evaluations: list[dict[str, Any]] = []   # Tasks waiting for LLM evaluation
    pending_merges: list[dict[str, Any]] = []         # Tasks ready to merge (evaluation complete)
    cached_pairs_to_apply: list[dict[str, Any]] = []  # Collect cached merge pairs

    active_evaluation_count = 0  # Count of ongoing LLM evaluations
    active_merge_count = 0       # Count of ongoing merges (includes LLM summarization)

    def get_active_llm_count() -> int:
        """Get total number of active LLM calls (evaluations + merges)"""
        return active_evaluation_count + active_merge_count

    def can_start_new_llm_task() -> bool:
        """Check if we can start a new LLM task (evaluation or merge)"""
        return get_active_llm_count() < llm_concurrency_limit

    async def process_evaluation_result(task_info: dict):
        """Process completed evaluation and optionally queue for merge"""
        nonlocal llm_evaluated_pairs, merged_pairs, active_evaluation_count, active_entities

        task: asyncio.Task = task_info["task"]
        i = task_info["i"]
        j = task_info["j"]
        entity_a_id = task_info["entity_a_id"]
        entity_b_id = task_info["entity_b_id"]
        similarity = task_info["similarity"]

        try:
            response = await task
            active_evaluation_count -= 1  # Evaluation LLM call complete
        except Exception as e:
            active_evaluation_count -= 1
            print(f"Error processing pair ({entity_a_id}, {entity_b_id}): {str(e)}")
            await schedule_next_tasks()  # Try to schedule more tasks
            return

        # Skip if entities already inactive (merged earlier)
        if i not in active_entities or j not in active_entities:
            await schedule_next_tasks()  # Try to schedule more tasks
            return

        # Fetch current descriptions (after any prior merges)
        current_entity_a = await rag_instance.chunk_entity_relation_graph.get_node(entity_a_id)
        current_entity_b = await rag_instance.chunk_entity_relation_graph.get_node(entity_b_id)

        if current_entity_a is None or current_entity_b is None:
            # Entities might have been removed due to earlier merges
            active_entities.discard(i)
            active_entities.discard(j)
            await schedule_next_tasks()  # Try to schedule more tasks
            return

        current_desc_a = current_entity_a.get("description", "")
        current_desc_b = current_entity_b.get("description", "")

        desc_a_used = task_info["initial_desc_a"]
        desc_b_used = task_info["initial_desc_b"]

        # Re-run LLM if entity descriptions changed since the initial evaluation
        if current_desc_a != desc_a_used or current_desc_b != desc_b_used:
            try:
                active_evaluation_count += 1  # Re-evaluation LLM call starts
                response = await evaluate_pair_llm(
                    entity_a_id,
                    entity_b_id,
                    current_desc_a,
                    current_desc_b,
                )
                active_evaluation_count -= 1  # Re-evaluation LLM call complete
                desc_a_used = current_desc_a
                desc_b_used = current_desc_b
            except Exception as e:
                active_evaluation_count -= 1
                print(f"Error re-evaluating pair ({entity_a_id}, {entity_b_id}): {str(e)}")
                await schedule_next_tasks()  # Try to schedule more tasks
                return

        llm_evaluated_pairs += 1

        # Check if LLM decided to merge
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Queue for merge
            pending_merges.append({
                "i": i,
                "j": j,
                "entity_a_id": entity_a_id,
                "entity_b_id": entity_b_id,
                "similarity": similarity,
                "desc_a_used": desc_a_used,
                "desc_b_used": desc_b_used,
                "tool_calls": response.tool_calls
            })
        else:
            # Cache skip decision
            content_a_final = f"{entity_a_id}|{desc_a_used}"
            content_b_final = f"{entity_b_id}|{desc_b_used}"
            hash_a_final = hashlib.md5(content_a_final.encode("utf-8")).hexdigest()
            hash_b_final = hashlib.md5(content_b_final.encode("utf-8")).hexdigest()
            entity_pair_id_final = f"{min(entity_a_id, entity_b_id)}||{max(entity_a_id, entity_b_id)}"

            entity_pair_cache[entity_pair_id_final] = {
                "hash_a": hash_a_final,
                "hash_b": hash_b_final,
                "decision": "skipped",
                "similarity": float(similarity),
                "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S"),
                "entity_a_id": entity_a_id,
                "entity_b_id": entity_b_id,
            }

        # Evaluation complete, try to schedule more tasks
        await schedule_next_tasks()

    async def execute_merge_task(merge_info: dict):
        """Execute merge operation (including LLM summarization)"""
        nonlocal merged_pairs, active_merge_count, entities, active_entities

        entity_a_id = merge_info["entity_a_id"]
        entity_b_id = merge_info["entity_b_id"]
        similarity = merge_info["similarity"]
        i = merge_info["i"]
        j = merge_info["j"]

        try:
            print(f"NEW MERGE: {entity_a_id} <- {entity_b_id} (similarity: {similarity:.3f})")

            active_merge_count += 1  # Merge (with LLM summarization) starts

            for call in merge_info["tool_calls"]:
                if call["name"] == "merge_entities_tool":
                    try:
                        merge_result = await merge_entities_tool.ainvoke(call["args"])
                        merge_successful = True
                        break
                    except Exception as e:
                        print(f"TOOL EXECUTION FAILED: {str(e)}")
                        merge_successful = False
                        break

            active_merge_count -= 1  # Merge complete

            if merge_successful:
                updated_node = await rag_instance.chunk_entity_relation_graph.get_node(entity_a_id)
                if updated_node and "description" in updated_node:
                    new_desc = updated_node["description"]
                    entities[i]["description"] = new_desc

                active_entities.discard(j)
                merged_pairs += 1
                decision = "merged"
            else:
                decision = "skipped"

        except Exception as e:
            active_merge_count -= 1
            print(f"Error executing merge ({entity_a_id}, {entity_b_id}): {str(e)}")
            decision = "skipped"

        # Cache decision
        desc_a_used = merge_info["desc_a_used"]
        desc_b_used = merge_info["desc_b_used"]
        content_a_final = f"{entity_a_id}|{desc_a_used}"
        content_b_final = f"{entity_b_id}|{desc_b_used}"
        hash_a_final = hashlib.md5(content_a_final.encode("utf-8")).hexdigest()
        hash_b_final = hashlib.md5(content_b_final.encode("utf-8")).hexdigest()
        entity_pair_id_final = f"{min(entity_a_id, entity_b_id)}||{max(entity_a_id, entity_b_id)}"

        entity_pair_cache[entity_pair_id_final] = {
            "hash_a": hash_a_final,
            "hash_b": hash_b_final,
            "decision": decision,
            "similarity": float(similarity),
            "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S"),
            "entity_a_id": entity_a_id,
            "entity_b_id": entity_b_id,
        }

        # Merge complete, try to schedule more tasks
        await schedule_next_tasks()

    def find_next_available_merge() -> dict | None:
        """Find first merge task that doesn't require waiting for locks"""
        nonlocal active_entities

        for idx, merge_info in enumerate(pending_merges):
            entity_a_id = merge_info["entity_a_id"]
            entity_b_id = merge_info["entity_b_id"]

            # Check if entities are still active
            i = merge_info["i"]
            j = merge_info["j"]
            if i not in active_entities or j not in active_entities:
                # Skip this task, remove from queue
                pending_merges.pop(idx)
                return None  # Try again with next iteration

            # Check if entities are locked
            if not lock_mgr.is_any_locked(entity_a_id, entity_b_id):
                # Found an available merge task
                return pending_merges.pop(idx)

        # All tasks are blocked or queue is empty
        return None

    async def schedule_next_tasks():
        """Schedule next available evaluation or merge tasks"""
        nonlocal active_evaluation_count, active_merge_count

        # Try to start new merge tasks (priority: clear pending merges first)
        while can_start_new_llm_task() and pending_merges:
            merge_info = find_next_available_merge()
            if merge_info is None:
                # All pending merges are blocked
                break

            # Start merge task in background
            asyncio.create_task(execute_merge_task(merge_info))

        # Try to start new evaluation tasks
        while can_start_new_llm_task() and pending_evaluations:
            task_info = pending_evaluations.pop(0)
            active_evaluation_count += 1

            # Process evaluation in background
            asyncio.create_task(process_evaluation_result(task_info))

    # Pre-compute entity description hashes for cache comparison
    print(f"Pre-computing entity description hashes...")
    entity_hashes = {}
    for i, entity in enumerate(entities):
        content = f"{entity['entity_id']}|{entity['description']}"
        entity_hashes[i] = hashlib.md5(content.encode("utf-8")).hexdigest()

    logger.info(f"Pre-computed {len(entity_hashes)} entity description hashes")

    # Pre-filter and queue candidate pairs
    print("Filtering and queueing candidate pairs...")
    pairs_to_evaluate = []

    for i, j, similarity in candidate_pairs:
        # Skip if either entity has already been merged
        if i not in active_entities or j not in active_entities:
            continue

        entity_a = entities[i]
        entity_b = entities[j]

        # Use pre-computed hashes (current description hashes)
        hash_a = entity_hashes[i]
        hash_b = entity_hashes[j]

        # Create entity pair ID (based on entity IDs, not descriptions)
        entity_a_id = entity_a['entity_id']
        entity_b_id = entity_b['entity_id']
        entity_pair_id = f"{min(entity_a_id, entity_b_id)}||{max(entity_a_id, entity_b_id)}"

        # Check cache with new logic
        should_evaluate = False  # Default: do not evaluate
        cache_reason = ""

        if entity_pair_id in entity_pair_cache:
            cached_data = entity_pair_cache[entity_pair_id]

            # Case 1: Already merged - skip (don't need to compare again)
            if cached_data["decision"] == "merged":
                cached_pairs_to_apply.append({
                    'i': i,
                    'j': j,
                    'entity_a': entity_a,
                    'entity_b': entity_b,
                    'cached_result': cached_data
                })
                logger.debug(
                    f"Cache HIT (merged before): {entity_a_id} vs {entity_b_id}"
                )
                continue

            # Case 2: Previously skipped - check if descriptions changed
            cached_hash_a = cached_data.get("hash_a", "")
            cached_hash_b = cached_data.get("hash_b", "")

            if hash_a == cached_hash_a and hash_b == cached_hash_b:
                # Case 2a: Descriptions unchanged - use cache (skip)
                cached_skips += 1
                logger.debug(
                    f"Cache HIT (skip, desc unchanged): {entity_a_id} vs {entity_b_id}"
                )
                continue
            else:
                # Case 2b: Descriptions changed - need to re-evaluate
                should_evaluate = True
                cache_reason = "desc_changed"
        else:
            # Case 3: Never compared before - need to evaluate
            should_evaluate = True
            cache_reason = "new_pair"
            logger.debug(f"New pair (not in cache): {entity_a_id} vs {entity_b_id}")

        # If we reach here, we need LLM evaluation
        if should_evaluate:
            pairs_to_evaluate.append((i, j, similarity, entity_a, entity_b))

    print(f"Queued {len(pairs_to_evaluate)} pairs for LLM evaluation (after cache filtering)")

    # Process pairs with progress bar showing actual work
    total_work_items = len(pairs_to_evaluate)
    completed_items = 0

    pbar = atqdm(
        total=total_work_items,
        desc="Evaluating & merging pairs",
        unit="pair"
    )

    # Track completed tasks to update progress
    completed_evaluations = 0
    last_completed = 0

    async def update_progress():
        """Update progress bar based on completed evaluations"""
        nonlocal completed_evaluations, last_completed
        current_completed = llm_evaluated_pairs
        if current_completed > last_completed:
            delta = current_completed - last_completed
            pbar.update(delta)
            last_completed = current_completed
            pbar.set_postfix({
                'merged': merged_pairs,
                'cached': cached_merges,
                'eval': llm_evaluated_pairs,
                'pending': len(pending_evaluations) + len(pending_merges),
                'active': get_active_llm_count()
            })

    # Start queueing evaluation tasks
    for i, j, similarity, entity_a, entity_b in pairs_to_evaluate:
        # Schedule new LLM evaluation
        task = asyncio.create_task(
            evaluate_pair_llm(
                entity_a["entity_id"],
                entity_b["entity_id"],
                entity_a["description"],
                entity_b["description"],
            )
        )

        pending_evaluations.append(
            {
                "task": task,
                "i": i,
                "j": j,
                "entity_a_id": entity_a["entity_id"],
                "entity_b_id": entity_b["entity_id"],
                "similarity": similarity,
                "initial_desc_a": entity_a["description"],
                "initial_desc_b": entity_b["description"],
            }
        )

        # Schedule next available tasks
        await schedule_next_tasks()

        # Update progress periodically
        await update_progress()

    # Process remaining pending tasks (evaluations and merges)
    while len(pending_evaluations) > 0 or len(pending_merges) > 0 or get_active_llm_count() > 0:
        # Try to schedule more tasks
        await schedule_next_tasks()

        # Wait a bit to let tasks progress
        await asyncio.sleep(0.1)

        # Update progress bar
        await update_progress()

    # Final update
    await update_progress()

    # Close the main progress bar only after ALL tasks are done
    pbar.close()

    # Apply cached merges in batch (after all LLM evaluations complete)
    if cached_pairs_to_apply:
        print(f"\nApplying {len(cached_pairs_to_apply)} cached merge decisions...")

        # Batch validate entity existence
        validation_tasks = []
        for cached_pair in cached_pairs_to_apply:
            validation_tasks.append(
                rag_instance.chunk_entity_relation_graph.get_node(cached_pair['entity_a']["entity_id"])
            )
            validation_tasks.append(
                rag_instance.chunk_entity_relation_graph.get_node(cached_pair['entity_b']["entity_id"])
            )

        validation_results = await asyncio.gather(*validation_tasks)

        # Process cached merges
        merge_tasks = []
        for idx, cached_pair in enumerate(cached_pairs_to_apply):
            i = cached_pair['i']
            j = cached_pair['j']

            # Check validation results (2 nodes per pair)
            entity_a_exists = validation_results[idx * 2]
            entity_b_exists = validation_results[idx * 2 + 1]

            # Skip if either entity has already been merged
            if i not in active_entities or j not in active_entities:
                continue

            if entity_a_exists is None or entity_b_exists is None:
                skipped_already_merged += 1
                active_entities.discard(j)
                print(
                    f"SKIPPED (cached): Entities already merged - {cached_pair['entity_a']['entity_id']} & {cached_pair['entity_b']['entity_id']}"
                )
                continue

            # Schedule merge operation
            merge_tasks.append({
                'task': merge_entities_tool.ainvoke({
                    "a_entity_id": cached_pair['entity_a']["entity_id"],
                    "b_entity_id": cached_pair['entity_b']["entity_id"],
                }),
                'i': i,
                'j': j,
                'entity_a_id': cached_pair['entity_a']["entity_id"]
            })

        # Execute all cached merges in parallel
        if merge_tasks:
            merge_results = await asyncio.gather(*[mt['task'] for mt in merge_tasks], return_exceptions=True)

            # Update entity descriptions and active_entities
            for merge_task, result in zip(merge_tasks, merge_results):
                if isinstance(result, Exception):
                    print(f"Failed to apply cached merge: {result}")
                    continue

                # Update entity description
                updated_node = await rag_instance.chunk_entity_relation_graph.get_node(
                    merge_task['entity_a_id']
                )
                if updated_node and "description" in updated_node:
                    entities[merge_task['i']]["description"] = updated_node["description"]

                active_entities.discard(merge_task['j'])
                merged_pairs += 1
                cached_merges += 1

    processing_time = time_module.time() - start_time
    remaining_entities = len(active_entities)


    # Save updated cache to disk
    try:
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(entity_pair_cache, f, ensure_ascii=False, indent=2)
        # print(f"Saved {len(entity_pair_cache)} entity pair comparisons to cache")  # Remove this line
    except Exception as e:
        print(f"Failed to save entity pair cache: {e}")

    result = {
        "total_entities": total_entities,
        "candidate_pairs": len(candidate_pairs),
        "llm_evaluated_pairs": llm_evaluated_pairs,
        "merged_pairs": merged_pairs,
        "new_merges": merged_pairs - cached_merges,
        "cached_merges": cached_merges,
        "cached_skips": cached_skips,
        "skipped_already_merged": skipped_already_merged,
        "remaining_entities": remaining_entities,
        "processing_time": processing_time,
        "similarity_threshold": threshold,
        "total_cached_pairs": len(entity_pair_cache),
        "entities_summarized": summarization_stats["entities_summarized"],
        "relations_summarized": summarization_stats["relations_summarized"],
        "entity_summarization_llm_calls": summarization_stats["entity_llm_calls"],
        "relation_summarization_llm_calls": summarization_stats["relation_llm_calls"]
    }

    # Ensure all merge operations are fully completed before finishing
    if merged_pairs > 0:
        # print(f"Finalizing {merged_pairs} merge operations...")
        # Force a final graph persistence to ensure all changes are written
        await rag_instance.chunk_entity_relation_graph.index_done_callback()

    # Print summary
    # print(f"Agentic merging summary: {merged_pairs} merges ({result['new_merges']} new, {cached_merges} cached), {skipped_already_merged} skipped (already merged)")

    return result
