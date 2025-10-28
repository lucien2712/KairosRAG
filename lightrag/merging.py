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
from .utils import logger
from .prompt import PROMPTS


async def single_pass_agentic_merging(rag_instance, threshold: float = 0.8, langchain_client=None) -> dict:
    """
    Asynchronously perform intelligent entity merging using vector similarity and LLM decision making.

    Args:
        rag_instance: LightRAG instance containing all necessary components
        threshold: Cosine similarity threshold for candidate pair filtering (default: 0.8)
        langchain_client: Optional shared LangChain ChatOpenAI client (if None, creates new one)

    Returns:
        dict: Statistics about the merging process
    """
    import time as time_module
    from .operate import _handle_entity_relation_summary
    from .utils import compute_mdhash_id

    start_time = time_module.time()
    print(f"Starting agentic entity merging with threshold={threshold}")

    # Load entity comparison cache
    cache_file_path = os.path.join(rag_instance.working_dir, "entity_merge_cache.json")
    entity_pair_cache = {}
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                entity_pair_cache = json.load(f)
            print(f"Loaded {len(entity_pair_cache)} cached entity pair comparisons")
        except Exception as e:
            print(f"Failed to load entity pair cache: {e}")
            entity_pair_cache = {}

    # Initialize LLM using rag_instance.tool_llm_model_name or use shared client
    if langchain_client is None:
        llm = ChatOpenAI(
            model=rag_instance.tool_llm_model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
        )
        print(f"LLM initialized: {rag_instance.tool_llm_model_name}")
    else:
        llm = langchain_client
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

    # Define the merge tool for LLM
    @tool
    async def merge_entities_tool(a_entity_id: str, b_entity_id: str) -> str:
        """
        If two entities represent the same real-world entity, merge b_entity_id into a_entity_id.
        """
        try:
            # print(f"TOOL CALLED: Merging {b_entity_id} -> {a_entity_id}")

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
                for src_id, tgt_id in edge_tuples:
                    # Get the full edge data
                    edge = await rag_instance.chunk_entity_relation_graph.get_edge(src_id, tgt_id)

                    if edge and 'description' in edge:
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

                        # Update relationship if description changed
                        if summarized_desc != description:
                            # Update graph storage
                            updated_edge_data = dict(edge)
                            updated_edge_data['description'] = summarized_desc
                            await rag_instance.chunk_entity_relation_graph.upsert_edge(
                                src_id,
                                tgt_id,
                                updated_edge_data
                            )

                            # Update vector database
                            rel_vdb_id = compute_mdhash_id(src_id + tgt_id, prefix="rel-")
                            keywords = edge.get('keywords', '')
                            rel_content = f"{keywords}\t{src_id}\n{tgt_id}\n{summarized_desc}"

                            await rag_instance.relationships_vdb.upsert(
                                {rel_vdb_id: {
                                    "src_id": src_id,
                                    "tgt_id": tgt_id,
                                    "content": rel_content
                                }}
                            )

                            summarization_stats["relations_summarized"] += 1
                            if llm_used:
                                summarization_stats["relation_llm_calls"] += 1

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
    entities = []
    entity_embeddings = []

    if entities_vdb_data:
        sample_entity = next(iter(entities_vdb_data.values()))


    for entity_name, entity_data in entities_vdb_data.items():
        # print(f"Processing entity: {entity_name}")
        # print(f"    - Has content: {'content' in entity_data}")
        # print(f"    - Has vector: {'vector' in entity_data}")

        if 'content' in entity_data and 'vector' in entity_data:
            # Get description from graph node data
            node_data = entity_id_to_node.get(entity_name, {})
            description = node_data.get('description', '')

            # print(f"    - Found in graph: {entity_name in entity_id_to_node}")
            # print(f"    - Description length: {len(description)}")

            # Decode the base64 vector (matching nano_vector_db_impl storage format)
            try:
                # Decode base64
                compressed_vector = base64.b64decode(entity_data['vector'])
                # Decompress with zlib
                vector_bytes = zlib.decompress(compressed_vector)
                # Convert to numpy array (stored as float16, convert to float32)
                vector_array = np.frombuffer(vector_bytes, dtype=np.float16).astype(np.float32)

                # Only add if we have a reasonable vector size
                if len(vector_array) > 0:
                    entities.append({
                        'entity_id': entity_name,
                        'description': description,
                    })
                    entity_embeddings.append(vector_array.tolist())
                    # print(f"    - Added to entities list (vector dim: {len(vector_array)})")
                else:
                    print(f"    - Empty vector after processing")
            except Exception as e:
                print(f"    - Error decoding vector: {e}")
        else:
            print(f"    - Missing required fields")

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

    # Check and standardize vector dimensions
    vector_dimensions = [len(emb) for emb in entity_embeddings]
    max_dim = max(vector_dimensions)
    min_dim = min(vector_dimensions)

    # print(f"Vector dimensions - Min: {min_dim}, Max: {max_dim}")

    # Pad shorter vectors with zeros to match the longest
    standardized_embeddings = []
    for emb in entity_embeddings:
        if len(emb) < max_dim:
            padded_emb = emb + [0.0] * (max_dim - len(emb))
            standardized_embeddings.append(padded_emb)
        else:
            standardized_embeddings.append(emb[:max_dim])  # Truncate if longer

    try:
        embeddings = np.array(standardized_embeddings, dtype=np.float32)
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

    # Step 4: Find candidate pairs above threshold
    print(f"Filtering candidate pairs with similarity >= {threshold}...")
    candidate_pairs = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:
                candidate_pairs.append((i, j, float(similarity)))

    print(f"Found {len(candidate_pairs)} candidate pairs for LLM evaluation")

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

    pending_tasks: list[dict[str, Any]] = []

    async def process_next_pending_task():
        nonlocal llm_evaluated_pairs, merged_pairs, cached_merges, cached_skips, skipped_already_merged

        if not pending_tasks:
            return

        task_info = pending_tasks.pop(0)
        task: asyncio.Task = task_info["task"]
        i = task_info["i"]
        j = task_info["j"]
        entity_a_id = task_info["entity_a_id"]
        entity_b_id = task_info["entity_b_id"]
        similarity = task_info["similarity"]

        try:
            response = await task
        except Exception as e:
            print(f"Error processing pair ({entity_a_id}, {entity_b_id}): {str(e)}")
            return

        # Skip if entities already inactive (merged earlier)
        if i not in active_entities or j not in active_entities:
            return

        # Fetch current descriptions (after any prior merges)
        current_entity_a = await rag_instance.chunk_entity_relation_graph.get_node(entity_a_id)
        current_entity_b = await rag_instance.chunk_entity_relation_graph.get_node(entity_b_id)

        if current_entity_a is None or current_entity_b is None:
            # Entities might have been removed due to earlier merges
            active_entities.discard(i)
            active_entities.discard(j)
            return

        current_desc_a = current_entity_a.get("description", "")
        current_desc_b = current_entity_b.get("description", "")

        desc_a_used = task_info["initial_desc_a"]
        desc_b_used = task_info["initial_desc_b"]

        # Re-run LLM if entity descriptions changed since the initial evaluation
        if current_desc_a != desc_a_used or current_desc_b != desc_b_used:
            try:
                response = await evaluate_pair_llm(
                    entity_a_id,
                    entity_b_id,
                    current_desc_a,
                    current_desc_b,
                )
                desc_a_used = current_desc_a
                desc_b_used = current_desc_b
            except Exception as e:
                print(f"Error re-evaluating pair ({entity_a_id}, {entity_b_id}): {str(e)}")
                return

        decision = "skipped"
        merge_successful = False

        if hasattr(response, "tool_calls") and response.tool_calls:
            print(
                f"NEW MERGE: {entity_a_id} <- {entity_b_id} (similarity: {similarity:.3f})"
            )

            for call in response.tool_calls:
                if call["name"] == "merge_entities_tool":
                    try:
                        merge_result = await merge_entities_tool.ainvoke(call["args"])
                        merge_successful = True
                        break
                    except Exception as e:
                        print(f"TOOL EXECUTION FAILED: {str(e)}")
                        merge_successful = False
                        break

            if merge_successful:
                updated_node = await rag_instance.chunk_entity_relation_graph.get_node(
                    entity_a_id
                )
                if updated_node and "description" in updated_node:
                    entities[i]["description"] = updated_node["description"]

                active_entities.discard(j)
                merged_pairs += 1
                decision = "merged"
            else:
                decision = "skipped"

        # Cache decision using latest descriptions
        content_a_final = f"{entity_a_id}|{desc_a_used}"
        content_b_final = f"{entity_b_id}|{desc_b_used}"
        hash_a_final = hashlib.md5(content_a_final.encode("utf-8")).hexdigest()
        hash_b_final = hashlib.md5(content_b_final.encode("utf-8")).hexdigest()
        pair_key_final = f"{min(hash_a_final, hash_b_final)}_{max(hash_a_final, hash_b_final)}"

        entity_pair_cache[pair_key_final] = {
            "decision": decision,
            "similarity": float(similarity),
            "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S"),
            "entity_a_id": entity_a_id,
            "entity_b_id": entity_b_id,
        }

        # Update statistics for LLM evaluations (count once per final evaluation)
        llm_evaluated_pairs += 1

    for i, j, similarity in candidate_pairs:
        # Skip if either entity has already been merged
        if i not in active_entities or j not in active_entities:
            continue

        entity_a = entities[i]
        entity_b = entities[j]

        # Create hash key for this entity pair based on content (not ID)
        content_a = f"{entity_a['entity_id']}|{entity_a['description']}"
        content_b = f"{entity_b['entity_id']}|{entity_b['description']}"
        hash_a = hashlib.md5(content_a.encode("utf-8")).hexdigest()
        hash_b = hashlib.md5(content_b.encode("utf-8")).hexdigest()

        # Create consistent pair key (sorted to ensure consistent ordering)
        pair_key = f"{min(hash_a, hash_b)}_{max(hash_a, hash_b)}"

        # Check cache first
        if pair_key in entity_pair_cache:
            # Process any pending asynchronous evaluations before applying cached decision
            while pending_tasks:
                await process_next_pending_task()

            cached_result = entity_pair_cache[pair_key]

            if cached_result["decision"] == "merged":
                cached_merges += 1

                try:
                    current_entities = await rag_instance.chunk_entity_relation_graph.get_node(
                        entity_a["entity_id"]
                    )
                    target_entities = await rag_instance.chunk_entity_relation_graph.get_node(
                        entity_b["entity_id"]
                    )

                    if current_entities is None or target_entities is None:
                        skipped_already_merged += 1
                        active_entities.discard(j)
                        print(
                            f"SKIPPED: Entities already merged - {entity_a['entity_id']} & {entity_b['entity_id']}"
                        )
                        continue

                    result = await merge_entities_tool.ainvoke(
                        {
                            "a_entity_id": entity_a["entity_id"],
                            "b_entity_id": entity_b["entity_id"],
                        }
                    )

                    updated_node = await rag_instance.chunk_entity_relation_graph.get_node(
                        entity_a["entity_id"]
                    )
                    if updated_node and "description" in updated_node:
                        entities[i]["description"] = updated_node["description"]

                    active_entities.discard(j)
                    merged_pairs += 1
                except Exception as e:
                    print(f"Failed to apply cached merge: {e}")
            else:
                cached_skips += 1
            continue

        # Schedule new LLM evaluation with concurrency control
        task = asyncio.create_task(
            evaluate_pair_llm(
                entity_a["entity_id"],
                entity_b["entity_id"],
                entity_a["description"],
                entity_b["description"],
            )
        )

        pending_tasks.append(
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

        # If concurrency limit reached, process the oldest pending task
        if len(pending_tasks) >= llm_concurrency_limit:
            await process_next_pending_task()

    # Process remaining pending evaluations
    while pending_tasks:
        await process_next_pending_task()

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
