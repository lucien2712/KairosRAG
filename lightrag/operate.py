from __future__ import annotations
from functools import partial

import asyncio
import json
import re
import os
import json_repair
import numpy as np
from typing import Any, AsyncIterator
from collections import Counter, defaultdict

from .utils import (
    logger,
    compute_mdhash_id,
    Tokenizer,
    is_float_regex,
    sanitize_and_normalize_extracted_text,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    get_conversation_turns,
    use_llm_func_with_cache,
    update_chunk_cache_list,
    remove_think_tags,
    pick_by_weighted_polling,
    pick_by_vector_similarity,
    cosine_similarity,
    process_chunks_unified,
    build_file_path,
    find_table_boundaries,
    get_token_positions,
    find_token_position_near_char,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import PROMPTS
from .constants import (
    GRAPH_FIELD_SEP,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
)
from .kg.shared_storage import get_storage_keyed_lock
import time
from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    table_context_tokens: int = 100,
) -> list[dict[str, Any]]:
    """Enhanced chunking with table-aware processing.

    Tables are kept complete in single chunks with additional context tokens
    before and after the table content.
    """
    # Find all table boundaries in the content
    table_boundaries = find_table_boundaries(content)

    if not table_boundaries:
        # No tables found, use original chunking logic
        return _original_chunking_by_token_size(
            tokenizer, content, split_by_character, split_by_character_only,
            overlap_token_size, max_token_size
        )

    results: list[dict[str, Any]] = []

    # Process each table as an independent chunk with smart context boundaries
    processed_char_ranges = []

    for i, (table_start_char, table_end_char) in enumerate(table_boundaries):
        # Calculate initial context boundaries
        context_start_char = max(0, table_start_char - (table_context_tokens * 4))
        context_end_char = min(len(content), table_end_char + (table_context_tokens * 4))

        # Adjust boundaries to avoid including other tables
        # Check if context overlaps with previous table
        if i > 0:
            prev_table_start, prev_table_end = table_boundaries[i-1]
            if context_start_char < prev_table_end:
                # Adjust start to avoid including previous table
                context_start_char = prev_table_end + 1

        # Check if context overlaps with next table
        if i < len(table_boundaries) - 1:
            next_table_start, next_table_end = table_boundaries[i+1]
            if context_end_char > next_table_start:
                # Adjust end to avoid including next table
                context_end_char = next_table_start - 1

        # Ensure we still include the complete table
        context_start_char = min(context_start_char, table_start_char)
        context_end_char = max(context_end_char, table_end_char)

        # Extract complete content using character positions
        complete_table_content = content[context_start_char:context_end_char]

        # Get accurate token count for the complete content
        complete_tokens = tokenizer.encode(complete_table_content)

        # Verify table completeness - should be exactly 1 table per chunk
        table_starts = complete_table_content.count('<table')
        table_ends = complete_table_content.count('</table>')

        results.append({
            "tokens": len(complete_tokens),
            "content": complete_table_content.strip(),
            "chunk_order_index": len(results),
            "contains_table": True,
            "table_complete": table_starts == table_ends and table_starts == 1,
        })

        # Record the processed character range
        processed_char_ranges.append((context_start_char, context_end_char))

    # Process non-table content with regular chunking
    processed_char_ranges.sort()  # Sort by start position
    current_char = 0

    for start_char, end_char in processed_char_ranges:
        # Process content before this table chunk
        if current_char < start_char:
            before_table_content = content[current_char:start_char].strip()
            if before_table_content:  # Only if there's content
                # Apply regular chunking to this section
                before_chunks = _original_chunking_by_token_size(
                    tokenizer, before_table_content, split_by_character,
                    split_by_character_only, overlap_token_size, max_token_size
                )
                # Update chunk indices
                for chunk in before_chunks:
                    chunk["chunk_order_index"] = len(results)
                    results.append(chunk)
        current_char = end_char

    # Process remaining content after last table
    if current_char < len(content):
        remaining_content = content[current_char:].strip()
        if remaining_content:
            remaining_chunks = _original_chunking_by_token_size(
                tokenizer, remaining_content, split_by_character,
                split_by_character_only, overlap_token_size, max_token_size
            )
            # Update chunk indices
            for chunk in remaining_chunks:
                chunk["chunk_order_index"] = len(results)
                results.append(chunk)

    # Sort results by chunk order to maintain document flow
    results.sort(key=lambda x: x.get("chunk_order_index", 0))

    # Re-index the chunks
    for i, chunk in enumerate(results):
        chunk["chunk_order_index"] = i

    return results


def _original_chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> list[dict[str, Any]]:
    """Original chunking logic without table awareness."""
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > max_token_size:
                    for start in range(
                        0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + max_token_size]
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + max_token_size])
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


def _chunk_token_sequence(
    tokenizer: Tokenizer,
    token_sequence: list[int],
    overlap_token_size: int,
    max_token_size: int,
    start_index: int
) -> list[dict[str, Any]]:
    """Chunk a sequence of tokens using regular chunking logic."""
    results = []

    for i, start in enumerate(
        range(0, len(token_sequence), max_token_size - overlap_token_size)
    ):
        chunk_tokens = token_sequence[start : start + max_token_size]
        chunk_content = tokenizer.decode(chunk_tokens)

        results.append({
            "tokens": len(chunk_tokens),
            "content": chunk_content.strip(),
            "chunk_order_index": start_index + i,
            "contains_table": False,
        })

    return results


async def _handle_entity_relation_summary(
    description_type: str,
    entity_or_relation_name: str,
    description_list: list[str],
    seperator: str,
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> tuple[str, bool]:
    """Handle entity relation description summary using map-reduce approach.

    This function summarizes a list of descriptions using a map-reduce strategy:
    1. If total tokens < summary_context_size and len(description_list) < force_llm_summary_on_merge, no need to summarize
    2. If total tokens < summary_max_tokens, summarize with LLM directly
    3. Otherwise, split descriptions into chunks that fit within token limits
    4. Summarize each chunk, then recursively process the summaries
    5. Continue until we get a final summary within token limits or num of descriptions is less than force_llm_summary_on_merge

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        description_list: List of description strings to summarize
        global_config: Global configuration containing tokenizer and limits
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Tuple of (final_summarized_description_string, llm_was_used_boolean)
    """
    # Handle empty input
    if not description_list:
        return "", False

    # If only one description, return it directly (no need for LLM call)
    if len(description_list) == 1:
        return description_list[0], False

    # Get configuration
    tokenizer: Tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]
    summary_max_tokens = global_config["summary_max_tokens"]
    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    current_list = description_list[:]  # Copy the list to avoid modifying original
    llm_was_used = False  # Track whether LLM was used during the entire process

    # Iterative map-reduce process
    while True:
        # Calculate total tokens in current list
        total_tokens = sum(len(tokenizer.encode(desc)) for desc in current_list)

        # If total length is within limits, perform final summarization
        if total_tokens <= summary_context_size or len(current_list) <= 2:
            if (
                len(current_list) < force_llm_summary_on_merge
                and total_tokens < summary_max_tokens
            ):
                # no LLM needed, just join the descriptions
                final_description = seperator.join(current_list)
                return final_description if final_description else "", llm_was_used
            else:
                if total_tokens > summary_context_size and len(current_list) <= 2:
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize descpriton found"
                    )
                # Final summarization of remaining descriptions - LLM will be used
                final_summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    current_list,
                    global_config,
                    llm_response_cache,
                )
                return final_summary, True  # LLM was used for final summarization

        # Need to split into chunks - Map phase
        # Ensure each chunk has minimum 2 descriptions to guarantee progress
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Currently least 3 descriptions in current_list
        for i, desc in enumerate(current_list):
            desc_tokens = len(tokenizer.encode(desc))

            # If adding current description would exceed limit, finalize current chunk
            if current_tokens + desc_tokens > summary_context_size and current_chunk:
                # Ensure we have at least 2 descriptions in the chunk (when possible)
                if len(current_chunk) == 1:
                    # Force add one more description to ensure minimum 2 per chunk
                    current_chunk.append(desc)
                    chunks.append(current_chunk)
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize descpriton found"
                    )
                    current_chunk = []  # next group is empty
                    current_tokens = 0
                else:  # curren_chunk is ready for summary in reduce phase
                    chunks.append(current_chunk)
                    current_chunk = [desc]  # leave it for next group
                    current_tokens = desc_tokens
            else:
                current_chunk.append(desc)
                current_tokens += desc_tokens

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"   Summarizing {entity_or_relation_name}: Map {len(current_list)} descriptions into {len(chunks)} groups"
        )

        # Reduce phase: summarize each group from chunks
        new_summaries = []
        for chunk in chunks:
            if len(chunk) == 1:
                # Optimization: single description chunks don't need LLM summarization
                new_summaries.append(chunk[0])
            else:
                # Multiple descriptions need LLM summarization
                summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    chunk,
                    global_config,
                    llm_response_cache,
                )
                new_summaries.append(summary)
                llm_was_used = True  # Mark that LLM was used in reduce phase

        # Update current list with new summaries for next iteration
        current_list = new_summaries


async def _summarize_descriptions(
    description_type: str,
    description_name: str,
    description_list: list[str],
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> str:
    """Helper function to summarize a list of descriptions using LLM.

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        descriptions: List of description strings to summarize
        global_config: Global configuration containing LLM function and settings
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Summarized description string
    """
    use_llm_func: callable = global_config["llm_model_func"]
    # Apply higher priority (8) to entity/relation summary tasks
    use_llm_func = partial(use_llm_func, _priority=8)

    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)

    summary_length_recommended = global_config["summary_length_recommended"]

    prompt_template = PROMPTS["summarize_entity_descriptions"]

    # Join descriptions and apply token-based truncation if necessary
    joined_descriptions = "\n\n".join(description_list)
    tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]

    # Token-based truncation to ensure input fits within limits
    tokens = tokenizer.encode(joined_descriptions)
    if len(tokens) > summary_context_size:
        truncated_tokens = tokens[:summary_context_size]
        joined_descriptions = tokenizer.decode(truncated_tokens)

    # Prepare context for the prompt
    context_base = dict(
        description_type=description_type,
        description_name=description_name,
        description_list=joined_descriptions,
        summary_length=summary_length_recommended,
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)

    logger.debug(
        f"Summarizing {len(description_list)} descriptions for: {description_name}"
    )

    # Use LLM function with cache (higher priority for summary generation)
    summary = await use_llm_func_with_cache(
        use_prompt,
        use_llm_func,
        llm_response_cache=llm_response_cache,
        cache_type="extract",
    )
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
    timestamp: str = "",
):
    if len(record_attributes) < 4 or "entity" not in record_attributes[0]:
        if len(record_attributes) > 1 and "entity" in record_attributes[0]:
            logger.warning(
                f"Entity extraction failed in {chunk_key}: expecting 4 fields but got {len(record_attributes)}"
            )
            logger.warning(f"Entity extracted: {record_attributes[1]}")
            # Return the malformed record for potential retry
            return "MALFORMED_ENTITY"
        return None

    try:
        entity_name = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )

        # Validate entity name after all cleaning steps
        if not entity_name or not entity_name.strip():
            logger.warning(
                f"Entity extraction error: entity name became empty after cleaning. Original: '{record_attributes[1]}'"
            )
            return None

        # Process entity type with same cleaning pipeline
        entity_type = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        if not entity_type.strip() or any(
            char in entity_type for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
        ):
            logger.warning(
                f"Entity extraction error: invalid entity type in: {record_attributes}"
            )
            return None

        # Captitalize first letter of entity_type
        entity_type = entity_type.title()

        # Process entity description with same cleaning pipeline
        entity_description = sanitize_and_normalize_extracted_text(record_attributes[3])

        if not entity_description.strip():
            logger.warning(
                f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
            )
            return None

        # Automatically prepend timestamp to description if provided and not already present
        if timestamp and timestamp.strip() and not entity_description.startswith(timestamp):
            entity_description = f"[Time: {timestamp}] {entity_description}"

        return dict(
            entity_name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=chunk_key,
            file_path=file_path,
        )

    except ValueError as e:
        logger.error(
            f"Entity extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Entity extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
    timestamp: str = "",
):
    if len(record_attributes) < 5 or "relationship" not in record_attributes[0]:
        if len(record_attributes) > 1 and "relationship" in record_attributes[0]:
            logger.warning(
                f"Relation extraction failed in {chunk_key}: expecting 5 fields but got {len(record_attributes)}"
            )
            logger.warning(f"Relation extracted: {record_attributes[1]}")
            # Return the malformed record for potential retry
            return "MALFORMED_RELATIONSHIP"
        return None

    try:
        source = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )
        target = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        # Validate entity names after all cleaning steps
        if not source:
            logger.warning(
                f"Relationship extraction error: source entity became empty after cleaning. Original: '{record_attributes[1]}'"
            )
            return None

        if not target:
            logger.warning(
                f"Relationship extraction error: target entity became empty after cleaning. Original: '{record_attributes[2]}'"
            )
            return None

        if source == target:
            logger.debug(
                f"Relationship source and target are the same in: {record_attributes}"
            )
            return None

        # Process keywords with same cleaning pipeline
        edge_keywords = sanitize_and_normalize_extracted_text(
            record_attributes[3], remove_inner_quotes=True
        )
        edge_keywords = edge_keywords.replace("，", ",")

        # Process relationship description with same cleaning pipeline
        edge_description = sanitize_and_normalize_extracted_text(record_attributes[4])

        # Automatically prepend timestamp to description if provided and not already present
        if timestamp and timestamp.strip() and not edge_description.startswith(timestamp):
            edge_description = f"{timestamp}: {edge_description}"

        edge_source_id = chunk_key
        weight = (
            float(record_attributes[-1].strip('"').strip("'"))
            if is_float_regex(record_attributes[-1].strip('"').strip("'"))
            else 1.0
        )

        return dict(
            src_id=source,
            tgt_id=target,
            weight=weight,
            description=edge_description,
            keywords=edge_keywords,
            source_id=edge_source_id,
            file_path=file_path,
        )

    except ValueError as e:
        logger.warning(
            f"Relationship extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.warning(
            f"Relationship extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def _rebuild_knowledge_from_chunks(
    entities_to_rebuild: dict[str, set[str]],
    relationships_to_rebuild: dict[tuple[str, str], set[str]],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_storage: BaseKVStorage,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    pipeline_status: dict | None = None,
    pipeline_status_lock=None,
) -> None:
    """Rebuild entity and relationship descriptions from cached extraction results with parallel processing

    This method uses cached LLM extraction results instead of calling LLM again,
    following the same approach as the insert process. Now with parallel processing
    controlled by llm_model_max_async and using get_storage_keyed_lock for data consistency.

    Args:
        entities_to_rebuild: Dict mapping entity_name -> set of remaining chunk_ids
        relationships_to_rebuild: Dict mapping (src, tgt) -> set of remaining chunk_ids
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_storage: Text chunks storage
        llm_response_cache: LLM response cache
        global_config: Global configuration containing llm_model_max_async
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
    """
    if not entities_to_rebuild and not relationships_to_rebuild:
        return

    # Get all referenced chunk IDs
    all_referenced_chunk_ids = set()
    for chunk_ids in entities_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)
    for chunk_ids in relationships_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)

    status_message = f"Rebuilding knowledge from {len(all_referenced_chunk_ids)} cached chunk extractions (parallel processing)"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Get cached extraction results for these chunks using storage
    #    cached_results： chunk_id -> [list of extraction result from LLM cache sorted by created_at]
    cached_results = await _get_cached_extraction_results(
        llm_response_cache,
        all_referenced_chunk_ids,
        text_chunks_storage=text_chunks_storage,
    )

    if not cached_results:
        status_message = "No cached extraction results found, cannot rebuild"
        logger.warning(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
        return

    # Process cached results to get entities and relationships for each chunk
    chunk_entities = {}  # chunk_id -> {entity_name: [entity_data]}
    chunk_relationships = {}  # chunk_id -> {(src, tgt): [relationship_data]}

    for chunk_id, extraction_results in cached_results.items():
        try:
            # Handle multiple extraction results per chunk
            chunk_entities[chunk_id] = defaultdict(list)
            chunk_relationships[chunk_id] = defaultdict(list)

            # process multiple LLM extraction results for a single chunk_id
            for extraction_result in extraction_results:
                entities, relationships = await _parse_extraction_result(
                    text_chunks_storage=text_chunks_storage,
                    extraction_result=extraction_result,
                    chunk_id=chunk_id,
                )

                # Merge entities and relationships from this extraction result
                # Only keep the first occurrence of each entity_name in the same chunk_id
                for entity_name, entity_list in entities.items():
                    if (
                        entity_name not in chunk_entities[chunk_id]
                        or len(chunk_entities[chunk_id][entity_name]) == 0
                    ):
                        chunk_entities[chunk_id][entity_name].extend(entity_list)

                # Only keep the first occurrence of each rel_key in the same chunk_id
                for rel_key, rel_list in relationships.items():
                    if (
                        rel_key not in chunk_relationships[chunk_id]
                        or len(chunk_relationships[chunk_id][rel_key]) == 0
                    ):
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)

        except Exception as e:
            status_message = (
                f"Failed to parse cached extraction result for chunk {chunk_id}: {e}"
            )
            logger.info(status_message)  # Per requirement, change to info
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = status_message
                    pipeline_status["history_messages"].append(status_message)
            continue

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # Counters for tracking progress
    rebuilt_entities_count = 0
    rebuilt_relationships_count = 0
    failed_entities_count = 0
    failed_relationships_count = 0

    async def _locked_rebuild_entity(entity_name, chunk_ids):
        nonlocal rebuilt_entities_count, failed_entities_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                try:
                    await _rebuild_single_entity(
                        knowledge_graph_inst=knowledge_graph_inst,
                        entities_vdb=entities_vdb,
                        entity_name=entity_name,
                        chunk_ids=chunk_ids,
                        chunk_entities=chunk_entities,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                    )
                    rebuilt_entities_count += 1
                    status_message = (
                        f"Rebuilt `{entity_name}` from {len(chunk_ids)} chunks"
                    )
                    logger.info(status_message)
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)
                except Exception as e:
                    failed_entities_count += 1
                    status_message = f"Failed to rebuild `{entity_name}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    async def _locked_rebuild_relationship(src, tgt, chunk_ids):
        nonlocal rebuilt_relationships_count, failed_relationships_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            # Sort src and tgt to ensure order-independent lock key generation
            sorted_key_parts = sorted([src, tgt])
            async with get_storage_keyed_lock(
                sorted_key_parts,
                namespace=namespace,
                enable_logging=False,
            ):
                try:
                    await _rebuild_single_relationship(
                        knowledge_graph_inst=knowledge_graph_inst,
                        relationships_vdb=relationships_vdb,
                        src=src,
                        tgt=tgt,
                        chunk_ids=chunk_ids,
                        chunk_relationships=chunk_relationships,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                    )
                    rebuilt_relationships_count += 1
                    status_message = (
                        f"Rebuilt `{src} - {tgt}` from {len(chunk_ids)} chunks"
                    )
                    logger.info(status_message)
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)
                except Exception as e:
                    failed_relationships_count += 1
                    status_message = f"Failed to rebuild `{src} - {tgt}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info
                    if pipeline_status is not None and pipeline_status_lock is not None:
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = status_message
                            pipeline_status["history_messages"].append(status_message)

    # Create tasks for parallel processing
    tasks = []

    # Add entity rebuilding tasks
    for entity_name, chunk_ids in entities_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_entity(entity_name, chunk_ids))
        tasks.append(task)

    # Add relationship rebuilding tasks
    for (src, tgt), chunk_ids in relationships_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_relationship(src, tgt, chunk_ids))
        tasks.append(task)

    # Log parallel processing start
    status_message = f"Starting parallel rebuild of {len(entities_to_rebuild)} entities and {len(relationships_to_rebuild)} relationships (async: {graph_max_async})"
    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)

    # Execute all tasks in parallel with semaphore control and early failure detection
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception
    for task in done:
        if task.exception():
            # If a task failed, cancel all pending tasks
            for pending_task in pending:
                pending_task.cancel()

            # Wait for cancellation to complete
            if pending:
                await asyncio.wait(pending)

            # Re-raise the exception to notify the caller
            raise task.exception()

    # Final status report
    status_message = f"KG rebuild completed: {rebuilt_entities_count} entities and {rebuilt_relationships_count} relationships rebuilt successfully."
    if failed_entities_count > 0 or failed_relationships_count > 0:
        status_message += f" Failed: {failed_entities_count} entities, {failed_relationships_count} relationships."

    logger.info(status_message)
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = status_message
            pipeline_status["history_messages"].append(status_message)


async def _get_cached_extraction_results(
    llm_response_cache: BaseKVStorage,
    chunk_ids: set[str],
    text_chunks_storage: BaseKVStorage,
) -> dict[str, list[str]]:
    """Get cached extraction results for specific chunk IDs

    This function retrieves cached LLM extraction results for the given chunk IDs and returns
    them sorted by creation time. The results are sorted at two levels:
    1. Individual extraction results within each chunk are sorted by create_time (earliest first)
    2. Chunks themselves are sorted by the create_time of their earliest extraction result

    Args:
        llm_response_cache: LLM response cache storage
        chunk_ids: Set of chunk IDs to get cached results for
        text_chunks_storage: Text chunks storage for retrieving chunk data and LLM cache references

    Returns:
        Dict mapping chunk_id -> list of extraction_result_text, where:
        - Keys (chunk_ids) are ordered by the create_time of their first extraction result
        - Values (extraction results) are ordered by create_time within each chunk
    """
    cached_results = {}

    # Collect all LLM cache IDs from chunks
    all_cache_ids = set()

    # Read from storage
    chunk_data_list = await text_chunks_storage.get_by_ids(list(chunk_ids))
    for chunk_data in chunk_data_list:
        if chunk_data and isinstance(chunk_data, dict):
            llm_cache_list = chunk_data.get("llm_cache_list", [])
            if llm_cache_list:
                all_cache_ids.update(llm_cache_list)
        else:
            logger.warning(f"Chunk data is invalid or None: {chunk_data}")

    if not all_cache_ids:
        logger.warning(f"No LLM cache IDs found for {len(chunk_ids)} chunk IDs")
        return cached_results

    # Batch get LLM cache entries
    cache_data_list = await llm_response_cache.get_by_ids(list(all_cache_ids))

    # Process cache entries and group by chunk_id
    valid_entries = 0
    for cache_entry in cache_data_list:
        if (
            cache_entry is not None
            and isinstance(cache_entry, dict)
            and cache_entry.get("cache_type") == "extract"
            and cache_entry.get("chunk_id") in chunk_ids
        ):
            chunk_id = cache_entry["chunk_id"]
            extraction_result = cache_entry["return"]
            create_time = cache_entry.get(
                "create_time", 0
            )  # Get creation time, default to 0
            valid_entries += 1

            # Support multiple LLM caches per chunk
            if chunk_id not in cached_results:
                cached_results[chunk_id] = []
            # Store tuple with extraction result and creation time for sorting
            cached_results[chunk_id].append((extraction_result, create_time))

    # Sort extraction results by create_time for each chunk and collect earliest times
    chunk_earliest_times = {}
    for chunk_id in cached_results:
        # Sort by create_time (x[1]), then extract only extraction_result (x[0])
        cached_results[chunk_id].sort(key=lambda x: x[1])
        # Store the earliest create_time for this chunk (first item after sorting)
        chunk_earliest_times[chunk_id] = cached_results[chunk_id][0][1]
        # Extract only extraction_result (x[0])
        cached_results[chunk_id] = [item[0] for item in cached_results[chunk_id]]

    # Sort cached_results by the earliest create_time of each chunk
    sorted_chunk_ids = sorted(
        chunk_earliest_times.keys(), key=lambda chunk_id: chunk_earliest_times[chunk_id]
    )

    # Rebuild cached_results in sorted order
    sorted_cached_results = {}
    for chunk_id in sorted_chunk_ids:
        sorted_cached_results[chunk_id] = cached_results[chunk_id]

    logger.info(
        f"Found {valid_entries} valid cache entries, {len(sorted_cached_results)} chunks with results"
    )
    return sorted_cached_results


async def _retry_malformed_extraction(
    original_content: str,
    chunk_key: str,
    file_path: str,
    timestamp: str,
    malformed_records: list[str],
    use_llm_func: callable,
    context_base: dict,
    llm_response_cache=None,
    max_retries: int = 2,
) -> tuple[dict, dict]:
    """Retry extraction for malformed records using complete prompt.

    Args:
        original_content: The original chunk content
        chunk_key: The chunk key for logging
        file_path: File path for source attribution
        timestamp: Timestamp for source attribution
        malformed_records: List of malformed record strings
        use_llm_func: LLM function to use
        context_base: Context for prompt formatting
        llm_response_cache: Cache for LLM responses
        max_retries: Maximum number of retry attempts

    Returns:
        tuple: (nodes_dict, edges_dict) containing successfully extracted entities and relationships
    """
    from lightrag.prompt import PROMPTS

    retry_nodes = defaultdict(list)
    retry_edges = defaultdict(list)

    if not malformed_records:
        return dict(retry_nodes), dict(retry_edges)

    # Format the complete entity extraction prompt for retry
    entity_extract_prompt = PROMPTS["entity_extraction"]

    retry_prompt_text = entity_extract_prompt.format(
        **{**context_base, "input_text": original_content, "timestamp": timestamp}
    ) + f"\n\nRETRY CONTEXT:\nYour previous extraction contained malformed records: {malformed_records}\nPlease re-extract ALL entities and relationships from the text above, ensuring correct format."

    for attempt in range(max_retries):
        try:
            # Retry attempt in progress

            # Use LLM function with cache
            retry_result = await use_llm_func_with_cache(
                retry_prompt_text,
                use_llm_func,
                llm_response_cache=llm_response_cache,
                cache_type="extract_retry",
                chunk_id=f"{chunk_key}_retry_{attempt}",
            )

            # Process the retry result
            retry_nodes_attempt, retry_edges_attempt = await _process_extraction_result(
                retry_result,
                f"{chunk_key}_retry_{attempt}",
                file_path,
                timestamp,
                tuple_delimiter=context_base["tuple_delimiter"],
                record_delimiter=context_base["record_delimiter"],
                completion_delimiter=context_base["completion_delimiter"],
                is_retry=True,  # Flag to prevent infinite retry loops
            )

            # If we got successful results, merge them
            if retry_nodes_attempt or retry_edges_attempt:
                for entity_name, entities in retry_nodes_attempt.items():
                    if entity_name == "__MALFORMED_RECORDS__":
                        continue  # Skip special malformed records key
                    retry_nodes[entity_name].extend(entities)
                for edge_key, edges in retry_edges_attempt.items():
                    retry_edges[edge_key].extend(edges)

                logger.info(f"Retry successful: recovered {len(retry_nodes_attempt)} entities + {len(retry_edges_attempt)} relationships")
                break

        except Exception as e:
            logger.warning(f"Retry attempt {attempt + 1} failed for {chunk_key}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All retry attempts failed for {chunk_key}")

    return dict(retry_nodes), dict(retry_edges)


async def _process_extraction_result(
    result: str,
    chunk_key: str,
    file_path: str = "unknown_source",
    timestamp: str = "",
    tuple_delimiter: str = "<|>",
    record_delimiter: str = "##",
    completion_delimiter: str = "<|COMPLETE|>",
    is_retry: bool = False,
) -> tuple[dict, dict]:
    """Process a single extraction result (either initial or gleaning)
    Args:
        result (str): The extraction result to process
        chunk_key (str): The chunk key for source tracking
        file_path (str): The file path for citation
        tuple_delimiter (str): Delimiter for tuple fields
        record_delimiter (str): Delimiter for records
        completion_delimiter (str): Delimiter for completion
    Returns:
        tuple: (nodes_dict, edges_dict) containing the extracted entities and relationships
    """
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    malformed_records = []  # Collect malformed records for retry

    # Standardize Chinese brackets around record_delimiter to English brackets
    bracket_pattern = f"[）)](\\s*{re.escape(record_delimiter)}\\s*)[（(]"
    result = re.sub(bracket_pattern, ")\\1(", result)

    records = split_string_by_multi_markers(
        result,
        [record_delimiter, completion_delimiter],
    )

    for record in records:
        # Remove outer brackets (support English and Chinese brackets)
        record = record.strip()
        if record.startswith("(") or record.startswith("（"):
            record = record[1:]
        if record.endswith(")") or record.endswith("）"):
            record = record[:-1]

        record = record.strip()
        if record is None:
            continue

        if tuple_delimiter == "<|>":
            # fix entity<| with entity<|>
            record = re.sub(r"^entity<\|(?!>)", r"entity<|>", record)
            # fix relationship<| with relationship<|>
            record = re.sub(r"^relationship<\|(?!>)", r"relationship<|>", record)
            # fix <||> with <|>
            record = record.replace("<||>", "<|>")
            # fix  < | > with <|>
            record = record.replace("< | >", "<|>")
            # fix <<|>> with <|>
            record = record.replace("<<|>>", "<|>")
            # fix <|>> with <|>
            record = record.replace("<|>>", "<|>")
            # fix <<|> with <|>
            record = record.replace("<<|>", "<|>")

        record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])

        # Try to parse as entity
        entity_data = await _handle_single_entity_extraction(
            record_attributes, chunk_key, file_path, timestamp
        )
        if entity_data == "MALFORMED_ENTITY":
            # Silently collect malformed entity record for retry
            malformed_records.append(record)
            continue
        elif entity_data is not None:
            maybe_nodes[entity_data["entity_name"]].append(entity_data)
            continue

        # Try to parse as relationship
        relationship_data = await _handle_single_relationship_extraction(
            record_attributes, chunk_key, file_path, timestamp
        )
        if relationship_data == "MALFORMED_RELATIONSHIP":
            # Silently collect malformed relationship record for retry
            malformed_records.append(record)
            continue
        elif relationship_data is not None:
            maybe_edges[
                (relationship_data["src_id"], relationship_data["tgt_id"])
            ].append(relationship_data)

    # If we have malformed records and this is not already a retry, store them for retry
    if malformed_records and not is_retry:
        # Store malformed records in a way that can be accessed for retry
        maybe_nodes["__MALFORMED_RECORDS__"] = malformed_records

    return dict(maybe_nodes), dict(maybe_edges)


async def _parse_extraction_result(
    text_chunks_storage: BaseKVStorage, extraction_result: str, chunk_id: str
) -> tuple[dict, dict]:
    """Parse cached extraction result using the same logic as extract_entities

    Args:
        text_chunks_storage: Text chunks storage to get chunk data
        extraction_result: The cached LLM extraction result
        chunk_id: The chunk ID for source tracking

    Returns:
        Tuple of (entities_dict, relationships_dict)
    """

    # Get chunk data for file_path and timestamp from storage
    chunk_data = await text_chunks_storage.get_by_id(chunk_id)
    file_path = (
        chunk_data.get("file_path", "unknown_source")
        if chunk_data
        else "unknown_source"
    )
    timestamp = (
        chunk_data.get("timestamp", "")
        if chunk_data
        else ""
    )

    # Call the shared processing function
    return await _process_extraction_result(
        extraction_result,
        chunk_id,
        file_path,
        timestamp,
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    )


async def _rebuild_single_entity(
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name: str,
    chunk_ids: set[str],
    chunk_entities: dict,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
) -> None:
    """Rebuild a single entity from cached extraction results"""

    # Get current entity data
    current_entity = await knowledge_graph_inst.get_node(entity_name)
    if not current_entity:
        return

    # Helper function to update entity in both graph and vector storage
    async def _update_entity_storage(
        final_description: str, entity_type: str, file_paths: set[str]
    ):
        # Update entity in graph storage
        updated_entity_data = {
            **current_entity,
            "description": final_description,
            "entity_type": entity_type,
            "source_id": GRAPH_FIELD_SEP.join(chunk_ids),
            "file_path": GRAPH_FIELD_SEP.join(file_paths)
            if file_paths
            else current_entity.get("file_path", "unknown_source"),
        }
        await knowledge_graph_inst.upsert_node(entity_name, updated_entity_data)

        # Update entity in vector database
        entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")

        entity_content = f"{entity_name}\n{final_description}"
        await entities_vdb.upsert(
            {
                entity_vdb_id: {
                    "content": entity_content,
                    "entity_name": entity_name,
                    "source_id": updated_entity_data["source_id"],
                    "description": final_description,
                    "entity_type": entity_type,
                    "file_path": updated_entity_data["file_path"],
                }
            }
        )

    # Collect all entity data from relevant chunks
    all_entity_data = []
    for chunk_id in chunk_ids:
        if chunk_id in chunk_entities and entity_name in chunk_entities[chunk_id]:
            all_entity_data.extend(chunk_entities[chunk_id][entity_name])

    if not all_entity_data:
        logger.warning(
            f"No entity data found for `{entity_name}`, trying to rebuild from relationships"
        )

        # Get all edges connected to this entity
        edges = await knowledge_graph_inst.get_node_edges(entity_name)
        if not edges:
            logger.warning(f"No relations attached to entity `{entity_name}`")
            return

        # Collect relationship data to extract entity information
        relationship_descriptions = []
        file_paths = set()

        # Get edge data for all connected relationships
        for src_id, tgt_id in edges:
            edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)
            if edge_data:
                if edge_data.get("description"):
                    relationship_descriptions.append(edge_data["description"])

                if edge_data.get("file_path"):
                    edge_file_paths = edge_data["file_path"].split(GRAPH_FIELD_SEP)
                    file_paths.update(edge_file_paths)

        # deduplicate descriptions
        description_list = list(dict.fromkeys(relationship_descriptions))

        # Generate final description from relationships or fallback to current
        if description_list:
            final_description, _ = await _handle_entity_relation_summary(
                "Entity",
                entity_name,
                description_list,
                GRAPH_FIELD_SEP,
                global_config,
                llm_response_cache=llm_response_cache,
            )
        else:
            final_description = current_entity.get("description", "")

        entity_type = current_entity.get("entity_type", "UNKNOWN")
        await _update_entity_storage(final_description, entity_type, file_paths)
        return

    # Process cached entity data
    descriptions = []
    entity_types = []
    file_paths = set()

    for entity_data in all_entity_data:
        if entity_data.get("description"):
            descriptions.append(entity_data["description"])
        if entity_data.get("entity_type"):
            entity_types.append(entity_data["entity_type"])
        if entity_data.get("file_path"):
            file_paths.add(entity_data["file_path"])

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    entity_types = list(dict.fromkeys(entity_types))

    # Get most common entity type
    entity_type = (
        max(set(entity_types), key=entity_types.count)
        if entity_types
        else current_entity.get("entity_type", "UNKNOWN")
    )

    # Generate final description from entities or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Entity",
            entity_name,
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        final_description = current_entity.get("description", "")

    await _update_entity_storage(final_description, entity_type, file_paths)


async def _rebuild_single_relationship(
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    src: str,
    tgt: str,
    chunk_ids: set[str],
    chunk_relationships: dict,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
) -> None:
    """Rebuild a single relationship from cached extraction results

    Note: This function assumes the caller has already acquired the appropriate
    keyed lock for the relationship pair to ensure thread safety.
    """

    # Get current relationship data
    current_relationship = await knowledge_graph_inst.get_edge(src, tgt)
    if not current_relationship:
        return

    # Collect all relationship data from relevant chunks
    all_relationship_data = []
    for chunk_id in chunk_ids:
        if chunk_id in chunk_relationships:
            # Check both (src, tgt) and (tgt, src) since relationships can be bidirectional
            for edge_key in [(src, tgt), (tgt, src)]:
                if edge_key in chunk_relationships[chunk_id]:
                    all_relationship_data.extend(
                        chunk_relationships[chunk_id][edge_key]
                    )

    if not all_relationship_data:
        logger.warning(f"No relation data found for `{src}-{tgt}`")
        return

    # Merge descriptions and keywords
    descriptions = []
    keywords = []
    weights = []
    file_paths = set()

    for rel_data in all_relationship_data:
        if rel_data.get("description"):
            descriptions.append(rel_data["description"])
        if rel_data.get("keywords"):
            keywords.append(rel_data["keywords"])
        if rel_data.get("weight"):
            weights.append(rel_data["weight"])
        if rel_data.get("file_path"):
            file_paths.add(rel_data["file_path"])

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    keywords = list(dict.fromkeys(keywords))

    combined_keywords = (
        ", ".join(set(keywords))
        if keywords
        else current_relationship.get("keywords", "")
    )

    weight = sum(weights) if weights else current_relationship.get("weight", 1.0)

    # Generate final description from relations or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Relation",
            f"{src}-{tgt}",
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        # fallback to keep current(unchanged)
        final_description = current_relationship.get("description", "")

    # Update relationship in graph storage
    updated_relationship_data = {
        **current_relationship,
        "description": final_description
        if final_description
        else current_relationship.get("description", ""),
        "keywords": combined_keywords,
        "weight": weight,
        "source_id": GRAPH_FIELD_SEP.join(chunk_ids),
        "file_path": GRAPH_FIELD_SEP.join([fp for fp in file_paths if fp])
        if file_paths
        else current_relationship.get("file_path", "unknown_source"),
    }
    await knowledge_graph_inst.upsert_edge(src, tgt, updated_relationship_data)

    # Update relationship in vector database
    rel_vdb_id = compute_mdhash_id(src + tgt, prefix="rel-")
    rel_vdb_id_reverse = compute_mdhash_id(tgt + src, prefix="rel-")

    # Delete old vector records first (both directions to be safe)
    try:
        await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
    except Exception as e:
        logger.debug(
            f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
        )

    # Insert new vector record
    rel_content = f"{combined_keywords}\t{src}\n{tgt}\n{final_description}"
    await relationships_vdb.upsert(
        {
            rel_vdb_id: {
                "src_id": src,
                "tgt_id": tgt,
                "source_id": updated_relationship_data["source_id"],
                "content": rel_content,
                "keywords": combined_keywords,
                "description": final_description,
                "weight": weight,
                "file_path": updated_relationship_data["file_path"],
            }
        }
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(already_node["source_id"].split(GRAPH_FIELD_SEP))
        already_file_paths.extend(already_node["file_path"].split(GRAPH_FIELD_SEP))
        already_description.extend(already_node["description"].split(GRAPH_FIELD_SEP))

    # Filter out invalid data and extract entity types safely
    valid_nodes_data = [dp for dp in nodes_data if isinstance(dp, dict) and "entity_type" in dp]

    if not valid_nodes_data and not already_entity_types:
        logger.warning(f"No valid entity type found for {entity_name}, using 'unknown'")
        entity_type = "unknown"
    else:
        entity_type = sorted(
            Counter(
                [dp["entity_type"] for dp in valid_nodes_data] + already_entity_types
            ).items(),
            key=lambda x: x[1],
            reverse=True,
        )[0][0]  # Get the entity type with the highest count

    # merge and deduplicate description using valid nodes data
    description_list = list(
        dict.fromkeys(
            already_description
            + [dp["description"] for dp in valid_nodes_data if dp.get("description")]
        )
    )

    num_fragment = len(description_list)
    already_fragment = len(already_description)
    deduplicated_num = already_fragment + len(valid_nodes_data) - num_fragment
    if deduplicated_num > 0:
        dd_message = f"(dd:{deduplicated_num})"
    else:
        dd_message = ""
    if num_fragment > 0:
        # Get summary and LLM usage status
        description, llm_was_used = await _handle_entity_relation_summary(
            "Entity",
            entity_name,
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache,
        )

        # Log based on actual LLM usage
        if llm_was_used:
            status_message = f"LLMmrg: `{entity_name}` | {already_fragment}+{num_fragment-already_fragment}{dd_message}"
        else:
            status_message = f"Merged: `{entity_name}` | {already_fragment}+{num_fragment-already_fragment}{dd_message}"

        logger.info(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
    else:
        logger.error(f"Entity {entity_name} has no description")
        description = "(no description)"

    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in valid_nodes_data] + already_source_ids)
    )
    file_path = build_file_path(already_file_paths, valid_nodes_data, entity_name)

    node_data = dict(
        entity_id=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        file_path=file_path,
        created_at=int(time.time()),
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    added_entities: list = None,  # New parameter to track entities added during edge processing
):
    if src_id == tgt_id:
        return None

    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_file_paths = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            # Get weight with default 1.0 if missing
            already_weights.append(already_edge.get("weight", 1.0))

            # Get source_id with empty string default if missing or None
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    already_edge["source_id"].split(GRAPH_FIELD_SEP)
                )

            # Get file_path with empty string default if missing or None
            if already_edge.get("file_path") is not None:
                already_file_paths.extend(
                    already_edge["file_path"].split(GRAPH_FIELD_SEP)
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.extend(
                    already_edge["description"].split(GRAPH_FIELD_SEP)
                )

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )

    # Process edges_data with None checks
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)

    description_list = list(
        dict.fromkeys(
            already_description
            + [dp["description"] for dp in edges_data if dp.get("description")]
        )
    )

    num_fragment = len(description_list)
    already_fragment = len(already_description)
    deduplicated_num = already_fragment + len(edges_data) - num_fragment
    if deduplicated_num > 0:
        dd_message = f"(dd:{deduplicated_num})"
    else:
        dd_message = ""
    if num_fragment > 0:
        # Get summary and LLM usage status
        description, llm_was_used = await _handle_entity_relation_summary(
            "Relation",
            f"({src_id}, {tgt_id})",
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache,
        )

        # Log based on actual LLM usage
        if llm_was_used:
            status_message = f"LLMmrg: `{src_id} - {tgt_id}` | {already_fragment}+{num_fragment-already_fragment}{dd_message}"
        else:
            status_message = f"Merged: `{src_id} - {tgt_id}` | {already_fragment}+{num_fragment-already_fragment}{dd_message}"

        logger.info(status_message)
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = status_message
                pipeline_status["history_messages"].append(status_message)
    else:
        logger.error(f"Edge {src_id} - {tgt_id} has no description")
        description = "(no description)"

    # Split all existing and new keywords into individual terms, then combine and deduplicate
    all_keywords = set()
    # Process already_keywords (which are comma-separated)
    for keyword_str in already_keywords:
        if keyword_str:  # Skip empty strings
            all_keywords.update(k.strip() for k in keyword_str.split(",") if k.strip())
    # Process new keywords from edges_data
    for edge in edges_data:
        if edge.get("keywords"):
            all_keywords.update(
                k.strip() for k in edge["keywords"].split(",") if k.strip()
            )
    # Join all unique keywords with commas
    keywords = ",".join(sorted(all_keywords))

    source_id = GRAPH_FIELD_SEP.join(
        set(
            [dp["source_id"] for dp in edges_data if dp.get("source_id")]
            + already_source_ids
        )
    )
    file_path = build_file_path(already_file_paths, edges_data, f"{src_id}-{tgt_id}")

    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            node_data = {
                "entity_id": need_insert_id,
                "source_id": source_id,
                "description": description,
                "entity_type": "UNKNOWN",
                "file_path": file_path,
                "created_at": int(time.time()),
            }
            await knowledge_graph_inst.upsert_node(need_insert_id, node_data=node_data)

            # Track entities added during edge processing
            if added_entities is not None:
                entity_data = {
                    "entity_name": need_insert_id,
                    "entity_type": "UNKNOWN",
                    "description": description,
                    "source_id": source_id,
                    "file_path": file_path,
                    "created_at": int(time.time()),
                }
                added_entities.append(entity_data)

    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            file_path=file_path,
            created_at=int(time.time()),
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
        created_at=int(time.time()),
    )

    return edge_data


async def merge_nodes_and_edges(
    chunk_results: list,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    full_entities_storage: BaseKVStorage = None,
    full_relations_storage: BaseKVStorage = None,
    doc_id: str = None,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    current_file_number: int = 0,
    total_files: int = 0,
    file_path: str = "unknown_source",
) -> None:
    """Two-phase merge: process all entities first, then all relationships

    This approach ensures data consistency by:
    1. Phase 1: Process all entities concurrently
    2. Phase 2: Process all relationships concurrently (may add missing entities)
    3. Phase 3: Update full_entities and full_relations storage with final results

    Args:
        chunk_results: List of tuples (maybe_nodes, maybe_edges) containing extracted entities and relationships
        knowledge_graph_inst: Knowledge graph storage
        entity_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration
        full_entities_storage: Storage for document entity lists
        full_relations_storage: Storage for document relation lists
        doc_id: Document ID for storage indexing
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        llm_response_cache: LLM response cache
        current_file_number: Current file number for logging
        total_files: Total files for logging
        file_path: File path for logging
    """

    # Collect all nodes and edges from all chunks
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)

    for maybe_nodes, maybe_edges in chunk_results:
        # Collect nodes
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)

        # Collect edges with sorted keys for undirected graph
        for edge_key, edges in maybe_edges.items():
            sorted_edge_key = tuple(sorted(edge_key))
            all_edges[sorted_edge_key].extend(edges)

    total_entities_count = len(all_nodes)
    total_relations_count = len(all_edges)

    log_message = f"Merging stage {current_file_number}/{total_files}: {file_path}"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # ===== Phase 1: Process all entities concurrently =====
    log_message = f"Phase 1: Processing {total_entities_count} entities from {doc_id} (async: {graph_max_async})"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    async def _locked_process_entity_name(entity_name, entities):
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                entity_data = await _merge_nodes_then_upsert(
                    entity_name,
                    entities,
                    knowledge_graph_inst,
                    global_config,
                    pipeline_status,
                    pipeline_status_lock,
                    llm_response_cache,
                )
                if entity_vdb is not None:
                    data_for_vdb = {
                        compute_mdhash_id(entity_data["entity_name"], prefix="ent-"): {
                            "entity_name": entity_data["entity_name"],
                            "entity_type": entity_data["entity_type"],
                            "content": f"{entity_data['entity_name']}\n{entity_data['description']}",
                            "source_id": entity_data["source_id"],
                            "file_path": entity_data.get("file_path", "unknown_source"),
                        }
                    }
                    await entity_vdb.upsert(data_for_vdb)
                return entity_data

    # Create entity processing tasks
    entity_tasks = []
    for entity_name, entities in all_nodes.items():
        task = asyncio.create_task(_locked_process_entity_name(entity_name, entities))
        entity_tasks.append(task)

    # Execute entity tasks with error handling
    processed_entities = []
    if entity_tasks:
        done, pending = await asyncio.wait(
            entity_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        # Check if any task raised an exception
        for task in done:
            if task.exception():
                # If a task failed, cancel all pending tasks
                for pending_task in pending:
                    pending_task.cancel()
                # Wait for cancellation to complete
                if pending:
                    await asyncio.wait(pending)
                # Re-raise the exception to notify the caller
                raise task.exception()

        # If all tasks completed successfully, collect results
        processed_entities = [task.result() for task in entity_tasks]

    # ===== Phase 2: Process all relationships concurrently =====
    log_message = f"Phase 2: Processing {total_relations_count} relations from {doc_id} (async: {graph_max_async})"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    async def _locked_process_edges(edge_key, edges):
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            sorted_edge_key = sorted([edge_key[0], edge_key[1]])

            async with get_storage_keyed_lock(
                sorted_edge_key,
                namespace=namespace,
                enable_logging=False,
            ):
                added_entities = []  # Track entities added during edge processing
                edge_data = await _merge_edges_then_upsert(
                    edge_key[0],
                    edge_key[1],
                    edges,
                    knowledge_graph_inst,
                    global_config,
                    pipeline_status,
                    pipeline_status_lock,
                    llm_response_cache,
                    added_entities,  # Pass list to collect added entities
                )

                if edge_data is None:
                    return None, []

                if relationships_vdb is not None:
                    data_for_vdb = {
                        compute_mdhash_id(
                            edge_data["src_id"] + edge_data["tgt_id"], prefix="rel-"
                        ): {
                            "src_id": edge_data["src_id"],
                            "tgt_id": edge_data["tgt_id"],
                            "keywords": edge_data["keywords"],
                            "content": f"{edge_data['src_id']}\t{edge_data['tgt_id']}\n{edge_data['keywords']}\n{edge_data['description']}",
                            "source_id": edge_data["source_id"],
                            "file_path": edge_data.get("file_path", "unknown_source"),
                            "weight": edge_data.get("weight", 1.0),
                        }
                    }
                    await relationships_vdb.upsert(data_for_vdb)
                return edge_data, added_entities

    # Create relationship processing tasks
    edge_tasks = []
    for edge_key, edges in all_edges.items():
        task = asyncio.create_task(_locked_process_edges(edge_key, edges))
        edge_tasks.append(task)

    # Execute relationship tasks with error handling
    processed_edges = []
    all_added_entities = []

    if edge_tasks:
        done, pending = await asyncio.wait(
            edge_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        # Check if any task raised an exception
        for task in done:
            if task.exception():
                # If a task failed, cancel all pending tasks
                for pending_task in pending:
                    pending_task.cancel()
                # Wait for cancellation to complete
                if pending:
                    await asyncio.wait(pending)
                # Re-raise the exception to notify the caller
                raise task.exception()

        # If all tasks completed successfully, collect results
        for task in edge_tasks:
            edge_data, added_entities = task.result()
            if edge_data is not None:
                processed_edges.append(edge_data)
            all_added_entities.extend(added_entities)

    # ===== Phase 3: Update full_entities and full_relations storage =====
    if full_entities_storage and full_relations_storage and doc_id:
        try:
            # Merge all entities: original entities + entities added during edge processing
            final_entity_names = set()

            # Add original processed entities
            for entity_data in processed_entities:
                if entity_data and entity_data.get("entity_name"):
                    final_entity_names.add(entity_data["entity_name"])

            # Add entities that were added during relationship processing
            for added_entity in all_added_entities:
                if added_entity and added_entity.get("entity_name"):
                    final_entity_names.add(added_entity["entity_name"])

            # Collect all relation pairs
            final_relation_pairs = set()
            for edge_data in processed_edges:
                if edge_data:
                    src_id = edge_data.get("src_id")
                    tgt_id = edge_data.get("tgt_id")
                    if src_id and tgt_id:
                        relation_pair = tuple(sorted([src_id, tgt_id]))
                        final_relation_pairs.add(relation_pair)

            log_message = f"Phase 3: Updating final {len(final_entity_names)}({len(processed_entities)}+{len(all_added_entities)}) entities and  {len(final_relation_pairs)} relations from {doc_id}"
            logger.info(log_message)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

            # Update storage
            if final_entity_names:
                await full_entities_storage.upsert(
                    {
                        doc_id: {
                            "entity_names": list(final_entity_names),
                            "count": len(final_entity_names),
                        }
                    }
                )

            if final_relation_pairs:
                await full_relations_storage.upsert(
                    {
                        doc_id: {
                            "relation_pairs": [
                                list(pair) for pair in final_relation_pairs
                            ],
                            "count": len(final_relation_pairs),
                        }
                    }
                )

            logger.debug(
                f"Updated entity-relation index for document {doc_id}: {len(final_entity_names)} entities (original: {len(processed_entities)}, added: {len(all_added_entities)}), {len(final_relation_pairs)} relations"
            )

        except Exception as e:
            logger.error(
                f"Failed to update entity-relation index for document {doc_id}: {e}"
            )
            # Don't raise exception to avoid affecting main flow

    log_message = f"Completed merging: {len(processed_entities)} entities, {len(all_added_entities)} extra entities, {len(processed_edges)} relations"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)
        
    # Compute and store enhanced embeddings if node embedding is enabled
    if global_config.get("enable_node_embedding", False) and global_config.get("node_embedding"):
        try:
            # Prepare entities and relations data for graph construction
            entities_data = []
            relations_data = []
            text_embeddings = {}
            
            # Extract entity data from processed_entities (list of entity data)
            for entity_data in processed_entities:
                if entity_data:
                    entity_name = entity_data.get('entity_name', '')
                    if entity_name:
                        entities_data.append({
                            'entity_name': entity_name,
                            'description': entity_data.get('description', ''),
                            'entity_type': entity_data.get('entity_type', 'unknown')
                        })
                        
                        # Get text embedding if available
                        if hasattr(entity_data, 'get') and entity_data.get('embedding'):
                            text_embeddings[entity_name] = entity_data['embedding']
            
            # Extract relation data from processed_edges (list of edge data)
            for edge_data in processed_edges:
                if edge_data:
                    relations_data.append({
                        'src_id': edge_data.get('src_id', ''),
                        'tgt_id': edge_data.get('tgt_id', ''),
                        'description': edge_data.get('description', ''),
                        'keywords': edge_data.get('keywords', '')
                    })
                        
        except Exception as e:
            logger.error(f"Error computing enhanced embeddings: {e}")
            # Don't raise exception to avoid affecting main flow


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    text_chunks_storage: BaseKVStorage | None = None,
) -> list:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    entity_extract_max_retries = global_config.get("entity_extract_max_retries", 2)

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)
    entity_types = global_config["addon_params"].get(
        "entity_types", DEFAULT_ENTITY_TYPES
    )

    examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entity_continue_extraction"].format(**context_base)

    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        Returns:
            tuple: (maybe_nodes, maybe_edges) containing extracted entities and relationships
        """
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # Get file path from chunk data or use default
        file_path = chunk_dp.get("file_path", "unknown_source")
        # Get timestamp from chunk data
        timestamp = chunk_dp.get("timestamp", "")

        # Create cache keys collector for batch processing
        cache_keys_collector = []

        # Get initial extraction
        hint_prompt = entity_extract_prompt.format(
            **{**context_base, "input_text": content, "timestamp": timestamp}
        )

        final_result = await use_llm_func_with_cache(
            hint_prompt,
            use_llm_func,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            cache_keys_collector=cache_keys_collector,
        )

        # Store LLM cache reference in chunk (will be handled by use_llm_func_with_cache)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

        # Process initial extraction with file path and timestamp
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result,
            chunk_key,
            file_path,
            timestamp,
            tuple_delimiter=context_base["tuple_delimiter"],
            record_delimiter=context_base["record_delimiter"],
            completion_delimiter=context_base["completion_delimiter"],
        )

        # Check for malformed records and retry if needed
        malformed_records = maybe_nodes.pop("__MALFORMED_RECORDS__", [])
        if malformed_records:
            logger.info(f"Retrying {len(malformed_records)} malformed records in {chunk_key}")
            retry_nodes, retry_edges = await _retry_malformed_extraction(
                content,  # original chunk content
                chunk_key,
                file_path,
                timestamp,
                malformed_records,
                use_llm_func,
                context_base,
                llm_response_cache=llm_response_cache,
                max_retries=entity_extract_max_retries,
            )

            # Merge retry results with initial results (exclude special keys)
            for entity_name, entities in retry_nodes.items():
                if entity_name == "__MALFORMED_RECORDS__":
                    continue  # Skip special malformed records key
                if entity_name not in maybe_nodes:
                    maybe_nodes[entity_name] = []
                maybe_nodes[entity_name].extend(entities)

            for edge_key, edges in retry_edges.items():
                if edge_key not in maybe_edges:
                    maybe_edges[edge_key] = []
                maybe_edges[edge_key].extend(edges)

        # Process additional gleaning results
        if entity_extract_max_gleaning > 0:
            glean_result = await use_llm_func_with_cache(
                continue_prompt,
                use_llm_func,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
                chunk_id=chunk_key,
                cache_keys_collector=cache_keys_collector,
            )

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)

            # Process gleaning result separately with file path and timestamp
            glean_nodes, glean_edges = await _process_extraction_result(
                glean_result,
                chunk_key,
                file_path,
                timestamp,
                tuple_delimiter=context_base["tuple_delimiter"],
                record_delimiter=context_base["record_delimiter"],
                completion_delimiter=context_base["completion_delimiter"],
                is_retry=True,  # Prevent malformed records check in gleaning stage
            )

            # Merge results - only add entities and edges with new names
            for entity_name, entities in glean_nodes.items():
                if (
                    entity_name not in maybe_nodes
                ):  # Only accetp entities with new name in gleaning stage
                    maybe_nodes[entity_name] = []  # Explicitly create the list
                    maybe_nodes[entity_name].extend(entities)
            for edge_key, edges in glean_edges.items():
                if (
                    edge_key not in maybe_edges
                ):  # Only accetp edges with new name in gleaning stage
                    maybe_edges[edge_key] = []  # Explicitly create the list
                    maybe_edges[edge_key].extend(edges)

        # Batch update chunk's llm_cache_list with all collected cache keys
        if cache_keys_collector and text_chunks_storage:
            await update_chunk_cache_list(
                chunk_key,
                text_chunks_storage,
                cache_keys_collector,
                "entity_extraction",
            )

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # Return the extracted nodes and edges for centralized processing
        return maybe_nodes, maybe_edges

    # Get max async tasks limit from global_config
    chunk_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = asyncio.Semaphore(chunk_max_async)

    async def _process_with_semaphore(chunk):
        async with semaphore:
            return await _process_single_content(chunk)

    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    # Wait for tasks to complete or for the first exception to occur
    # This allows us to cancel remaining tasks if any task fails
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception
    for task in done:
        if task.exception():
            # If a task failed, cancel all pending tasks
            # This prevents unnecessary processing since the parent function will abort anyway
            for pending_task in pending:
                pending_task.cancel()

            # Wait for cancellation to complete
            if pending:
                await asyncio.wait(pending)

            # Re-raise the exception to notify the caller
            raise task.exception()

    # If all tasks completed successfully, collect results
    chunk_results = [task.result() for task in tasks]

    # Return the chunk_results for later processing in merge_nodes_and_edges
    return chunk_results


async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    chunks_vdb: BaseVectorStorage = None,
) -> tuple[str, str] | tuple[AsyncIterator[str], str]:
    if not query:
        return PROMPTS["fail_response"], ""

    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    # Handle cache
    args_hash = compute_args_hash(
        query_param.mode,
        query,
        query_param.response_type,
        query_param.top_k,
        query_param.chunk_top_k,
        query_param.max_entity_tokens,
        query_param.max_relation_tokens,
        query_param.max_total_tokens,
        query_param.hl_keywords or [],
        query_param.ll_keywords or [],
        query_param.user_prompt or "",
        query_param.enable_rerank,
    )
    cached_response = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response, ""

    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
        logger.warning("low_level_keywords is empty")
    if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
        logger.warning("high_level_keywords is empty")
    if hl_keywords == [] and ll_keywords == []:
        if len(query) < 50:
            logger.warning(f"Forced low_level_keywords to origin query: {query}")
            ll_keywords = [query]
        else:
            return PROMPTS["fail_response"], ""

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build context
    context = await _build_query_context(
        query,
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        global_config,
        chunks_vdb,
    )

    if query_param.only_need_context:
        return (context if context is not None else PROMPTS["fail_response"]), ""
    if context is None:
        return PROMPTS["fail_response"], ""

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # Build system prompt
    user_prompt = (
        query_param.user_prompt
        if query_param.user_prompt
        else PROMPTS["DEFAULT_USER_PROMPT"]
    )
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
        user_prompt=user_prompt,
    )

    if query_param.only_need_prompt:
        return sys_prompt, context

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(
        f"[kg_query] Sending to LLM: {len_of_prompts:,} tokens (Query: {len(tokenizer.encode(query))}, System: {len(tokenizer.encode(sys_prompt))})"
    )

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    if hashing_kv.global_config.get("enable_llm_cache"):
        # Save to cache with query parameters
        queryparam_dict = {
            "mode": query_param.mode,
            "response_type": query_param.response_type,
            "top_k": query_param.top_k,
            "chunk_top_k": query_param.chunk_top_k,
            "max_entity_tokens": query_param.max_entity_tokens,
            "max_relation_tokens": query_param.max_relation_tokens,
            "max_total_tokens": query_param.max_total_tokens,
            "hl_keywords": query_param.hl_keywords or [],
            "ll_keywords": query_param.ll_keywords or [],
            "user_prompt": query_param.user_prompt or "",
            "enable_rerank": query_param.enable_rerank,
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                mode=query_param.mode,
                cache_type="query",
                queryparam=queryparam_dict,
            ),
        )

    return response, context


# Multi-hop retrieval functions

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
                # Use same format as relationship storage
                rel_key = f"{src_id}-{tgt_id}"
                rel_keys.append(rel_key)
                rel_vdb_ids.append(compute_mdhash_id(rel_key, prefix="rel-"))
        
        # Check if relationships_vdb supports direct vector access (NanoVectorDB)
        if hasattr(relationships_vdb, '_get_client'):
            # Direct access for NanoVectorDB
            client = await relationships_vdb._get_client()
            vdb_results = client.get(rel_vdb_ids)
            
            relationship_vectors = {}
            for i, vdb_data in enumerate(vdb_results):
                if vdb_data and 'vector' in vdb_data:
                    rel_key = rel_keys[i]
                    try:
                        # Decode NanoVectorDB format: base64 + zlib + float16
                        import base64
                        import numpy as np
                        import zlib
                        
                        compressed_vector = base64.b64decode(vdb_data['vector'])
                        vector_bytes = zlib.decompress(compressed_vector)
                        vector_array = np.frombuffer(vector_bytes, dtype=np.float16).astype(np.float32)
                        relationship_vectors[rel_key] = vector_array.tolist()
                    except Exception as e:
                        logger.debug(f"Error decoding vector for {rel_key}: {e}")
                        
            return relationship_vectors
        else:
            # Fallback: use regular get_by_ids (without vectors) - will still compute embeddings
            logger.debug("Vector storage doesn't support direct vector access, will compute embeddings")
            return {}
            
    except Exception as e:
        logger.debug(f"Error getting relationship vectors batch: {e}")
        return {}

async def _process_all_chunks_unified(
    entity_chunks: list[dict],
    relation_chunks: list[dict],
    entities_context: list[dict],
    relations_context: list[dict],
    query: str,
    text_chunks_db: BaseKVStorage,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    query_embedding = None,
) -> list[dict]:
    """
    Unified chunk processing: merge all chunks, deduplicate, sort by occurrence count,
    then select by vector similarity.

    This replaces the separated entity/relation chunk processing for true global optimization.
    """
    if not entity_chunks and not relation_chunks:
        return []

    # Get configuration
    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 1: Merge all chunks
    all_chunks = entity_chunks + relation_chunks
    logger.debug(f"Unified chunk processing: {len(entity_chunks)} entity + {len(relation_chunks)} relation = {len(all_chunks)} total chunks")

    # Step 2: Deduplicate and calculate occurrence count
    chunk_occurrence = {}
    deduplicated_chunks = {}

    for chunk in all_chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            if chunk_id not in deduplicated_chunks:
                deduplicated_chunks[chunk_id] = chunk
                chunk_occurrence[chunk_id] = 1
            else:
                chunk_occurrence[chunk_id] += 1

    # Step 3: Sort by occurrence count (most important first)
    sorted_chunks = sorted(
        deduplicated_chunks.values(),
        key=lambda x: chunk_occurrence.get(x.get("chunk_id"), 0),
        reverse=True
    )

    logger.debug(f"After deduplication: {len(sorted_chunks)} unique chunks, max occurrence: {max(chunk_occurrence.values()) if chunk_occurrence else 0}")

    # Step 4: Use chunk_top_k as the target chunk count
    # Note: related_chunk_number is now enforced per-entity/relation during collection
    # chunk_top_k is the final global limit
    target_chunks = query_param.chunk_top_k or 20

    # Ensure at least some chunks if we have any
    if len(sorted_chunks) > 0:
        target_chunks = max(1, target_chunks)


    # Step 5: Apply selection algorithm on occurrence-sorted chunks
    selected_chunks = []

    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        # Get embedding function
        embedding_func_config = text_chunks_db.embedding_func
        if not embedding_func_config:
            raise ValueError("No embedding function found for unified chunks sorting with VECTOR method")

        actual_embedding_func = embedding_func_config.func
        if not actual_embedding_func:
            raise ValueError("Embedding function is None for unified chunks sorting with VECTOR method")

        # Create mock structure for compatibility with pick_by_vector_similarity
        mock_entities_with_chunks = [{
            "chunks": [chunk.get("chunk_id") for chunk in sorted_chunks if chunk.get("chunk_id")],
            "sorted_chunks": [chunk.get("chunk_id") for chunk in sorted_chunks if chunk.get("chunk_id")]
        }]

        # Select chunks using vector similarity on occurrence-sorted chunks
        selected_chunk_ids = await pick_by_vector_similarity(
            query=query,
            text_chunks_storage=text_chunks_db,
            chunks_vdb=chunks_vdb,
            num_of_chunks=target_chunks,
            entity_info=mock_entities_with_chunks,
            embedding_func=actual_embedding_func,
            query_embedding=query_embedding,
        )

        if not selected_chunk_ids:
            raise RuntimeError("No chunks selected by vector similarity in unified processing - this should not happen with valid data")

        # Reorder chunks based on selection
        chunk_id_to_chunk = {chunk.get("chunk_id"): chunk for chunk in sorted_chunks if chunk.get("chunk_id")}
        selected_chunks = []
        for chunk_id in selected_chunk_ids:
            if chunk_id in chunk_id_to_chunk:
                selected_chunks.append(chunk_id_to_chunk[chunk_id])

    elif kg_chunk_pick_method == "WEIGHT":
        # Use weighted polling on occurrence-sorted chunks
        mock_entities_with_chunks = [{
            "chunks": [chunk.get("chunk_id") for chunk in sorted_chunks if chunk.get("chunk_id")],
            "sorted_chunks": [chunk.get("chunk_id") for chunk in sorted_chunks if chunk.get("chunk_id")]
        }]

        selected_chunk_ids = pick_by_weighted_polling(
            mock_entities_with_chunks, target_chunks, min_related_chunks=1
        )

        # Reorder chunks based on selection
        chunk_id_to_chunk = {chunk.get("chunk_id"): chunk for chunk in sorted_chunks if chunk.get("chunk_id")}
        selected_chunks = []
        for chunk_id in selected_chunk_ids:
            if chunk_id in chunk_id_to_chunk:
                selected_chunks.append(chunk_id_to_chunk[chunk_id])

    logger.info(f"Unified chunk selection completed: {len(selected_chunks)} chunks selected from {len(sorted_chunks)} candidates")
    return selected_chunks


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
    decay = 0.8 ** current_hop
    
    # 4. Vectorized final score computation
    final_scores = (
        0.4 * ll_similarities + 
        0.4 * hl_similarities + 
        0.2 * decay
    )
    """
    0.45 * ll_similarities + 
        0.45 * hl_similarities + 
        0.1 * decay
    """
    
    return final_scores.tolist()


async def _multi_hop_expand(
    seed_nodes: list[dict],
    ll_keywords: str,
    hl_keywords: str,
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str] = None,
) -> tuple[list[dict], list[dict]]:
    """Multi-hop expansion using a three-way parallel approach 
    (Multi-hop + PPR + FastRP).
    
    The activation of each method is controlled by its respective parameters 
    (e.g., top_ppr_nodes).
    """
    
    if query_param.max_hop <= 0:
        return [], []
        
    # The 3-perspective expansion is the only expansion strategy.
    # The individual methods (PPR, FastRP) will gracefully deactivate 
    # if their parameters are 0 or if the node_embedding engine is not configured.
    # The semantic multi-hop (_original_multi_hop_expand) will always run as part of this.
    return await _semantic_expansion_plus_structural_analysis(
        seed_nodes, ll_keywords, hl_keywords, query_param, 
        knowledge_graph_inst, entities_vdb, relationships_vdb, global_config
    )


async def _semantic_expansion_plus_structural_analysis(
    seed_nodes: list[dict],
    ll_keywords: str,
    hl_keywords: str,
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Three-way parallel expansion: Multi-hop + PPR + Adaptive FastRP with independent pools."""

    logger.info(f"Three-way parallel expansion: multi_hop + ppr({query_param.top_ppr_nodes}) + adaptive_fastrp({query_param.top_fastrp_nodes})")


    # Create params for multi-hop expansion (keeps original top_neighbors)
    multihop_param = QueryParam(
        max_hop=query_param.max_hop,
        top_neighbors=query_param.top_neighbors,
        multi_hop_relevance_threshold=query_param.multi_hop_relevance_threshold,
        top_fastrp_nodes=0,  # Multi-hop only, no structural analysis
    )
    
    # Execute three approaches in parallel
    multihop_task = _original_multi_hop_expand(
        seed_nodes, ll_keywords, hl_keywords, multihop_param,
        knowledge_graph_inst, entities_vdb, relationships_vdb
    )
    
    ppr_task = _independent_ppr_analysis(
        seed_nodes, query_param, knowledge_graph_inst, global_config, ll_keywords, hl_keywords
    )
    
    fastrp_task = _independent_fastrp_analysis(
        seed_nodes, query_param, knowledge_graph_inst, global_config
    )
    
    # Wait for all three approaches to complete
    (multihop_entities, multihop_relations), ppr_entities, fastrp_entities = await asyncio.gather(
        multihop_task, ppr_task, fastrp_task
    )
    
    logger.info(f"Multi-hop found: {len(multihop_entities)} entities, {len(multihop_relations)} relations")
    logger.info(f"PPR found: {len(ppr_entities)} entities")
    logger.info(f"FastRP found: {len(fastrp_entities)} entities")
    
    # Find relations for PPR and FastRP entities
    ppr_relations = await _find_relations_for_entities(ppr_entities, relationships_vdb)
    fastrp_relations = await _find_relations_for_entities(fastrp_entities, relationships_vdb)
    
    # Merge and deduplicate results from three sources
    merged_entities = _merge_three_way_entities(
        multihop_entities, ppr_entities, fastrp_entities
    )
    merged_relations = _merge_three_way_relations(
        multihop_relations, ppr_relations, fastrp_relations
    )
    
    logger.info(f"Three-way merged results: {len(merged_entities)} entities, {len(merged_relations)} relations")
    
    return merged_entities, merged_relations


async def _independent_ppr_analysis(
    seed_nodes: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict[str, str],
    ll_keywords: str = "",
    hl_keywords: str = ""
) -> list[dict]:
    """Independent PPR analysis with Phase 1 + Phase 2 query-aware enhancements - select top N entities by PageRank score."""
    
    if query_param.top_ppr_nodes <= 0:
        return []
    
    try:
        # Get node embedding instance
        node_embedding = global_config.get("node_embedding")
        if not node_embedding:
            logger.warning("Node embedding not available for PPR analysis")
            return []
        
        # Extract seed entity names
        seed_entity_names = [node.get("entity_name", "") for node in seed_nodes if node.get("entity_name")]
        if not seed_entity_names:
            logger.warning("No seed entities for PPR analysis")
            return []
        
        logger.info(f"PPR analysis from {len(seed_entity_names)} seed entities")

        # Compute query-aware seed weights if ll_keywords provided
        entity_similarities = {}
        if ll_keywords.strip():
            try:
                # Get entities vector database for similarity computation
                entities_vdb = global_config.get("entities_vdb")
                if entities_vdb and hasattr(entities_vdb, 'embedding_func'):
                    # Compute query embedding
                    embedding_func = entities_vdb.embedding_func
                    if embedding_func and embedding_func.func:
                        query_embedding = await embedding_func.func([ll_keywords])
                        query_vec = np.array(query_embedding[0])

                        # Compute similarity scores for seed entities
                        for seed_entity in seed_entity_names:
                            # Create entity string for embedding (matching multi-hop format)
                            seed_info = next((node for node in seed_nodes if node.get("entity_name") == seed_entity), None)
                            if seed_info and seed_info.get("entity_description"):
                                entity_str = f"{seed_entity} {seed_info['entity_description']}"
                                entity_embedding = await embedding_func.func([entity_str])
                                entity_vec = np.array(entity_embedding[0])

                                # Compute cosine similarity
                                similarity = cosine_similarity(query_vec, entity_vec)
                                entity_similarities[seed_entity] = similarity

                        logger.debug(f"Computed query-aware similarities for {len(entity_similarities)} seed entities")

            except Exception as e:
                logger.warning(f"Could not compute query-aware similarities: {e}")

        # Phase 2: Compute relation similarities for edge reweighting
        relation_similarities = {}
        if hl_keywords.strip():
            try:
                # Get relationships vector database for similarity computation
                relationships_vdb = global_config.get("relationships_vdb")
                if relationships_vdb and hasattr(relationships_vdb, 'embedding_func'):
                    # Compute hl_keywords embedding
                    embedding_func = relationships_vdb.embedding_func
                    if embedding_func and embedding_func.func:
                        hl_query_embedding = await embedding_func.func([hl_keywords])
                        hl_query_vec = np.array(hl_query_embedding[0])

                        # Get all relations from knowledge graph
                        all_relations = await knowledge_graph_inst.get_all_relationships()

                        # Compute similarity scores for all relations
                        for relation in all_relations:
                            rel_keywords = relation.get("keywords", "")
                            if rel_keywords.strip():
                                # Create relation string for embedding (matching multi-hop format)
                                rel_str = f"{rel_keywords}\t{relation.get('source_id', '')}\n{relation.get('target_id', '')}\n{relation.get('description', '')}"
                                rel_embedding = await embedding_func.func([rel_str])
                                rel_vec = np.array(rel_embedding[0])

                                # Compute cosine similarity
                                similarity = cosine_similarity(hl_query_vec, rel_vec)
                                relation_similarities[rel_keywords] = similarity

                        logger.debug(f"Computed query-aware relation similarities for {len(relation_similarities)} relations")

            except Exception as e:
                logger.warning(f"Could not compute relation similarities: {e}")

        # Compute Personalized PageRank with both Phase 1 and Phase 2 enhancements
        ppr_scores = node_embedding.compute_personalized_pagerank(
            seed_entity_names,
            entity_similarities=entity_similarities if entity_similarities else None,
            relation_similarities=relation_similarities if relation_similarities else None,
            tau=0.1,      # Temperature for softmax weighting (Phase 1)
            alpha=0.3     # Edge reweighting strength (Phase 2)
        )
        if not ppr_scores:
            logger.warning("No Personalized PageRank scores computed")
            return []
        
        # Get all entities (excluding seeds) and sort by PPR score
        all_entities = [
            (entity_name, score) for entity_name, score in ppr_scores.items()
            if entity_name not in seed_entity_names
        ]
        
        if not all_entities:
            logger.warning("No entities found for PPR ranking")
            return []
        
        # Sort by PageRank score and select top N
        all_entities.sort(key=lambda x: x[1], reverse=True)
        top_entities = all_entities[:query_param.top_ppr_nodes]
        
        logger.info(f"PPR selected top {len(top_entities)} entities from {len(all_entities)} total entities")
        
        # Get entity details from knowledge graph
        ppr_entities = []
        for entity_name, ppr_score in top_entities:
            entity_data = await knowledge_graph_inst.get_node(entity_name)
            if entity_data:
                entity_copy = dict(entity_data)
                entity_copy.update({
                    "entity_name": entity_name,
                    "pagerank_score": ppr_score,
                    "discovery_method": "ppr_analysis",
                    "rank": entity_data.get("degree", 0),
                })
                ppr_entities.append(entity_copy)
        
        logger.info(f"PPR analysis completed: {len(ppr_entities)} entities with avg PageRank: {sum(e['pagerank_score'] for e in ppr_entities) / len(ppr_entities) if ppr_entities else 0:.4f}")
        
        return ppr_entities
        
    except Exception as e:
        logger.error(f"PPR analysis failed: {e}")
        return []


async def _independent_fastrp_analysis(
    seed_nodes: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict[str, str]
) -> list[dict]:
    """Independent FastRP analysis - select top N entities by structural similarity."""
    
    if query_param.top_fastrp_nodes <= 0:
        return []
    
    try:
        # Get node embedding instance
        node_embedding = global_config.get("node_embedding")
        if not node_embedding:
            logger.warning("Node embedding not available for FastRP analysis")
            return []
        
        # Extract seed entity names
        seed_entity_names = [node.get("entity_name", "") for node in seed_nodes if node.get("entity_name")]
        if not seed_entity_names:
            logger.warning("No seed entities for FastRP analysis")
            return []
        
        logger.info(f"FastRP analysis from {len(seed_entity_names)} seed entities")
        
        # Get all entities for similarity calculation
        all_entity_names = list(node_embedding.fastrp_embeddings.keys())
        candidate_entities = [name for name in all_entity_names if name not in seed_entity_names]
        
        if not candidate_entities:
            logger.warning("No candidate entities for FastRP analysis")
            return []
        
        # Calculate FastRP similarity for all candidates
        smart_candidates = []
        for entity_name in candidate_entities:
            try:
                adaptive_fastrp_similarity = _compute_adaptive_fastrp_similarity(
                    entity_name, seed_entity_names, node_embedding
                )
                
                # Get entity details from knowledge graph
                entity_data = await knowledge_graph_inst.get_node(entity_name)
                if entity_data:
                    smart_candidates.append({
                        "entity_name": entity_name,
                        "adaptive_fastrp_similarity": adaptive_fastrp_similarity,
                        "discovery_method": "fastrp_analysis",
                        "rank": entity_data.get("degree", 0),
                        **entity_data
                    })
                    
            except Exception as e:
                logger.debug(f"Error processing FastRP candidate {entity_name}: {e}")
        
        if not smart_candidates:
            logger.warning("No valid FastRP candidates found")
            return []
        
        # Sort by FastRP similarity and select top N
        smart_candidates.sort(key=lambda x: x["adaptive_fastrp_similarity"], reverse=True)
        top_smart_entities = smart_candidates[:query_param.top_fastrp_nodes]

        logger.info(f"Adaptive FastRP selected top {len(top_smart_entities)} entities from {len(smart_candidates)} candidates")
        logger.info(f"Adaptive FastRP analysis completed: avg similarity: {sum(e['adaptive_fastrp_similarity'] for e in top_smart_entities) / len(top_smart_entities) if top_smart_entities else 0:.4f}")

        return top_smart_entities
        
    except Exception as e:
        logger.error(f"FastRP analysis failed: {e}")
        return []



def _compute_adaptive_fastrp_similarity(target_entity: str, seed_entities: list[str], node_embedding) -> float:
    """Compute Adaptive FastRP similarity using precomputed embeddings."""
    try:
        from lightrag.gnn.structural_similarity import compute_adaptive_fastrp_similarity
        return compute_adaptive_fastrp_similarity(target_entity, seed_entities, node_embedding)
    except Exception as e:
        logger.error(f"Error computing adaptive fastrp similarity: {e}")
        return 0.0


async def _find_relations_for_entities(
    entities: list[dict], 
    relationships_vdb: BaseVectorStorage
) -> list[dict]:
    """Find relations connecting given entities."""
    if not entities:
        return []
    
    try:
        entity_names = [e.get("entity_name", "") for e in entities]
        relations = []
        
        for entity_name in entity_names:
            try:
                # Search for relations involving this entity
                relation_results = await relationships_vdb.query(entity_name, top_k=10)
                
                for relation_data in relation_results:
                    src_id = relation_data.get("src_id", "")
                    tgt_id = relation_data.get("tgt_id", "")
                    
                    # Keep relations that connect our entities
                    if (src_id in entity_names and tgt_id in entity_names):
                        relation_info = {
                            "src_id": src_id,
                            "tgt_id": tgt_id,
                            "description": relation_data.get("description", ""),
                            "keywords": relation_data.get("keywords", ""),
                            "discovery_method": "entity_relation",
                            "rank": relation_data.get("rank", 0),
                            **relation_data
                        }
                        
                        # Avoid duplicates
                        if not any(
                            r["src_id"] == src_id and r["tgt_id"] == tgt_id 
                            for r in relations
                        ):
                            relations.append(relation_info)
                            
            except Exception as e:
                logger.debug(f"Error finding relations for {entity_name}: {e}")
        
        logger.debug(f"Found {len(relations)} relations for {len(entity_names)} entities")
        return relations
        
    except Exception as e:
        logger.error(f"Error finding relations for entities: {e}")
        return []


def _merge_three_way_entities(
    multihop_entities: list[dict],
    ppr_entities: list[dict],
    fastrp_entities: list[dict]
) -> list[dict]:
    """Merge and deduplicate entities from three sources."""

    seen_entities = set()
    merged = []
    entity_methods = {}

    # Combine all entities from three sources
    all_entities = [
        (multihop_entities, "multihop"),
        (ppr_entities, "ppr"),
        (fastrp_entities, "fastrp")
    ]

    # Process all entities and track discovery methods
    for entities, method in all_entities:
        for entity in entities:
            entity_name = entity.get("entity_name", "")
            if not entity_name:
                continue

            if entity_name not in seen_entities:
                # First time seeing this entity, add it
                entity_copy = dict(entity)
                entity_copy["discovery_method"] = method
                entity_methods[entity_name] = [method]
                merged.append(entity_copy)
                seen_entities.add(entity_name)
            else:
                # Entity already seen, just update discovery methods
                if entity_name in entity_methods:
                    entity_methods[entity_name].append(method)

    # Update discovery method for entities found by multiple sources
    for entity in merged:
        entity_name = entity.get("entity_name", "")
        if entity_name in entity_methods and len(entity_methods[entity_name]) > 1:
            methods = "+".join(sorted(entity_methods[entity_name]))
            entity["discovery_method"] = methods

    logger.info(f"Three-way entity merge: multihop={len(multihop_entities)}, ppr={len(ppr_entities)}, fastrp={len(fastrp_entities)} -> total={len(merged)}")

    return merged


def _merge_three_way_relations(
    multihop_relations: list[dict],
    ppr_relations: list[dict], 
    fastrp_relations: list[dict]
) -> list[dict]:
    """Merge and deduplicate relations from three sources."""
    
    seen_relations = set()
    merged = []
    
    # Create unique keys for relations (src_id + tgt_id)
    def relation_key(rel):
        src_id = rel.get("src_id", "")
        tgt_id = rel.get("tgt_id", "")
        return f"{src_id}|{tgt_id}"
    
    # Add all relations with deduplication
    all_relations = multihop_relations + ppr_relations + fastrp_relations
    
    for relation in all_relations:
        key = relation_key(relation)
        if key not in seen_relations:
            merged.append(relation)
            seen_relations.add(key)
    
    logger.info(f"Three-way relation merge: multihop={len(multihop_relations)}, ppr={len(ppr_relations)}, fastrp={len(fastrp_relations)} -> total={len(merged)}")
    
    return merged


async def _original_multi_hop_expand(
    seed_nodes: list[dict],
    ll_keywords: str,
    hl_keywords: str,
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
) -> tuple[list[dict], list[dict]]:
    """Original multi-hop expansion logic (semantic path only)."""
    
    logger.info(f"Starting multi-hop expansion: max_hop={query_param.max_hop}, top_neighbors={query_param.top_neighbors}, threshold={query_param.multi_hop_relevance_threshold}")
    
    # Pre-compute query embeddings for performance optimization
    query_ll_embedding = None
    query_hl_embedding = None
    try:
        # Pre-compute LL query embedding
        if ll_keywords and entities_vdb.embedding_func and entities_vdb.embedding_func.func:
            query_ll_result = await entities_vdb.embedding_func.func([ll_keywords])
            query_ll_embedding = query_ll_result[0]
        
        # Pre-compute HL query embedding  
        if hl_keywords and relationships_vdb.embedding_func and relationships_vdb.embedding_func.func:
            query_hl_result = await relationships_vdb.embedding_func.func([hl_keywords])
            query_hl_embedding = query_hl_result[0]
            
        logger.debug(f"Pre-computed query embeddings: ll={'✓' if query_ll_embedding else '✗'}, hl={'✓' if query_hl_embedding else '✗'}")
    except Exception as e:
        logger.debug(f"Error pre-computing query embeddings: {e}")
    
    all_expanded_entities = []
    all_expanded_relations = []
    current_nodes = seed_nodes.copy()
    visited_nodes = set(node["entity_name"] for node in seed_nodes)
    visited_relations = set()  # Track visited (src_id, tgt_id) pairs to prevent duplicates and cycles
    
    # BFS expansion for all hops
    for hop in range(1, query_param.max_hop + 1):
        logger.info(f"Multi-hop: Starting hop {hop} with {len(current_nodes)} current nodes")
        if not current_nodes:
            logger.info(f"Multi-hop: No current nodes at hop {hop}, stopping expansion")
            break
            
        logger.debug(f"Processing hop {hop} with {len(current_nodes)} nodes")
        next_nodes = []
        hop_relations = []
        
        # Get neighbors for current nodes
        node_names = [node["entity_name"] for node in current_nodes]
        # logger.info(f"Multi-hop: Fetching edges for {len(node_names)} nodes")
        batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)
        # logger.info(f"Multi-hop: Retrieved edges dict with {len(batch_edges_dict)} entries")
        
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
        
    
    logger.info(f"Multi-hop expansion completed: {len(all_expanded_entities)} entities, {len(all_expanded_relations)} relations")
    return all_expanded_entities, all_expanded_relations


async def get_keywords_from_query(
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Retrieves high-level and low-level keywords for RAG operations.

    This function checks if keywords are already provided in query parameters,
    and if not, extracts them from the query text using LLM.

    Args:
        query: The user's query text
        query_param: Query parameters that may contain pre-defined keywords
        global_config: Global configuration dictionary
        hashing_kv: Optional key-value storage for caching results

    Returns:
        A tuple containing (high_level_keywords, low_level_keywords)
    """
    # Check if pre-defined keywords are already provided
    if query_param.hl_keywords or query_param.ll_keywords:
        return query_param.hl_keywords, query_param.ll_keywords

    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )
    return hl_keywords, ll_keywords


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # 1. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(
        param.mode,
        text,
        param.hl_keywords or [],
        param.ll_keywords or [],
    )
    cached_response = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_response is not None:
        try:
            keywords_data = json_repair.loads(cached_response)
            return keywords_data.get("high_level_keywords", []), keywords_data.get(
                "low_level_keywords", []
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 2. Build the examples
    examples = "\n".join(PROMPTS["keywords_extraction_examples"])

    language = global_config["addon_params"].get("language", DEFAULT_SUMMARY_LANGUAGE)

    # 3. Process conversation history
    # history_context = ""
    # if param.conversation_history:
    #     history_context = get_conversation_turns(
    #         param.conversation_history, param.history_turns
    #     )

    # 4. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text,
        examples=examples,
        language=language,
    )

    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(kw_prompt))
    logger.debug(
        f"[extract_keywords] Sending to LLM: {len_of_prompts:,} tokens (Prompt: {len_of_prompts})"
    )

    # 5. Call the LLM for keyword extraction
    if param.model_func:
        use_model_func = param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    result = await use_model_func(kw_prompt, keyword_extraction=True)

    # 6. Parse out JSON from the LLM response
    result = remove_think_tags(result)
    try:
        keywords_data = json_repair.loads(result)
        if not keywords_data:
            logger.error("No JSON-like structure found in the LLM respond.")
            return [], []
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"LLM respond: {result}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    # 7. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        if hashing_kv.global_config.get("enable_llm_cache"):
            # Save to cache with query parameters
            queryparam_dict = {
                "mode": param.mode,
                "response_type": param.response_type,
                "top_k": param.top_k,
                "chunk_top_k": param.chunk_top_k,
                "max_entity_tokens": param.max_entity_tokens,
                "max_relation_tokens": param.max_relation_tokens,
                "max_total_tokens": param.max_total_tokens,
                "hl_keywords": param.hl_keywords or [],
                "ll_keywords": param.ll_keywords or [],
                "user_prompt": param.user_prompt or "",
                "enable_rerank": param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=json.dumps(cache_data),
                    prompt=text,
                    mode=param.mode,
                    cache_type="keywords",
                    queryparam=queryparam_dict,
                ),
            )

    return hl_keywords, ll_keywords


async def _get_vector_context(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    query_embedding: list[float] = None,
) -> list[dict]:
    """
    Retrieve text chunks from the vector database without reranking or truncation.

    This function performs vector search to find relevant text chunks for a query.
    Reranking and truncation will be handled later in the unified processing.

    Args:
        query: The query string to search for
        chunks_vdb: Vector database containing document chunks
        query_param: Query parameters including chunk_top_k and ids
        query_embedding: Optional pre-computed query embedding to avoid redundant embedding calls

    Returns:
        List of text chunks with metadata
    """
    try:
        # Use chunk_top_k if specified, otherwise fall back to top_k
        search_top_k = query_param.chunk_top_k or query_param.top_k

        results = await chunks_vdb.query(
            query, top_k=search_top_k, query_embedding=query_embedding
        )
        if not results:
            logger.info(f"Naive query: 0 chunks (chunk_top_k: {search_top_k})")
            return []

        valid_chunks = []
        for result in results:
            if "content" in result:
                chunk_with_metadata = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                    "source_type": "vector",  # Mark the source type
                    "chunk_id": result.get("id"),  # Add chunk_id for deduplication
                }
                valid_chunks.append(chunk_with_metadata)

        logger.info(
            f"Naive query: {len(valid_chunks)} chunks (chunk_top_k: {search_top_k})"
        )
        return valid_chunks

    except Exception as e:
        logger.error(f"Error in _get_vector_context: {e}")
        return []


async def _build_query_context(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    chunks_vdb: BaseVectorStorage = None,
):
    if not query:
        logger.warning("Query is empty, skipping context building")
        return ""

    logger.info(f"Process {os.getpid()} building query context...")

    # Collect chunks from different sources separately
    vector_chunks = []
    entity_chunks = []
    relation_chunks = []
    entities_context = []
    relations_context = []

    # Store original data for later text chunk retrieval
    local_entities = []
    local_relations = []
    global_entities = []
    global_relations = []

    # Track chunk sources and metadata for final logging
    chunk_tracking = {}  # chunk_id -> {source, frequency, order}

    # Pre-compute query embedding once for all vector operations
    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    query_embedding = None
    if query and (kg_chunk_pick_method == "VECTOR" or chunks_vdb):
        embedding_func_config = text_chunks_db.embedding_func
        if embedding_func_config and embedding_func_config.func:
            try:
                query_embedding = await embedding_func_config.func([query])
                query_embedding = query_embedding[
                    0
                ]  # Extract first embedding from batch result
                logger.debug("Pre-computed query embedding for all vector operations")
            except Exception as e:
                logger.warning(f"Failed to pre-compute query embedding: {e}")
                query_embedding = None

    # Handle local and global modes
    if query_param.mode == "local" and len(ll_keywords) > 0:
        local_entities, local_relations = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
            global_config,
        )

    elif query_param.mode == "global" and len(hl_keywords) > 0:
        global_relations, global_entities = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
        )

    else:  # hybrid or mix mode
        if len(ll_keywords) > 0:
            local_entities, local_relations = await _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                query_param,
                global_config,
            )
        if len(hl_keywords) > 0:
            global_relations, global_entities = await _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                query_param,
            )

        # Get vector chunks first if in mix mode
        if query_param.mode == "mix" and chunks_vdb:
            vector_chunks = await _get_vector_context(
                query,
                chunks_vdb,
                query_param,
                query_embedding,
            )
            # Track vector chunks with source metadata
            for i, chunk in enumerate(vector_chunks):
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id:
                    chunk_tracking[chunk_id] = {
                        "source": "C",
                        "frequency": 1,  # Vector chunks always have frequency 1
                        "order": i + 1,  # 1-based order in vector search results
                    }
                else:
                    logger.warning(f"Vector chunk missing chunk_id: {chunk}")

    # Use round-robin merge to combine local and global data fairly
    final_entities = []
    seen_entities = set()

    # Round-robin merge entities
    max_len = max(len(local_entities), len(global_entities))
    for i in range(max_len):
        # First from local
        if i < len(local_entities):
            entity = local_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)

        # Then from global
        if i < len(global_entities):
            entity = global_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)

    # Round-robin merge relations
    final_relations = []
    seen_relations = set()

    max_len = max(len(local_relations), len(global_relations))
    for i in range(max_len):
        # First from local
        if i < len(local_relations):
            relation = local_relations[i]
            # Build relation unique identifier
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

        # Then from global
        if i < len(global_relations):
            relation = global_relations[i]
            # Build relation unique identifier
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )

            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

    # Generate entities context
    entities_context = []
    for i, n in enumerate(final_entities):
        created_at = n.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Get file path from node data
        file_path = n.get("file_path", "unknown_source")

        entities_context.append(
            {
                "id": i + 1,
                "entity": n["entity_name"],
                "type": n.get("entity_type", "UNKNOWN"),
                "description": n.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": file_path,
                "source_id": n.get("source_id", ""),  # For chunk retrieval
            }
        )

    # Generate relations context
    relations_context = []
    for i, e in enumerate(final_relations):
        created_at = e.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Get file path from edge data
        file_path = e.get("file_path", "unknown_source")

        # Handle different relation data formats
        if "src_tgt" in e:
            entity1, entity2 = e["src_tgt"]
        else:
            entity1, entity2 = e.get("src_id"), e.get("tgt_id")

        relations_context.append(
            {
                "id": i + 1,
                "entity1": entity1,
                "entity2": entity2,
                "description": e.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": file_path,
                "source_id": e.get("source_id", ""),  # For chunk retrieval
            }
        )

    logger.debug(
        f"Initial KG query results: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Unified token control system - Apply precise token limits to entities and relations
    tokenizer = text_chunks_db.global_config.get("tokenizer")
    # Get new token limits from query_param (with fallback to global_config)
    max_entity_tokens = getattr(
        query_param,
        "max_entity_tokens",
        text_chunks_db.global_config.get(
            "max_entity_tokens", DEFAULT_MAX_ENTITY_TOKENS
        ),
    )
    max_relation_tokens = getattr(
        query_param,
        "max_relation_tokens",
        text_chunks_db.global_config.get(
            "max_relation_tokens", DEFAULT_MAX_RELATION_TOKENS
        ),
    )
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        text_chunks_db.global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Truncate entities based on complete JSON serialization
    if entities_context:
        # Process entities context to replace GRAPH_FIELD_SEP with : in file_path fields
        for entity in entities_context:
            # Keep file_path and created_at for all entities
            if "file_path" in entity and entity["file_path"]:
                entity["file_path"] = entity["file_path"].replace(GRAPH_FIELD_SEP, ";")

        entities_context = truncate_list_by_token_size(
            entities_context,
            key=lambda x: json.dumps(x, ensure_ascii=False),
            max_token_size=max_entity_tokens,
            tokenizer=tokenizer,
        )

    # Truncate relations based on complete JSON serialization
    if relations_context:
        # Process relations context to replace GRAPH_FIELD_SEP with : in file_path fields
        for relation in relations_context:
            # Keep file_path and created_at for all relations
            if "file_path" in relation and relation["file_path"]:
                relation["file_path"] = relation["file_path"].replace(
                    GRAPH_FIELD_SEP, ";"
                )

        relations_context = truncate_list_by_token_size(
            relations_context,
            key=lambda x: json.dumps(x, ensure_ascii=False),
            max_token_size=max_relation_tokens,
            tokenizer=tokenizer,
        )

    # After truncation, get text chunks based on final entities and relations
    logger.info(
        f"Truncated KG query results: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Create filtered data based on truncated context
    final_node_datas = []
    if entities_context and final_entities:
        final_entity_names = {e["entity"] for e in entities_context}
        seen_nodes = set()
        for node in final_entities:
            name = node.get("entity_name")
            if name in final_entity_names and name not in seen_nodes:
                final_node_datas.append(node)
                seen_nodes.add(name)

    final_edge_datas = []
    if relations_context and final_relations:
        final_relation_pairs = {(r["entity1"], r["entity2"]) for r in relations_context}
        seen_edges = set()
        for edge in final_relations:
            src, tgt = edge.get("src_id"), edge.get("tgt_id")
            if src is None or tgt is None:
                src, tgt = edge.get("src_tgt", (None, None))

            pair = (src, tgt)
            if pair in final_relation_pairs and pair not in seen_edges:
                final_edge_datas.append(edge)
                seen_edges.add(pair)

    # Get text chunks based on final filtered data
    # To preserve the influence of entity order,  entiy-based chunks should not be deduplcicated by vector_chunks
    if final_node_datas:
        entity_chunks = await _find_related_text_unit_from_entities(
            final_node_datas,
            query_param,
            text_chunks_db,
            knowledge_graph_inst,
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Find deduplcicated chunks from edge
    # Deduplication cause chunks solely relation-based to be prioritized and sent to the LLM when re-ranking is disabled
    if final_edge_datas:
        relation_chunks = await _find_related_text_unit_from_relations(
            final_edge_datas,
            query_param,
            text_chunks_db,
            entity_chunks,
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Store initial chunks for later processing (after multi-hop expansion)
    # Collect all chunks without merging or truncation
    initial_chunks = {
        "vector_chunks": vector_chunks,
        "entity_chunks": entity_chunks,
        "relation_chunks": relation_chunks,
    }
    
    origin_len = len(vector_chunks) + len(entity_chunks) + len(relation_chunks)
    logger.info(f"Collected initial chunks: {origin_len} total (Vector: {len(vector_chunks)}, Entity: {len(entity_chunks)}, Relation: {len(relation_chunks)})")
    
    # Process initial chunks to build text_units_context before any early returns
    # Round-robin merge of initial chunks for text_units_context
    initial_merged_chunks = []
    seen_chunk_ids = set()
    max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks)) if any([vector_chunks, entity_chunks, relation_chunks]) else 0

    for i in range(max_len):
        # Add from each source in round-robin fashion: vector → entity → relation
        for chunk_list in [vector_chunks, entity_chunks, relation_chunks]:
            if i < len(chunk_list):
                chunk = chunk_list[i]
                chunk_id = chunk.get("chunk_id") or chunk.get("id") or f"chunk_{i}"
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    initial_merged_chunks.append({
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                        "timestamp": chunk.get("timestamp", ""),
                    })

    # Apply token processing to initial chunks for text_units_context
    text_units_context = []
    if initial_merged_chunks:
        # Apply chunk_top_k truncation if specified
        if query_param.chunk_top_k and len(initial_merged_chunks) > query_param.chunk_top_k:
            initial_merged_chunks = initial_merged_chunks[:query_param.chunk_top_k]

        # Process chunks through unified processing (for token limits, reranking, etc.)
        tokenizer = text_chunks_db.global_config.get("tokenizer")
        if tokenizer:
            from lightrag.utils import process_chunks_unified
            try:
                # Calculate a reasonable token limit for initial chunks
                max_total_tokens = getattr(query_param, "max_total_tokens", text_chunks_db.global_config.get("max_total_tokens", 120000))
                # Reserve about 30% of tokens for chunks, rest for KG context and system prompt
                initial_chunk_token_limit = int(max_total_tokens * 0.3)

                processed_chunks = await process_chunks_unified(
                    query=query,
                    unique_chunks=initial_merged_chunks,
                    query_param=query_param,
                    global_config=text_chunks_db.global_config,
                    source_type=query_param.mode,
                    chunk_token_limit=initial_chunk_token_limit,
                )

                # Build text_units_context from processed chunks
                for i, chunk in enumerate(processed_chunks):
                    chunk_context = {
                        "id": i + 1,
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                    }
                    # Only add timestamp field if it exists and is not empty
                    timestamp = chunk.get("timestamp", "")
                    if timestamp:
                        chunk_context["The content is from"] = timestamp
                    text_units_context.append(chunk_context)
            except Exception as e:
                logger.warning(f"Failed to process initial chunks: {e}")
                # Fallback to raw chunks
                for i, chunk in enumerate(initial_merged_chunks):
                    chunk_context = {
                        "id": i + 1,
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                    }
                    # Only add timestamp field if it exists and is not empty
                    timestamp = chunk.get("timestamp", "")
                    if timestamp:
                        chunk_context["The content is from"] = timestamp
                    text_units_context.append(chunk_context)

    # not necessary to use LLM to generate a response
    if not entities_context and not relations_context:
        return None

    # Keep original round-robin ordering logic for fair data balance

    # Remove internal fields before sending to LLM
    fields_to_remove = {"source_id", "created_at", "file_path","id"}
    entities_for_llm = [{k: v for k, v in e.items() if k not in fields_to_remove} for e in entities_context]
    relations_for_llm = [{k: v for k, v in r.items() if k not in fields_to_remove} for r in relations_context]
    chunks_for_llm = [{k: v for k, v in c.items() if k != "id"} for c in text_units_context]

    entities_str = json.dumps(entities_for_llm, ensure_ascii=False)
    relations_str = json.dumps(relations_for_llm, ensure_ascii=False)
    text_units_str = json.dumps(chunks_for_llm, ensure_ascii=False)

    result = f"""-----Entities(KG)-----

```json
{entities_str}
```

-----Relationships(KG)-----

```json
{relations_str}
```

-----Document Chunks(DC)-----

```json
{text_units_str}
```

"""
    
    # Multi-hop expansion if enabled
    if query_param.max_hop > 0 and (entities_context or relations_context):
        logger.info(f"Performing multi-hop expansion with max_hop={query_param.max_hop}")
        logger.info(f"Initial entities: {len(entities_context)}, relations: {len(relations_context)}")
        
        # Collect initial chunks for expansion processing with source labels
        merged_chunks = []
        
        # Add vector chunks with source label
        for chunk in initial_chunks["vector_chunks"]:
            merged_chunks.append({
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
                "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "timestamp": chunk.get("timestamp", ""),
                "source": "vector",
            })
        
        # Add entity chunks with source label
        for chunk in initial_chunks["entity_chunks"]:
            merged_chunks.append({
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
                "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "timestamp": chunk.get("timestamp", ""),
                "source": "entity",
            })
        
        # Add relation chunks with source label
        for chunk in initial_chunks["relation_chunks"]:
            merged_chunks.append({
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
                "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "timestamp": chunk.get("timestamp", ""),
                "source": "relation",
            })
        
        # Collect seed nodes for multi-hop expansion
        seed_nodes = final_entities.copy()
        
        # Perform multi-hop expansion
        expanded_entities, expanded_relations = await _multi_hop_expand(
            seed_nodes=seed_nodes,
            ll_keywords=ll_keywords,
            hl_keywords=hl_keywords,
            knowledge_graph_inst=knowledge_graph_inst,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            query_param=query_param,
            global_config=global_config,
        )

        # Merge expanded results with existing context using round-robin
        if expanded_entities:
            # Create context format for expanded entities
            expanded_entities_context = []
            for i, n in enumerate(expanded_entities):
                created_at = n.get("created_at", "UNKNOWN")
                if isinstance(created_at, (int, float)):
                    created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
                
                file_path = n.get("file_path", "unknown_source")
                
                expanded_entities_context.append({
                    "id": 0,  # Will be reassigned after round-robin merge
                    "entity": n["entity_name"],
                    "type": n.get("entity_type", "UNKNOWN"),
                    "description": n.get("description", "UNKNOWN"),
                    "created_at": created_at,
                    "file_path": file_path,
                    "source_id": n.get("source_id", ""),  # For chunk retrieval
                })
            
            # Round-robin merge initial and expanded entities
            original_entities = entities_context.copy()
            entities_context = []
            seen_entities = set()
            
            max_len = max(len(original_entities), len(expanded_entities_context))
            for i in range(max_len):
                # First from original (initial query results)
                if i < len(original_entities):
                    entity = original_entities[i]
                    entity_name = entity.get("entity")
                    if entity_name and entity_name not in seen_entities:
                        entities_context.append(entity)
                        seen_entities.add(entity_name)
                
                # Then from expanded (multi-hop results)
                if i < len(expanded_entities_context):
                    entity = expanded_entities_context[i]
                    entity_name = entity.get("entity")
                    if entity_name and entity_name not in seen_entities:
                        entities_context.append(entity)
                        seen_entities.add(entity_name)
        
        if expanded_relations:
            # Create context format for expanded relations
            expanded_relations_context = []
            for i, e in enumerate(expanded_relations):
                created_at = e.get("created_at", "UNKNOWN")
                if isinstance(created_at, (int, float)):
                    created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
                
                file_path = e.get("file_path", "unknown_source")
                
                if "src_tgt" in e:
                    entity1, entity2 = e["src_tgt"]
                else:
                    entity1, entity2 = e.get("src_id"), e.get("tgt_id")
                
                expanded_relations_context.append({
                    "id": 0,  # Will be reassigned after round-robin merge
                    "entity1": entity1,
                    "entity2": entity2,
                    "description": e.get("description", "UNKNOWN"),
                    "created_at": created_at,
                    "file_path": file_path,
                    "source_id": e.get("source_id", ""),  # For chunk retrieval
                })
            
            # Round-robin merge initial and expanded relations
            original_relations = relations_context.copy()
            relations_context = []
            seen_relations = set()
            
            max_len = max(len(original_relations), len(expanded_relations_context))
            for i in range(max_len):
                # First from original (initial query results)
                if i < len(original_relations):
                    relation = original_relations[i]
                    rel_key = tuple(sorted([relation.get("entity1"), relation.get("entity2")]))
                    if rel_key not in seen_relations:
                        relations_context.append(relation)
                        seen_relations.add(rel_key)
                
                # Then from expanded (multi-hop results) 
                if i < len(expanded_relations_context):
                    relation = expanded_relations_context[i]
                    rel_key = tuple(sorted([relation.get("entity1"), relation.get("entity2")]))
                    if rel_key not in seen_relations:
                        relations_context.append(relation)
                        seen_relations.add(rel_key)
        
        logger.info(f"Multi-hop expansion completed: added {len(expanded_entities)} entities, {len(expanded_relations)} relations")

        # Note: Token limits will be applied AFTER Recognition Memory filtering

        # Get source chunks for expanded entities and relations separately
        entity_source_ids = set()
        for entity in expanded_entities:
            source_id = entity.get("source_id", "")
            if source_id:
                if GRAPH_FIELD_SEP in source_id:
                    entity_source_ids.update(source_id.split(GRAPH_FIELD_SEP))
                else:
                    entity_source_ids.add(source_id)
        
        relation_source_ids = set()
        for relation in expanded_relations:
            source_id = relation.get("source_id", "")
            if source_id:
                if GRAPH_FIELD_SEP in source_id:
                    relation_source_ids.update(source_id.split(GRAPH_FIELD_SEP))
                else:
                    relation_source_ids.add(source_id)

        # Get chunks for expanded entities separately
        expanded_entity_chunks = []
        if entity_source_ids:
            for source_id in entity_source_ids:
                chunk_data = await text_chunks_db.get_by_id(source_id)
                if chunk_data:
                    expanded_entity_chunks.append({
                        "content": chunk_data["content"],
                        "file_path": chunk_data.get("file_path", "unknown_source"),
                        "timestamp": chunk_data.get("timestamp", ""),
                        "chunk_id": source_id,
                        "source": "expanded_entity"
                    })
        
        # Get chunks for expanded relations separately  
        expanded_relation_chunks = []
        if relation_source_ids:
            for source_id in relation_source_ids:
                chunk_data = await text_chunks_db.get_by_id(source_id)
                if chunk_data:
                    expanded_relation_chunks.append({
                        "content": chunk_data["content"],
                        "file_path": chunk_data.get("file_path", "unknown_source"),
                        "timestamp": chunk_data.get("timestamp", ""),
                        "chunk_id": source_id,
                        "source": "expanded_relation"
                    })
            
        logger.info(f"After expansion: {len(entities_context)} entities, {len(relations_context)} relations")

        # Recognition Memory filtering (applied BEFORE chunk retrieval)
        if query_param.enable_recognition and (entities_context or relations_context):
            from lightrag.recognition import recognition_memory_filter

            # Convert context format to entity/relation format for filtering
            entities_for_filtering = []
            for entity in entities_context:
                entities_for_filtering.append({
                    "entity_name": entity.get("entity"),
                    "entity_type": entity.get("type"),
                    "description": entity.get("description"),
                    "created_at": entity.get("created_at"),
                    "file_path": entity.get("file_path"),
                    "source_id": entity.get("source_id", ""),  # Preserve source_id
                })

            relations_for_filtering = []
            for relation in relations_context:
                relations_for_filtering.append({
                    "src_id": relation.get("entity1"),
                    "tgt_id": relation.get("entity2"),
                    "description": relation.get("description"),
                    "created_at": relation.get("created_at"),
                    "file_path": relation.get("file_path"),
                    "source_id": relation.get("source_id", ""),  # Preserve source_id
                })

            # Use OpenAI to filter
            tool_llm_model_name = global_config.get("tool_llm_model_name", "gpt-4o-mini")
            filtered_entities, filtered_relations = await recognition_memory_filter(
                query=query,
                entities=entities_for_filtering,
                relations=relations_for_filtering,
                batch_size=query_param.recognition_batch_size,
                tool_llm_model_name=tool_llm_model_name,
                global_config=global_config,
            )

            # Rebuild context with filtered data
            entities_context = []
            for i, n in enumerate(filtered_entities):
                entities_context.append({
                    "id": i + 1,
                    "entity": n.get("entity_name"),
                    "type": n.get("entity_type", "UNKNOWN"),
                    "description": n.get("description", "UNKNOWN"),
                    "created_at": n.get("created_at", "UNKNOWN"),
                    "file_path": n.get("file_path", "unknown_source"),
                    "source_id": n.get("source_id", ""),  # Keep source_id for chunk retrieval
                })

            relations_context = []
            for i, e in enumerate(filtered_relations):
                relations_context.append({
                    "id": i + 1,
                    "entity1": e.get("src_id"),
                    "entity2": e.get("tgt_id"),
                    "description": e.get("description", "UNKNOWN"),
                    "created_at": e.get("created_at", "UNKNOWN"),
                    "file_path": e.get("file_path", "unknown_source"),
                    "source_id": e.get("source_id", ""),  # Keep source_id for chunk retrieval
                })

            

        # Apply token limits AFTER Recognition Memory filtering
        entities_context = truncate_list_by_token_size(
            entities_context,
            key=lambda x: json.dumps(x, ensure_ascii=False),
            max_token_size=max_entity_tokens,
            tokenizer=tokenizer,
        )

        relations_context = truncate_list_by_token_size(
            relations_context,
            key=lambda x: json.dumps(x, ensure_ascii=False),
            max_token_size=max_relation_tokens,
            tokenizer=tokenizer,
        )

        # logger.info(f"After token truncation: {len(entities_context)} entities, {len(relations_context)} relations")

        # Now retrieve and process chunks based on (potentially filtered) entities and relations
        # This happens regardless of Recognition Memory setting
        # Extract source_ids from current entities_context and relations_context
        # Limit each entity/relation to related_chunk_number chunks (using vector similarity if available)
        related_chunk_number = global_config.get("related_chunk_number", 5)

        # Helper function to select top N chunks by similarity
        async def select_top_chunks_by_similarity(chunk_ids: list[str], query: str, limit: int) -> list[str]:
            """Select top N chunks by vector similarity to query."""
            if len(chunk_ids) <= limit:
                return chunk_ids

            # Try to use vector similarity if available
            chunks_vdb = global_config.get("chunks_vdb")
            if chunks_vdb and chunks_vdb.embedding_func:
                try:
                    # Get query embedding
                    query_embedding = await chunks_vdb.embedding_func.func([query])
                    if query_embedding:
                        query_vec = query_embedding[0]

                        # Get chunk vectors and calculate similarities
                        chunk_similarities = []
                        for chunk_id in chunk_ids:
                            chunk_vec = await chunks_vdb.get_vector(chunk_id)
                            if chunk_vec:
                                from .utils import cosine_similarity
                                similarity = cosine_similarity(query_vec, chunk_vec)
                                chunk_similarities.append((chunk_id, similarity))

                        # Sort by similarity and take top N
                        if chunk_similarities:
                            chunk_similarities.sort(key=lambda x: x[1], reverse=True)
                            return [chunk_id for chunk_id, _ in chunk_similarities[:limit]]
                except Exception as e:
                    logger.debug(f"Failed to use vector similarity for chunk selection: {e}")

            # Fallback: just take first N chunks
            return chunk_ids[:limit]

        # Use lists instead of sets to preserve occurrence information
        # Each chunk_id may appear multiple times (from different entities/relations)
        # This is important for accurate occurrence counting in unified processing
        filtered_entity_chunks = []
        for entity in entities_context:
            source_id = entity.get("source_id", "")
            if source_id:
                if GRAPH_FIELD_SEP in source_id:
                    chunk_ids = source_id.split(GRAPH_FIELD_SEP)
                    # Limit to related_chunk_number chunks per entity (with similarity ranking)
                    limited_chunk_ids = await select_top_chunks_by_similarity(chunk_ids, query, related_chunk_number)
                else:
                    limited_chunk_ids = [source_id]

                # Retrieve chunks for this entity (allow duplicates across entities)
                for chunk_id in limited_chunk_ids:
                    chunk_data = await text_chunks_db.get_by_id(chunk_id)
                    if chunk_data:
                        filtered_entity_chunks.append({
                            "content": chunk_data["content"],
                            "file_path": chunk_data.get("file_path", "unknown_source"),
                            "timestamp": chunk_data.get("timestamp", ""),
                            "chunk_id": chunk_id,
                            "source": "entity"
                        })

        filtered_relation_chunks = []
        for relation in relations_context:
            source_id = relation.get("source_id", "")
            if source_id:
                if GRAPH_FIELD_SEP in source_id:
                    chunk_ids = source_id.split(GRAPH_FIELD_SEP)
                    # Limit to related_chunk_number chunks per relation (with similarity ranking)
                    limited_chunk_ids = await select_top_chunks_by_similarity(chunk_ids, query, related_chunk_number)
                else:
                    limited_chunk_ids = [source_id]

                # Retrieve chunks for this relation (allow duplicates across relations)
                for chunk_id in limited_chunk_ids:
                    chunk_data = await text_chunks_db.get_by_id(chunk_id)
                    if chunk_data:
                        filtered_relation_chunks.append({
                            "content": chunk_data["content"],
                            "file_path": chunk_data.get("file_path", "unknown_source"),
                            "timestamp": chunk_data.get("timestamp", ""),
                            "chunk_id": chunk_id,
                            "source": "relation"
                        })

        # Use unified chunk processing on filtered chunks
        logger.debug(f"Starting unified chunk processing: {len(filtered_entity_chunks)} entity + {len(filtered_relation_chunks)} relation chunks")

        final_chunks_from_kg = await _process_all_chunks_unified(
            entity_chunks=filtered_entity_chunks,
            relation_chunks=filtered_relation_chunks,
            entities_context=entities_context,
            relations_context=relations_context,
            query=query,
            text_chunks_db=text_chunks_db,
            chunks_vdb=chunks_vdb,
            query_param=query_param,
            query_embedding=query_embedding,
        )

        # Prepare final chunks for round-robin merge
        # Vector chunks from initial search (if any)
        final_vector_chunks = []
        for chunk in initial_chunks["vector_chunks"]:
            final_vector_chunks.append({
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
                "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "timestamp": chunk.get("timestamp", ""),
                "source": "vector",
            })

        # Chunks from (potentially filtered) entities and relations
        final_unified_chunks = []
        for chunk in final_chunks_from_kg:
            final_unified_chunks.append({
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
                "chunk_id": chunk.get("chunk_id"),
                "timestamp": chunk.get("timestamp", ""),
                "source": "kg",  # From knowledge graph entities/relations
            })
        
        # Apply round-robin merge with deduplication between vector and unified chunks
        final_merged_chunks = []
        seen_chunk_ids = set()
        max_len = max(len(final_vector_chunks), len(final_unified_chunks)) if any([
                     final_vector_chunks, final_unified_chunks]) else 0
        
        for i in range(max_len):
            # Add from each source in round-robin fashion: vector → unified (entity+relation)
            for chunk_list in [final_vector_chunks, final_unified_chunks]:
                if i < len(chunk_list):
                    chunk = chunk_list[i]
                    chunk_id = chunk.get("chunk_id", "")
                    if chunk_id and chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        final_merged_chunks.append({
                            "content": chunk["content"],
                            "file_path": chunk.get("file_path", "unknown_source"),
                            "chunk_id": chunk_id,
                            "timestamp": chunk.get("timestamp", ""),
                        })

        total_chunks_before = len(final_vector_chunks) + len(final_unified_chunks)
        # logger.info(f"Round-robin merged total chunks from {total_chunks_before} to {len(final_merged_chunks)}")
        
        # Apply chunk_top_k truncation if specified
        if query_param.chunk_top_k and len(final_merged_chunks) > query_param.chunk_top_k:
            final_merged_chunks = final_merged_chunks[:query_param.chunk_top_k]
            # logger.info(f"Truncated chunks to chunk_top_k={query_param.chunk_top_k}: {len(final_merged_chunks)} chunks")
        
        # Apply token processing to final merged chunks
        tokenizer = text_chunks_db.global_config.get("tokenizer")
        if final_merged_chunks and tokenizer:
            # Calculate dynamic token limit for text chunks
            # Remove internal fields before sending to LLM
            fields_to_remove = {"source_id", "created_at", "file_path","id"}
            entities_for_llm = [{k: v for k, v in e.items() if k not in fields_to_remove} for e in entities_context]
            relations_for_llm = [{k: v for k, v in r.items() if k not in fields_to_remove} for r in relations_context]
            chunks_for_llm = [{k: v for k, v in c.items() if k != "id"} for c in text_units_context]

            entities_str = json.dumps(entities_for_llm, ensure_ascii=False)
            relations_str = json.dumps(relations_for_llm, ensure_ascii=False)
            
            # Calculate base context tokens (entities + relations + template)
            kg_context_template = """-----Entities(KG)-----

```json
{entities_str}
```

-----Relationships(KG)-----

```json
{relations_str}
```

-----Document Chunks(DC)-----

```json
[]
```

"""
            kg_context = kg_context_template.format(
                entities_str=entities_str, relations_str=relations_str
            )
            kg_context_tokens = len(tokenizer.encode(kg_context))
            
            # Calculate system prompt overhead
            user_prompt = query_param.user_prompt if query_param.user_prompt else ""
            response_type = (
                query_param.response_type
                if query_param.response_type
                else "Multiple Paragraphs"
            )
            
            # Get the system prompt template from PROMPTS
            sys_prompt_template = text_chunks_db.global_config.get(
                "system_prompt_template", PROMPTS["rag_response"]
            )
            
            # Create sample system prompt for overhead calculation
            sample_sys_prompt = sys_prompt_template.format(
                history="",
                context_data="",  # Empty for overhead calculation
                response_type=response_type,
                user_prompt=user_prompt,
            )
            sys_prompt_template_tokens = len(tokenizer.encode(sample_sys_prompt))
            
            # Total system prompt overhead = template + query tokens
            query_tokens = len(tokenizer.encode(query))
            sys_prompt_overhead = sys_prompt_template_tokens + query_tokens
            
            buffer_tokens = 100  # Safety buffer
            
            # Get max_total_tokens from query_param or global_config
            max_total_tokens = getattr(
                query_param,
                "max_total_tokens",
                text_chunks_db.global_config.get("max_total_tokens", 120000),
            )
            
            # Calculate available tokens for text chunks
            used_tokens = kg_context_tokens + sys_prompt_overhead + buffer_tokens
            available_chunk_tokens = max_total_tokens - used_tokens
            
            logger.debug(
                f"Token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_overhead}, "
                f"KG: {kg_context_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
            )
            
            # Apply token truncation to chunks using the dynamic limit
            from lightrag.utils import process_chunks_unified
            final_truncated_chunks = await process_chunks_unified(
                query=query,
                unique_chunks=final_merged_chunks,
                query_param=query_param,
                global_config=text_chunks_db.global_config,
                source_type=query_param.mode,
                chunk_token_limit=available_chunk_tokens,
            )
            
            # Rebuild text_units_context with final processed chunks
            text_units_context = []
            for i, chunk in enumerate(final_truncated_chunks):
                chunk_context = {
                    "id": i + 1,
                    "content": chunk["content"],
                    "file_path": chunk.get("file_path", "unknown_source"),
                }
                # Only add timestamp field if it exists and is not empty
                timestamp = chunk.get("timestamp", "")
                if timestamp:
                    chunk_context["The content is from"] = timestamp
                text_units_context.append(chunk_context)
            
            # logger.info(
            #     f"Final chunk processing: {len(final_merged_chunks)} -> {len(text_units_context)} "
            #    f"(available tokens: {available_chunk_tokens})"
            #)
            
            # Apply chunk tracking if enabled
            if final_truncated_chunks and chunk_tracking:
                chunk_tracking_log = []
                for chunk in final_truncated_chunks:
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id and chunk_id in chunk_tracking:
                        tracking_info = chunk_tracking[chunk_id]
                        source = tracking_info["source"]
                        frequency = tracking_info["frequency"]
                        order = tracking_info["order"]
                        chunk_tracking_log.append(f"{source}{frequency}/{order}")
                    else:
                        chunk_tracking_log.append("?0/0")
                
                if chunk_tracking_log:
                    logger.info(f"chunks: {' '.join(chunk_tracking_log)}")
        
        logger.info(f"Final processed context: {len(entities_context)} entities, {len(relations_context)} relations, {len(text_units_context)} chunks")

    else:
        # No multi-hop expansion, but still need to process initial chunks
        logger.info("No multi-hop expansion, processing initial chunks only")
        
        # Process initial chunks with round-robin merge and truncation
        # Create merged_chunks with source labels
        merged_chunks = []
        
        # Add vector chunks with source label
        for chunk in initial_chunks["vector_chunks"]:
            merged_chunks.append({
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
                "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "timestamp": chunk.get("timestamp", ""),
                "source": "vector",
            })
        
        # Add entity chunks with source label
        for chunk in initial_chunks["entity_chunks"]:
            merged_chunks.append({
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
                "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "timestamp": chunk.get("timestamp", ""),
                "source": "entity",
            })
        
        # Add relation chunks with source label
        for chunk in initial_chunks["relation_chunks"]:
            merged_chunks.append({
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
                "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "timestamp": chunk.get("timestamp", ""),
                "source": "relation",
            })
        
        # Apply round-robin merge with deduplication
        final_merged_chunks = []
        seen_chunk_ids = set()
        
        # Separate chunks by source for round-robin merge
        # IMPORTANT: Preserve the original ordering within each source type
        vector_chunks = [c for c in merged_chunks if c.get("source") == "vector"]
        entity_chunks = [c for c in merged_chunks if c.get("source") == "entity"]
        relation_chunks = [c for c in merged_chunks if c.get("source") == "relation"]
        
        max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks)) if any([
                     vector_chunks, entity_chunks, relation_chunks]) else 0
        
        for i in range(max_len):
            # Add from each source in round-robin fashion
            for chunk_list in [vector_chunks, entity_chunks, relation_chunks]:
                if i < len(chunk_list):
                    chunk = chunk_list[i]
                    chunk_id = chunk.get("chunk_id", "")
                    if chunk_id and chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        final_merged_chunks.append({
                            "content": chunk["content"],
                            "file_path": chunk.get("file_path", "unknown_source"),
                            "chunk_id": chunk_id,
                            "timestamp": chunk.get("timestamp", ""),
                        })
        
        total_chunks_before = len(merged_chunks)
        logger.info(f"Round-robin merged total chunks from {total_chunks_before} to {len(final_merged_chunks)}")
        
        # Apply chunk_top_k truncation if specified
        if query_param.chunk_top_k and len(final_merged_chunks) > query_param.chunk_top_k:
            final_merged_chunks = final_merged_chunks[:query_param.chunk_top_k]
            logger.info(f"Truncated chunks to chunk_top_k={query_param.chunk_top_k}: {len(final_merged_chunks)} chunks")
        
        # Apply token processing to final merged chunks
        tokenizer = text_chunks_db.global_config.get("tokenizer")
        if final_merged_chunks and tokenizer:
            # Same token processing logic as in multi-hop expansion
            # Remove internal fields before token calculation
            fields_to_remove = {"source_id", "created_at", "file_path","id"}
            entities_for_token_calc = [{k: v for k, v in e.items() if k not in fields_to_remove} for e in entities_context]
            relations_for_token_calc = [{k: v for k, v in r.items() if k not in fields_to_remove} for r in relations_context]

            entities_str = json.dumps(entities_for_token_calc, ensure_ascii=False)
            relations_str = json.dumps(relations_for_token_calc, ensure_ascii=False)
            
            kg_context_template = """-----Entities(KG)-----

```json
{entities_str}
```

-----Relationships(KG)-----

```json
{relations_str}
```

-----Document Chunks(DC)-----

```json
[]
```

"""
            kg_context = kg_context_template.format(
                entities_str=entities_str, relations_str=relations_str
            )
            kg_context_tokens = len(tokenizer.encode(kg_context))
            
            user_prompt = query_param.user_prompt if query_param.user_prompt else ""
            response_type = (
                query_param.response_type
                if query_param.response_type
                else "Multiple Paragraphs"
            )
            
            sys_prompt_template = text_chunks_db.global_config.get(
                "system_prompt_template", PROMPTS["rag_response"]
            )
            
            sample_sys_prompt = sys_prompt_template.format(
                history="",
                context_data="",
                response_type=response_type,
                user_prompt=user_prompt,
            )
            sys_prompt_template_tokens = len(tokenizer.encode(sample_sys_prompt))
            
            query_tokens = len(tokenizer.encode(query))
            sys_prompt_overhead = sys_prompt_template_tokens + query_tokens
            buffer_tokens = 100
            
            max_total_tokens = getattr(
                query_param,
                "max_total_tokens",
                text_chunks_db.global_config.get("max_total_tokens", 120000),
            )
            
            used_tokens = kg_context_tokens + sys_prompt_overhead + buffer_tokens
            available_chunk_tokens = max_total_tokens - used_tokens
            
            logger.debug(
                f"Token allocation - Total: {max_total_tokens}, SysPrompt: {sys_prompt_overhead}, "
                f"KG: {kg_context_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
            )
            
            from lightrag.utils import process_chunks_unified
            final_truncated_chunks = await process_chunks_unified(
                query=query,
                unique_chunks=final_merged_chunks,
                query_param=query_param,
                global_config=text_chunks_db.global_config,
                source_type=query_param.mode,
                chunk_token_limit=available_chunk_tokens,
            )
            
            # Rebuild text_units_context with final processed chunks
            text_units_context = []
            for i, chunk in enumerate(final_truncated_chunks):
                chunk_context = {
                    "id": i + 1,
                    "content": chunk["content"],
                    "file_path": chunk.get("file_path", "unknown_source"),
                }
                # Only add timestamp field if it exists and is not empty
                timestamp = chunk.get("timestamp", "")
                if timestamp:
                    chunk_context["The content is from"] = timestamp
                text_units_context.append(chunk_context)
            
            logger.info(
                f"Final chunk processing: {len(final_merged_chunks)} -> {len(text_units_context)} "
                f"(available tokens: {available_chunk_tokens})"
            )
            
            # Apply chunk tracking if enabled
            if final_truncated_chunks and chunk_tracking:
                chunk_tracking_log = []
                for chunk in final_truncated_chunks:
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id and chunk_id in chunk_tracking:
                        tracking_info = chunk_tracking[chunk_id]
                        source = tracking_info["source"]
                        frequency = tracking_info["frequency"]
                        order = tracking_info["order"]
                        chunk_tracking_log.append(f"{source}{frequency}/{order}")
                    else:
                        chunk_tracking_log.append("?0/0")
                
                if chunk_tracking_log:
                    logger.info(f"chunks: {' '.join(chunk_tracking_log)}")

        # Recognition Memory filtering (when multi-hop is disabled)
        if query_param.enable_recognition and (entities_context or relations_context):
            from lightrag.recognition import recognition_memory_filter

            logger.info(
                f"Recognition Memory: filtering {len(entities_context)} entities, "
                f"{len(relations_context)} relations (before filtering)"
            )

            # Convert context format to entity/relation format for filtering
            entities_for_filtering = []
            for entity in entities_context:
                entities_for_filtering.append({
                    "entity_name": entity.get("entity"),
                    "entity_type": entity.get("type"),
                    "description": entity.get("description"),
                    "created_at": entity.get("created_at"),
                    "file_path": entity.get("file_path"),
                    "source_id": entity.get("source_id", ""),
                })

            relations_for_filtering = []
            for relation in relations_context:
                relations_for_filtering.append({
                    "src_id": relation.get("entity1"),
                    "tgt_id": relation.get("entity2"),
                    "description": relation.get("description"),
                    "created_at": relation.get("created_at"),
                    "file_path": relation.get("file_path"),
                    "source_id": relation.get("source_id", ""),
                })

            # Use OpenAI to filter
            tool_llm_model_name = global_config.get("tool_llm_model_name", "gpt-4o-mini")
            filtered_entities, filtered_relations = await recognition_memory_filter(
                query=query,
                entities=entities_for_filtering,
                relations=relations_for_filtering,
                batch_size=query_param.recognition_batch_size,
                tool_llm_model_name=tool_llm_model_name,
                global_config=global_config,
            )

            # Rebuild context with filtered data
            entities_context = []
            for i, n in enumerate(filtered_entities):
                entities_context.append({
                    "id": i + 1,
                    "entity": n.get("entity_name"),
                    "type": n.get("entity_type", "UNKNOWN"),
                    "description": n.get("description", "UNKNOWN"),
                    "created_at": n.get("created_at", "UNKNOWN"),
                    "file_path": n.get("file_path", "unknown_source"),
                    "source_id": n.get("source_id", ""),
                })

            relations_context = []
            for i, e in enumerate(filtered_relations):
                relations_context.append({
                    "id": i + 1,
                    "entity1": e.get("src_id"),
                    "entity2": e.get("tgt_id"),
                    "description": e.get("description", "UNKNOWN"),
                    "created_at": e.get("created_at", "UNKNOWN"),
                    "file_path": e.get("file_path", "unknown_source"),
                    "source_id": e.get("source_id", ""),
                })

         

            # Apply token limits AFTER Recognition Memory filtering (max_hop=0 case)
            entities_context = truncate_list_by_token_size(
                entities_context,
                key=lambda x: json.dumps(x, ensure_ascii=False),
                max_token_size=max_entity_tokens,
                tokenizer=tokenizer,
            )

            relations_context = truncate_list_by_token_size(
                relations_context,
                key=lambda x: json.dumps(x, ensure_ascii=False),
                max_token_size=max_relation_tokens,
                tokenizer=tokenizer,
            )

            # logger.info(f"After token truncation: {len(entities_context)} entities, {len(relations_context)} relations")

            # Note: text_units_context already generated in else block above
            # No need to regenerate chunks here

        # Final processed context log
        logger.info(
            f"Final processed context: {len(entities_context)} entities, "
            f"{len(relations_context)} relations, {len(text_units_context)} chunks"
        )

    # Re-assign IDs after merging expanded data and potential enhancement
        for i, entity in enumerate(entities_context):
            entity["id"] = i + 1
        for i, relation in enumerate(relations_context):
            relation["id"] = i + 1
        for i, chunk in enumerate(text_units_context):
            chunk["id"] = i + 1

        # Regenerate final result with updated context
        # Remove internal fields before sending to LLM
        fields_to_remove = {"source_id", "created_at", "file_path","id"}
        entities_for_llm = [{k: v for k, v in e.items() if k not in fields_to_remove} for e in entities_context]
        relations_for_llm = [{k: v for k, v in r.items() if k not in fields_to_remove} for r in relations_context]
        chunks_for_llm = [{k: v for k, v in c.items() if k != "id"} for c in text_units_context]

        entities_str = json.dumps(entities_for_llm, ensure_ascii=False)
        relations_str = json.dumps(relations_for_llm, ensure_ascii=False)
        text_units_str = json.dumps(chunks_for_llm, ensure_ascii=False)

        result = """-----Entities(KG)-----

```json
{entities_str}
```

-----Relationships(KG)-----

```json
{relations_str}
```

-----Document Chunks(DC)-----

```json
{text_units_str}
```

""".format(
            entities_str=entities_str,
            relations_str=relations_str,
            text_units_str=text_units_str,
        )

    return result


async def _get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str] = None,
):
    # get similar entities
    logger.info(
        f"Query nodes: {query}, top_k: {query_param.top_k}, cosine: {entities_vdb.cosine_better_than_threshold}"
    )

    # Always use regular text embeddings for initial entity retrieval
    results = await entities_vdb.query(query, top_k=query_param.top_k)

    if not len(results):
        return [], []

    # Extract all entity IDs from your results list
    node_ids = [r["entity_name"] for r in results]

    # Call the batch node retrieval and degree functions concurrently.
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(node_ids),
        knowledge_graph_inst.node_degrees_batch(node_ids),
    )

    # Now, if you need the node data and degree in order:
    node_datas = [nodes_dict.get(nid) for nid in node_ids]
    node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {
            **n,
            "entity_name": k["entity_name"],
            "rank": d,
            "created_at": k.get("created_at"),
        }
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    use_relations = await _find_most_related_edges_from_entities(
        node_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Local query: {len(node_datas)} entites, {len(use_relations)} relations"
    )

    # Entities are sorted by cosine similarity
    # Relations are sorted by rank + weight
    return node_datas, use_relations


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    node_names = [dp["entity_name"] for dp in node_datas]
    batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)

    all_edges = []
    seen = set()

    # Create set for fast lookup of entity names (only internal connections between these entities)
    entity_names_set = set(node_names)

    for node_name in node_names:
        this_edges = batch_edges_dict.get(node_name, [])
        for e in this_edges:
            # Only keep edges where both endpoints are in our seed entities (internal connections)
            if e[0] in entity_names_set and e[1] in entity_names_set:
                sorted_edge = tuple(sorted(e))
                if sorted_edge not in seen:
                    seen.add(sorted_edge)
                    all_edges.append(sorted_edge)

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]
    # For edge degrees, use tuples.
    edge_pairs_tuples = list(all_edges)  # all_edges is already a list of tuples

    # Call the batched functions concurrently.
    edge_data_dict, edge_degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
        knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
    )

    # Reconstruct edge_datas list in the same order as the deduplicated results.
    all_edges_data = []
    for pair in all_edges:
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    "Edge %s missing 'weight' attribute, using default value 1.0", pair
                )
                edge_props["weight"] = 1.0

            combined = {
                "src_tgt": pair,
                "rank": edge_degrees_dict.get(pair, 0),
                **edge_props,
            }
            all_edges_data.append(combined)

    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )

    return all_edges_data


async def _find_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
):
    """
    Find text chunks related to entities using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(node_datas)} entities")

    if not node_datas:
        return []

    # Step 1: Collect all text chunks for each entity
    entities_with_chunks = []
    for entity in node_datas:
        if entity.get("source_id"):
            chunks = split_string_by_multi_markers(
                entity["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                entities_with_chunks.append(
                    {
                        "entity_name": entity["entity_name"],
                        "chunks": chunks,
                        "entity_data": entity,
                    }
                )

    if not entities_with_chunks:
        logger.warning("No entities with text chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned entities)
    chunk_occurrence_count = {}
    for entity_info in entities_with_chunks:
        deduplicated_chunks = []
        for chunk_id in entity_info["chunks"]:
            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier entity, so skip it

        # Update entity's chunks to deduplicated chunks
        entity_info["chunks"] = deduplicated_chunks

    # Step 3: Skip sorting at initial stage - will be handled at combined stage
    total_entity_chunks = 0
    for entity_info in entities_with_chunks:
        # Just use chunks as-is without sorting
        entity_info["sorted_chunks"] = entity_info["chunks"]
        total_entity_chunks += len(entity_info["chunks"])

    # Step 4: Return all chunk_ids without complex selection - combined stage will handle sorting
    selected_chunk_ids = []
    for entity_info in entities_with_chunks:
        selected_chunk_ids.extend(entity_info["chunks"])

    logger.info(f"Collected {len(selected_chunk_ids)} entity-related chunks (sorting deferred to combined stage)")

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "entity"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "E",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final entity-related results
                }

    return result_chunks


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    query_param: QueryParam,
):
    logger.info(
        f"Query edges: {keywords}, top_k: {query_param.top_k}, cosine: {relationships_vdb.cosine_better_than_threshold}"
    )

    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return [], []

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in results]
    edge_data_dict = await knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)

    # Reconstruct edge_datas list in the same order as results.
    edge_datas = []
    for k in results:
        pair = (k["src_id"], k["tgt_id"])
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            # Keep edge data without rank, maintain vector search order
            combined = {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "created_at": k.get("created_at", None),
                **edge_props,
            }
            edge_datas.append(combined)

    # Relations maintain vector search order (sorted by similarity)

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas,
        query_param,
        knowledge_graph_inst,
    )

    logger.info(
        f"Global query: {len(use_entities)} entites, {len(edge_datas)} relations"
    )

    return edge_datas, use_entities


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    # Only get nodes data, no need for node degrees
    nodes_dict = await knowledge_graph_inst.get_nodes_batch(entity_names)

    # Rebuild the list in the same order as entity_names
    node_datas = []
    for entity_name in entity_names:
        node = nodes_dict.get(entity_name)
        if node is None:
            logger.warning(f"Node '{entity_name}' not found in batch retrieval.")
            continue
        # Combine the node data with the entity name, no rank needed
        combined = {**node, "entity_name": entity_name}
        node_datas.append(combined)

    return node_datas


async def _find_related_text_unit_from_relations(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    entity_chunks: list[dict] = None,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
):
    """
    Find text chunks related to relationships using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(edge_datas)} relations")

    if not edge_datas:
        return []

    # Step 1: Collect all text chunks for each relationship
    relations_with_chunks = []
    for relation in edge_datas:
        if relation.get("source_id"):
            chunks = split_string_by_multi_markers(
                relation["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                # Build relation identifier
                if "src_tgt" in relation:
                    rel_key = tuple(sorted(relation["src_tgt"]))
                else:
                    rel_key = tuple(
                        sorted([relation.get("src_id"), relation.get("tgt_id")])
                    )

                relations_with_chunks.append(
                    {
                        "relation_key": rel_key,
                        "chunks": chunks,
                        "relation_data": relation,
                    }
                )

    if not relations_with_chunks:
        logger.warning("No relation-related chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned relationships)
    # Also remove duplicates with entity_chunks

    # Extract chunk IDs from entity_chunks for deduplication
    entity_chunk_ids = set()
    if entity_chunks:
        for chunk in entity_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                entity_chunk_ids.add(chunk_id)

    chunk_occurrence_count = {}
    # Track unique chunk_ids that have been removed to avoid double counting
    removed_entity_chunk_ids = set()

    for relation_info in relations_with_chunks:
        deduplicated_chunks = []
        for chunk_id in relation_info["chunks"]:
            # Skip chunks that already exist in entity_chunks
            if chunk_id in entity_chunk_ids:
                # Only count each unique chunk_id once
                removed_entity_chunk_ids.add(chunk_id)
                continue

            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier relationship, so skip it

        # Update relationship's chunks to deduplicated chunks
        relation_info["chunks"] = deduplicated_chunks

    # Check if any relations still have chunks after deduplication
    relations_with_chunks = [
        relation_info
        for relation_info in relations_with_chunks
        if relation_info["chunks"]
    ]

    if not relations_with_chunks:
        logger.info(
            f"Find no additional relations-related chunks from {len(edge_datas)} relations"
        )
        return []

    # Step 3: Skip sorting at initial stage - will be handled at combined stage
    total_relation_chunks = 0
    for relation_info in relations_with_chunks:
        # Just use chunks as-is without sorting
        relation_info["sorted_chunks"] = relation_info["chunks"]
        total_relation_chunks += len(relation_info["chunks"])

    logger.info(
        f"Find {total_relation_chunks} additional chunks in {len(relations_with_chunks)} relations ({len(removed_entity_chunk_ids)} duplicated chunks removed)"
    )

    # Step 4: Return all chunk_ids without complex selection - combined stage will handle sorting
    selected_chunk_ids = []
    for relation_info in relations_with_chunks:
        selected_chunk_ids.extend(relation_info["chunks"])

    logger.info(f"Collected {len(selected_chunk_ids)} relation-related chunks (sorting deferred to combined stage)")

    logger.debug(
        f"KG related chunks: {len(entity_chunks)} from entitys, {len(selected_chunk_ids)} from relations"
    )

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "relationship"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "R",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final relation-related results
                }

    return result_chunks


async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # Apply higher priority (5) to query relation LLM function
        use_model_func = partial(use_model_func, _priority=5)

    # Handle cache
    args_hash = compute_args_hash(
        query_param.mode,
        query,
        query_param.response_type,
        query_param.top_k,
        query_param.chunk_top_k,
        query_param.max_entity_tokens,
        query_param.max_relation_tokens,
        query_param.max_total_tokens,
        query_param.hl_keywords or [],
        query_param.ll_keywords or [],
        query_param.user_prompt or "",
        query_param.enable_rerank,
    )
    cached_response = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    tokenizer: Tokenizer = global_config["tokenizer"]

    chunks = await _get_vector_context(query, chunks_vdb, query_param, None)

    if chunks is None or len(chunks) == 0:
        return PROMPTS["fail_response"]

    # Calculate dynamic token limit for chunks
    # Get token limits from query_param (with fallback to global_config)
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # Calculate conversation history tokens
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )
    history_tokens = len(tokenizer.encode(history_context)) if history_context else 0

    # Calculate system prompt template tokens (excluding content_data)
    user_prompt = query_param.user_prompt if query_param.user_prompt else ""
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # Use the provided system prompt or default
    sys_prompt_template = (
        system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    )

    # Create a sample system prompt with empty content_data to calculate overhead
    sample_sys_prompt = sys_prompt_template.format(
        content_data="",  # Empty for overhead calculation
        response_type=response_type,
        history=history_context,
        user_prompt=user_prompt,
    )
    sys_prompt_template_tokens = len(tokenizer.encode(sample_sys_prompt))

    # Total system prompt overhead = template + query tokens
    query_tokens = len(tokenizer.encode(query))
    sys_prompt_overhead = sys_prompt_template_tokens + query_tokens

    buffer_tokens = 100  # Safety buffer

    # Calculate available tokens for chunks
    used_tokens = sys_prompt_overhead + buffer_tokens
    available_chunk_tokens = max_total_tokens - used_tokens

    logger.debug(
        f"Naive query token allocation - Total: {max_total_tokens}, History: {history_tokens}, SysPrompt: {sys_prompt_overhead}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # Process chunks using unified processing with dynamic token limit
    processed_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=chunks,
        query_param=query_param,
        global_config=global_config,
        source_type="vector",
        chunk_token_limit=available_chunk_tokens,  # Pass dynamic limit
    )

    logger.info(f"Final context: {len(processed_chunks)} chunks")

    # Build text_units_context from processed chunks
    text_units_context = []
    for i, chunk in enumerate(processed_chunks):
        text_units_context.append(
            {
                "id": i + 1,
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
                "The content is from": chunk.get("timestamp", ""),
            }
        )

    # Remove id field before sending to LLM
    chunks_for_llm = [{k: v for k, v in c.items() if k != "id"} for c in text_units_context]
    text_units_str = json.dumps(chunks_for_llm, ensure_ascii=False)
    if query_param.only_need_context:
        return f"""
---Document Chunks(DC)---

```json
{text_units_str}
```

"""
    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # Build system prompt
    user_prompt = (
        query_param.user_prompt
        if query_param.user_prompt
        else PROMPTS["DEFAULT_USER_PROMPT"]
    )
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=text_units_str,
        response_type=query_param.response_type,
        history=history_context,
        user_prompt=user_prompt,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(
        f"[naive_query] Sending to LLM: {len_of_prompts:,} tokens (Query: {len(tokenizer.encode(query))}, System: {len(tokenizer.encode(sys_prompt))})"
    )

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    if hashing_kv.global_config.get("enable_llm_cache"):
        # Save to cache with query parameters
        queryparam_dict = {
            "mode": query_param.mode,
            "response_type": query_param.response_type,
            "top_k": query_param.top_k,
            "chunk_top_k": query_param.chunk_top_k,
            "max_entity_tokens": query_param.max_entity_tokens,
            "max_relation_tokens": query_param.max_relation_tokens,
            "max_total_tokens": query_param.max_total_tokens,
            "hl_keywords": query_param.hl_keywords or [],
            "ll_keywords": query_param.ll_keywords or [],
            "user_prompt": query_param.user_prompt or "",
            "enable_rerank": query_param.enable_rerank,
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                mode=query_param.mode,
                cache_type="query",
                queryparam=queryparam_dict,
            ),
        )

    return response


