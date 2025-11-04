import json
import asyncio
import os
import re
from openai import OpenAI
from .utils import logger
from .prompt import PROMPTS


def extract_json_from_response(response_content: str):
    """從 LLM 回應中提取 JSON 對象（支持 {} 和 []）

    Returns:
        dict or None: 提取的 JSON 對象，如果提取失敗則返回 None
    """
    try:
        # Try to parse the entire response first (most common case)
        response_content = response_content.strip()
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object by matching braces
        # Look for the first { and matching closing }
        stack = []
        start_idx = -1

        for i, char in enumerate(response_content):
            if char == '{':
                if not stack:
                    start_idx = i
                stack.append('{')
            elif char == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                    if not stack and start_idx != -1:
                        # Found complete JSON object
                        json_str = response_content[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            continue

        # Fallback: try to find array
        for i, char in enumerate(response_content):
            if char == '[':
                if not stack:
                    start_idx = i
                stack.append('[')
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                    if not stack and start_idx != -1:
                        # Found complete JSON array
                        json_str = response_content[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            continue

        logger.warning("No valid JSON found in response")
        logger.debug(f"Response content: {response_content[:500]}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in JSON extraction: {e}")
        logger.debug(f"Response content: {response_content[:500]}")
        return None


async def recognition_memory_filter(
    query: str,
    entities: list[dict],
    relations: list[dict],
    llm_model_func=None,  # 不再需要，保留參數為了向後兼容
    batch_size: int = 10,
    tool_llm_model_name: str = "gpt-4o-mini",  # 工具專用 LLM（僅支援 OpenAI-compatible API）
    global_config: dict = None,  # 新增參數：用於獲取 llm_model_max_async
    openai_client = None,  # 新增參數：共用的 OpenAI client（如果為 None 則建立新的）
    token_tracker = None,  # 新增參數：token tracker
) -> tuple[list[dict], list[dict]]:
    """
    使用 LLM 過濾 entities/relations（一起處理以保持上下文），完整保留原始數據

    Args:
        query: 用戶查詢
        entities: 3-perspective 檢索到的所有實體
        relations: 3-perspective 檢索到的所有關係
        llm_model_func: LLM 模型函數（已棄用，保留為了向後兼容）
        batch_size: 批次大小
        tool_llm_model_name: 工具專用 LLM 模型名稱（僅支援 OpenAI-compatible API）
        global_config: 全局配置字典，用於獲取 llm_model_max_async

    Returns:
        (filtered_entities, filtered_relations) - 完整保留所有欄位
    """
    if not entities and not relations:
        return [], []

    # 一起處理 entities 和 relations（而非並行分開處理）
    filtered_entities, filtered_relations = await _batch_recognize_combined(
        query, entities, relations, batch_size, tool_llm_model_name, global_config, openai_client, token_tracker
    )

    # 統計日誌
    entity_filter_rate = (1 - len(filtered_entities) / len(entities)) * 100 if entities else 0
    relation_filter_rate = (1 - len(filtered_relations) / len(relations)) * 100 if relations else 0

    logger.info(
        f"Recognition Memory: "
        f"entities {len(entities)}->{len(filtered_entities)} ({entity_filter_rate:.1f}% filtered), "
        f"relations {len(relations)}->{len(filtered_relations)} ({relation_filter_rate:.1f}% filtered)"
    )

    return filtered_entities, filtered_relations


async def _call_llm_with_retry(client, prompt: str, max_retries: int = 2, tool_llm_model_name: str = "gpt-4o-mini", token_tracker=None):
    """
    調用 LLM 並在失敗時重試

    Args:
        client: OpenAI client
        prompt: LLM prompt
        max_retries: 最大重試次數
        tool_llm_model_name: 工具專用 LLM 模型名稱（僅支援 OpenAI-compatible API）
        token_tracker: Optional TokenTracker instance for tracking token usage

    Returns:
        提取的 JSON 結果，或 None（失敗）
    """

    for attempt in range(max_retries + 1):
        try:
            # Prepare API call parameters
            api_params = {
                "model": tool_llm_model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }

            # Add reasoning_effort for GPT-5 series models
            if tool_llm_model_name.startswith("gpt-5"):
                api_params["reasoning_effort"] = "minimal"

            # Add timeout protection to prevent indefinite waiting
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: client.chat.completions.create(**api_params)
                ),
                timeout=180.0  # 3 minutes timeout
            )
            response_text = response.choices[0].message.content

            # Track token usage if token_tracker is provided
            if token_tracker and hasattr(response, 'usage') and response.usage:
                token_counts = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0),
                }
                token_tracker.add_usage(token_counts)

            # 提取 JSON
            result = extract_json_from_response(response_text)

            if result is not None and isinstance(result, dict):
                # Validate that ID lists contain only strings (no nested lists)
                def is_valid_id_list(ids: list) -> bool:
                    """Check if all elements in list are strings (no nested lists)."""
                    if not isinstance(ids, list):
                        return False
                    return all(isinstance(item, str) for item in ids)

                # Check both ID lists
                entity_ids = result.get("irrelevant_entity_ids", [])
                relation_ids = result.get("irrelevant_relation_ids", [])

                entity_ids_valid = is_valid_id_list(entity_ids) if entity_ids else True
                relation_ids_valid = is_valid_id_list(relation_ids) if relation_ids else True

                if not entity_ids_valid or not relation_ids_valid:
                    if attempt < max_retries:
                        logger.warning(
                            f"ID lists contain nested lists or non-string items. "
                            f"Entity IDs valid: {entity_ids_valid}, Relation IDs valid: {relation_ids_valid}. "
                            f"Retrying ({attempt + 1}/{max_retries})..."
                        )
                        continue
                    else:
                        logger.error(f"ID lists still invalid after {max_retries + 1} attempts")
                        return None

                return result
            else:
                if attempt < max_retries:
                    logger.warning(f"Failed to extract JSON, retrying ({attempt + 1}/{max_retries})...")
                    logger.debug(f"Response was: {response_text[:300]}")
                else:
                    logger.warning(f"Failed to extract JSON after {max_retries + 1} attempts")
                    logger.warning(f"Final response was: {response_text[:300]}")

        except asyncio.TimeoutError:
            if attempt < max_retries:
                logger.warning(f"LLM call timeout (180s), retrying ({attempt + 1}/{max_retries})...")
            else:
                logger.error(f"LLM call timeout after {max_retries + 1} attempts")
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"LLM call failed: {e}, retrying ({attempt + 1}/{max_retries})...")
            else:
                logger.error(f"LLM call failed after {max_retries + 1} attempts: {e}")

    return None


async def _process_single_relation_batch(
    query: str,
    relation_batch_items: list[tuple],
    entity_map: dict,
    client,
    tool_llm_model_name: str,
    batch_index: int,
    token_tracker=None,
) -> dict:
    """處理單個 relation batch 並返回結果"""
    relation_batch_ids = {rel_id: rel_data for rel_id, rel_data in relation_batch_items}

    # 收集這個 batch 的 relations 涉及的所有 entities
    batch_entity_names = set()
    entities_in_batch = set()
    for rel_id, rel_data in relation_batch_items:
        src_id = rel_data.get("src_id")
        tgt_id = rel_data.get("tgt_id")
        if src_id:
            batch_entity_names.add(src_id)
            entities_in_batch.add(src_id)
        if tgt_id:
            batch_entity_names.add(tgt_id)
            entities_in_batch.add(tgt_id)

    # 構建這個 batch 的 entities
    entity_batch = [entity_map[name] for name in batch_entity_names if name in entity_map]

    # 構建 JSON
    entities_data = [
        {"entity_name": e.get("entity_name"), "description": e.get("description", "")}
        for e in entity_batch
    ]
    relations_data = [
        {
            "id": rel_id,
            "src_id": rel_data.get("src_id"),
            "tgt_id": rel_data.get("tgt_id"),
            "description": rel_data.get("description", "")
        }
        for rel_id, rel_data in relation_batch_ids.items()
    ]

    entities_json = json.dumps(entities_data, ensure_ascii=False, indent=2)
    relations_json = json.dumps(relations_data, ensure_ascii=False, indent=2)

    prompt = PROMPTS["recognition_filter"].format(
        query=query,
        entities_json=entities_json,
        relations_json=relations_json
    )

    try:
        result = await _call_llm_with_retry(client, prompt, max_retries=2, tool_llm_model_name=tool_llm_model_name, token_tracker=token_tracker)

        if result is None:
            logger.warning(f"Failed to get valid response for relation batch {batch_index} after retries, keeping all items")
            return {
                "entities": entity_batch,
                "relations": list(relation_batch_ids.values()),
                "entities_in_relations": entities_in_batch,
            }

        # 獲取要移除的 IDs
        removed_entity_ids = set(result.get("irrelevant_entity_ids", []))
        removed_relation_ids = set(result.get("irrelevant_relation_ids", []))

        # 過濾 entities 和 relations
        kept_entities = [e for e in entity_batch if e.get("entity_name") not in removed_entity_ids]
        kept_relations = [rel_data for rel_id, rel_data in relation_batch_ids.items() if rel_id not in removed_relation_ids]

        return {
            "entities": kept_entities,
            "relations": kept_relations,
            "entities_in_relations": entities_in_batch,
        }

    except Exception as e:
        logger.warning(f"Recognition filter failed for relation batch {batch_index}: {e}. Keeping all items.")
        return {
            "entities": entity_batch,
            "relations": list(relation_batch_ids.values()),
            "entities_in_relations": entities_in_batch,
        }


async def _process_single_orphan_batch(
    query: str,
    orphan_batch: list[dict],
    client,
    tool_llm_model_name: str,
    batch_index: int,
    token_tracker=None,
) -> list[dict]:
    """處理單個 orphan entity batch 並返回過濾後的 entities"""
    entities_data = [
        {"entity_name": e.get("entity_name"), "description": e.get("description", "")}
        for e in orphan_batch
    ]

    entities_json = json.dumps(entities_data, ensure_ascii=False, indent=2)
    relations_json = "[]"

    prompt = PROMPTS["recognition_filter"].format(
        query=query,
        entities_json=entities_json,
        relations_json=relations_json
    )

    try:
        result = await _call_llm_with_retry(client, prompt, max_retries=2, tool_llm_model_name=tool_llm_model_name, token_tracker=token_tracker)

        if result is None:
            logger.warning(f"Failed to get valid response for orphan batch {batch_index} after retries, keeping all items")
            return orphan_batch

        removed_entity_ids = set(result.get("irrelevant_entity_ids", []))
        kept_entities = [e for e in orphan_batch if e.get("entity_name") not in removed_entity_ids]

        return kept_entities

    except Exception as e:
        logger.warning(f"Recognition orphan filter failed for batch {batch_index}: {e}. Keeping all items.")
        return orphan_batch


async def _batch_recognize_combined(
    query: str,
    entities: list[dict],
    relations: list[dict],
    batch_size: int,
    tool_llm_model_name: str = "gpt-4o-mini",
    global_config: dict = None,
    openai_client = None,  # 新增參數：共用的 OpenAI client
    token_tracker = None,  # 新增參數：token tracker
) -> tuple[list[dict], list[dict]]:
    """
    一起處理 entities 和 relations，讓 LLM 在評估 relations 時能看到 entity 資訊

    使用 asyncio.gather() 並行處理多個 batches 以提升速度

    處理流程：
    1. 並行處理所有 relation batches
    2. 並行處理所有 orphan entity batches
    3. 合併所有結果
    """
    # 使用傳入的 client，或 fallback 到建立新的（向後兼容）
    if openai_client is None:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        )
    else:
        client = openai_client

    # 為 relations 創建 ID 映射
    id_mapping = {}
    for i, rel in enumerate(relations):
        rel_id = f"rel_{i}"
        id_mapping[rel_id] = rel

    # 建立 entity name 到 entity data 的映射
    entity_map = {e.get("entity_name"): e for e in entities}

    # Phase 1: 並行處理所有 relation batches
    relation_items = list(id_mapping.items())
    relation_tasks = []

    # 使用 Semaphore 限制並發數（避免超過 API 限制）
    # 從 global_config 獲取 llm_model_max_async（默認值 4，可通過環境變量 MAX_ASYNC 修改）
    max_concurrent = global_config.get("llm_model_max_async", 4) if global_config else 4
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(batch_items, batch_idx):
        async with semaphore:
            return await _process_single_relation_batch(
                query, batch_items, entity_map, client, tool_llm_model_name, batch_idx, token_tracker
            )

    for i in range(0, len(relation_items), batch_size):
        relation_batch_items = relation_items[i : i + batch_size]
        batch_idx = i // batch_size + 1
        task = process_with_semaphore(relation_batch_items, batch_idx)
        relation_tasks.append(task)

    # 並行執行所有 relation batch 任務
    relation_results = await asyncio.gather(*relation_tasks, return_exceptions=True)

    # 合併結果
    filtered_entities = []
    filtered_relations = []
    entities_in_relations = set()
    seen_entities = set()  # 用於去重

    for result in relation_results:
        if isinstance(result, Exception):
            logger.error(f"Relation batch processing failed with exception: {result}")
            continue

        # 收集 entities（去重）
        for entity in result["entities"]:
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                filtered_entities.append(entity)
                seen_entities.add(entity_name)

        # 收集 relations
        filtered_relations.extend(result["relations"])

        # 收集 entities_in_relations
        entities_in_relations.update(result["entities_in_relations"])

    # Phase 2: 並行處理所有 orphan entity batches
    orphan_entities = [e for e in entities if e.get("entity_name") not in entities_in_relations]

    if orphan_entities:
        orphan_tasks = []

        async def process_orphan_with_semaphore(orphan_batch, batch_idx):
            async with semaphore:
                return await _process_single_orphan_batch(
                    query, orphan_batch, client, tool_llm_model_name, batch_idx, token_tracker
                )

        for i in range(0, len(orphan_entities), batch_size):
            orphan_batch = orphan_entities[i : i + batch_size]
            batch_idx = i // batch_size + 1
            task = process_orphan_with_semaphore(orphan_batch, batch_idx)
            orphan_tasks.append(task)

        # 並行執行所有 orphan batch 任務
        orphan_results = await asyncio.gather(*orphan_tasks, return_exceptions=True)

        # 合併 orphan 結果
        for result in orphan_results:
            if isinstance(result, Exception):
                logger.error(f"Orphan batch processing failed with exception: {result}")
                continue

            # 收集 orphan entities（去重）
            for entity in result:
                entity_name = entity.get("entity_name")
                if entity_name and entity_name not in seen_entities:
                    filtered_entities.append(entity)
                    seen_entities.add(entity_name)

    # Fallback: 只有在 entities 和 relations 同時都被完全移除時，才保留全部
    if not filtered_entities and not filtered_relations and (entities or relations):
        logger.warning("Recognition filtered all entities AND relations, keeping all data as fallback")
        filtered_entities = entities
        filtered_relations = relations

    return filtered_entities, filtered_relations


