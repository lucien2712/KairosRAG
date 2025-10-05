import json
import asyncio
import os
import re
from openai import OpenAI
from .utils import logger
from .prompt import PROMPTS

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


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
    llm_model_name: str = "gpt-4o-mini",  # 新增參數
) -> tuple[list[dict], list[dict]]:
    """
    使用 LLM 過濾 entities/relations（一起處理以保持上下文），完整保留原始數據

    Args:
        query: 用戶查詢
        entities: 3-perspective 檢索到的所有實體
        relations: 3-perspective 檢索到的所有關係
        llm_model_func: LLM 模型函數（使用現有的）
        batch_size: 批次大小
        llm_model_name: LLM 模型名稱

    Returns:
        (filtered_entities, filtered_relations) - 完整保留所有欄位
    """
    if not entities and not relations:
        return [], []

    # 一起處理 entities 和 relations（而非並行分開處理）
    filtered_entities, filtered_relations = await _batch_recognize_combined(
        query, entities, relations, batch_size, llm_model_name
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


async def _call_llm_with_retry(client, prompt: str, max_retries: int = 2, llm_model_name: str = "gpt-4o-mini"):
    """
    調用 LLM 並在失敗時重試

    Args:
        client: OpenAI client
        prompt: LLM prompt
        max_retries: 最大重試次數
        llm_model_name: LLM 模型名稱

    Returns:
        提取的 JSON 結果，或 None（失敗）
    """
    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model=llm_model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                )
            )
            response_text = response.choices[0].message.content

            # 提取 JSON
            result = extract_json_from_response(response_text)

            if result is not None and isinstance(result, dict):
                return result
            else:
                if attempt < max_retries:
                    logger.warning(f"Failed to extract JSON, retrying ({attempt + 1}/{max_retries})...")
                    logger.debug(f"Response was: {response_text[:300]}")
                else:
                    logger.warning(f"Failed to extract JSON after {max_retries + 1} attempts")
                    logger.warning(f"Final response was: {response_text[:300]}")

        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"LLM call failed: {e}, retrying ({attempt + 1}/{max_retries})...")
            else:
                logger.error(f"LLM call failed after {max_retries + 1} attempts: {e}")

    return None


async def _batch_recognize_combined(
    query: str,
    entities: list[dict],
    relations: list[dict],
    batch_size: int,
    llm_model_name: str = "gpt-4o-mini",
) -> tuple[list[dict], list[dict]]:
    """
    一起處理 entities 和 relations，讓 LLM 在評估 relations 時能看到 entity 資訊

    處理流程：
    1. 以 relation 為主進行 batch 切分
    2. 每個 batch 包含該 batch 的 relations + 這些 relations 涉及的所有 entities
    3. 最後處理沒有出現在任何 relation 中的落單 entities
    """
    client = OpenAI()

    # 為 relations 創建 ID 映射（完整保留原始 dict）
    id_mapping = {}
    for i, rel in enumerate(relations):
        # 生成唯一 ID
        rel_id = f"rel_{i}"
        id_mapping[rel_id] = rel

    # 建立 entity name 到 entity data 的映射
    entity_map = {e.get("entity_name"): e for e in entities}

    filtered_entities = []
    filtered_relations = []
    entities_in_relations = set()  # 追蹤哪些 entities 出現在 relations 中

    # Phase 1: 處理 relations (以 relation 為中心的 batch)
    relation_items = list(id_mapping.items())
    for i in range(0, len(relation_items), batch_size):
        relation_batch_items = relation_items[i : i + batch_size]
        relation_batch_ids = {rel_id: rel_data for rel_id, rel_data in relation_batch_items}

        # 收集這個 batch 的 relations 涉及的所有 entities
        batch_entity_names = set()
        for rel_id, rel_data in relation_batch_items:
            src_id = rel_data.get("src_id")
            tgt_id = rel_data.get("tgt_id")
            if src_id:
                batch_entity_names.add(src_id)
                entities_in_relations.add(src_id)
            if tgt_id:
                batch_entity_names.add(tgt_id)
                entities_in_relations.add(tgt_id)

        # 構建這個 batch 的 entities（只包含 relation 涉及的 entities）
        entity_batch = [entity_map[name] for name in batch_entity_names if name in entity_map]

        # 構建 entities JSON
        entities_data = [
            {
                "entity_name": e.get("entity_name"),
                "description": e.get("description", "")
            }
            for e in entity_batch
        ]

        # 構建 relations JSON
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

        prompt = PROMPTS["recognition_combined_filter"].format(
            query=query,
            entities_json=entities_json,
            relations_json=relations_json
        )

        try:
            # 調用 OpenAI with retry
            result = await _call_llm_with_retry(client, prompt, max_retries=2, llm_model_name=llm_model_name)

            # 如果提取失敗（經過 retry 後仍失敗），保留所有資料
            if result is None:
                logger.warning(f"Failed to get valid response for relation batch {i//batch_size + 1} after retries, keeping all items")
                filtered_entities.extend(entity_batch)
                # 保留所有 relations
                for rel_id, rel_data in relation_batch_ids.items():
                    filtered_relations.append(rel_data)
                continue

            # 獲取要移除的 IDs
            removed_entity_ids = set(result.get("irrelevant_entity_ids", []))
            removed_relation_ids = set(result.get("irrelevant_relation_ids", []))

            # 保留不在移除列表中的 entities（relation batch 中涉及的 entities）
            for entity in entity_batch:
                entity_name = entity.get("entity_name")
                if entity_name not in removed_entity_ids:
                    # 檢查是否已經加入過（避免重複）
                    if not any(e.get("entity_name") == entity_name for e in filtered_entities):
                        filtered_entities.append(entity)

            # 保留所有不在移除列表中的 relations
            for rel_id, rel_data in relation_batch_ids.items():
                if rel_id not in removed_relation_ids:
                    filtered_relations.append(rel_data)

        except Exception as e:
            logger.warning(
                f"Recognition combined filter failed for relation batch {i//batch_size + 1}: {e}. "
                f"Keeping all {len(relation_batch_ids)} relations in this batch."
            )
            # 失敗時保留所有 relations
            for rel_id, rel_data in relation_batch_ids.items():
                filtered_relations.append(rel_data)

    # Phase 2: 處理落單的 entities（沒有出現在任何 relation 中的）
    orphan_entities = [e for e in entities if e.get("entity_name") not in entities_in_relations]

    if orphan_entities:
        # logger.info(f"Processing {len(orphan_entities)} orphan entities (not in any relations)")

        # 對落單 entities 進行 batch 處理
        for i in range(0, len(orphan_entities), batch_size):
            orphan_batch = orphan_entities[i : i + batch_size]

            # 構建 entities JSON（只有 entities，沒有 relations）
            entities_data = [
                {
                    "entity_name": e.get("entity_name"),
                    "description": e.get("description", "")
                }
                for e in orphan_batch
            ]

            entities_json = json.dumps(entities_data, ensure_ascii=False, indent=2)
            relations_json = "[]"  # 空的 relations

            prompt = PROMPTS["recognition_combined_filter"].format(
                query=query,
                entities_json=entities_json,
                relations_json=relations_json
            )

            try:
                # 調用 OpenAI with retry
                result = await _call_llm_with_retry(client, prompt, max_retries=2, llm_model_name=llm_model_name)

                # 如果提取失敗（經過 retry 後仍失敗），保留所有資料
                if result is None:
                    logger.warning(f"Failed to get valid response for orphan batch {i//batch_size + 1} after retries, keeping all items")
                    filtered_entities.extend(orphan_batch)
                    continue

                # 獲取要移除的 entity IDs
                removed_entity_ids = set(result.get("irrelevant_entity_ids", []))

                # 保留不在移除列表中的 entities
                for entity in orphan_batch:
                    entity_name = entity.get("entity_name")
                    if entity_name not in removed_entity_ids:
                        filtered_entities.append(entity)

            except Exception as e:
                logger.warning(
                    f"Recognition orphan filter failed for batch {i//batch_size + 1}: {e}. "
                    f"Keeping all {len(orphan_batch)} orphan entities in this batch."
                )
                filtered_entities.extend(orphan_batch)

    # Fallback: 如果全部被過濾，保留 top-3
    if not filtered_entities and entities:
        logger.warning("Recognition filtered all entities, keeping top 3 as fallback")
        filtered_entities = entities[:3]

    if not filtered_relations and relations:
        logger.warning("Recognition filtered all relations, keeping top 3 as fallback")
        filtered_relations = relations[:3]

    return filtered_entities, filtered_relations


