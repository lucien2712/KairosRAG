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
) -> tuple[list[dict], list[dict]]:
    """
    使用 LLM 過濾 entities/relations（一起處理以保持上下文），完整保留原始數據

    Args:
        query: 用戶查詢
        entities: 3-perspective 檢索到的所有實體
        relations: 3-perspective 檢索到的所有關係
        llm_model_func: LLM 模型函數（使用現有的）
        batch_size: 批次大小

    Returns:
        (filtered_entities, filtered_relations) - 完整保留所有欄位
    """
    if not entities and not relations:
        return [], []

    # 一起處理 entities 和 relations（而非並行分開處理）
    filtered_entities, filtered_relations = await _batch_recognize_combined(
        query, entities, relations, batch_size
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


async def _batch_recognize_combined(
    query: str,
    entities: list[dict],
    relations: list[dict],
    batch_size: int,
) -> tuple[list[dict], list[dict]]:
    """
    一起處理 entities 和 relations，讓 LLM 在評估 relations 時能看到 entity 資訊
    """
    client = OpenAI()

    # 為 relations 創建 ID 映射（完整保留原始 dict）
    id_mapping = {}
    for i, rel in enumerate(relations):
        # 生成唯一 ID
        rel_id = f"rel_{i}"
        id_mapping[rel_id] = rel

    filtered_entities = []
    filtered_relations = []

    # 批次處理（但現在 entities 和 relations 一起處理）
    total_items = max(len(entities), len(relations))
    for i in range(0, total_items, batch_size):
        entity_batch = entities[i : i + batch_size] if entities else []
        relation_batch_items = list(id_mapping.items())[i : i + batch_size] if relations else []
        relation_batch = [rel_data for _, rel_data in relation_batch_items]
        relation_batch_ids = {rel_id: rel_data for rel_id, rel_data in relation_batch_items}

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
            # 調用 OpenAI
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                )
            )
            response_text = response.choices[0].message.content
            logger.debug(f"Combined filter LLM raw response: {response_text[:500]}")

            # 提取 JSON
            result = extract_json_from_response(response_text)

            # 如果提取失敗，保留所有資料
            if result is None or not isinstance(result, dict):
                logger.warning(f"Failed to extract JSON from LLM response for batch {i//batch_size + 1}, keeping all items")
                logger.warning(f"LLM response was: {response_text[:200]}")
                filtered_entities.extend(entity_batch)
                filtered_relations.extend(relation_batch)
                continue

            # 獲取要移除的 IDs
            removed_entity_ids = set(result.get("irrelevant_entity_ids", []))
            removed_relation_ids = set(result.get("irrelevant_relation_ids", []))

            # 保留所有不在移除列表中的 entities
            for entity in entity_batch:
                if entity.get("entity_name") not in removed_entity_ids:
                    filtered_entities.append(entity)

            # 保留所有不在移除列表中的 relations
            for rel_id, rel_data in relation_batch_ids.items():
                if rel_id not in removed_relation_ids:
                    filtered_relations.append(rel_data)

        except Exception as e:
            logger.warning(
                f"Recognition combined filter failed for batch {i//batch_size + 1}: {e}. "
                f"Keeping all {len(entity_batch)} entities and {len(relation_batch)} relations in this batch."
            )
            filtered_entities.extend(entity_batch)
            filtered_relations.extend(relation_batch)

    # Fallback: 如果全部被過濾，保留 top-3
    if not filtered_entities and entities:
        logger.warning("Recognition filtered all entities, keeping top 3 as fallback")
        filtered_entities = entities[:3]

    if not filtered_relations and relations:
        logger.warning("Recognition filtered all relations, keeping top 3 as fallback")
        filtered_relations = relations[:3]

    return filtered_entities, filtered_relations


