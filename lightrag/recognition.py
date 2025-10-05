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
    使用 LLM 過濾 entities/relations，完整保留原始數據

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

    # 並行處理 entities 和 relations
    tasks = []
    if entities:
        tasks.append(_batch_recognize_entities(query, entities, batch_size))
    if relations:
        tasks.append(_batch_recognize_relations(query, relations, batch_size))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 處理結果，檢查是否有異常
    filtered_entities = []
    filtered_relations = []

    if entities:
        result = results[0] if len(results) > 0 else []
        if isinstance(result, Exception):
            logger.error(f"Entity filtering failed: {type(result).__name__}: {result}")
            import traceback
            logger.error(f"Traceback: {''.join(traceback.format_exception(type(result), result, result.__traceback__))}")
            filtered_entities = entities  # Fallback: 使用原始數據
        else:
            filtered_entities = result

    if relations:
        idx = 1 if entities else 0
        result = results[idx] if len(results) > idx else []
        if isinstance(result, Exception):
            logger.error(f"Relation filtering failed: {type(result).__name__}: {result}")
            import traceback
            logger.error(f"Traceback: {''.join(traceback.format_exception(type(result), result, result.__traceback__))}")
            filtered_relations = relations  # Fallback: 使用原始數據
        else:
            filtered_relations = result

    # 統計日誌
    entity_filter_rate = (1 - len(filtered_entities) / len(entities)) * 100 if entities else 0
    relation_filter_rate = (1 - len(filtered_relations) / len(relations)) * 100 if relations else 0

    logger.info(
        f"Recognition Memory: "
        f"entities {len(entities)}->{len(filtered_entities)} ({entity_filter_rate:.1f}% filtered), "
        f"relations {len(relations)}->{len(filtered_relations)} ({relation_filter_rate:.1f}% filtered)"
    )

    return filtered_entities, filtered_relations


async def _batch_recognize_entities(
    query: str,
    entities: list[dict],
    batch_size: int,
) -> list[dict]:
    """批次過濾實體，完整保留原始數據"""
    if not entities:
        return []

    filtered = []

    for i in range(0, len(entities), batch_size):
        batch = entities[i:i+batch_size]

        # 構建 LLM 輸入 JSON（只發送實體資訊，query 分開傳遞）
        entities_data = [
            {
                "entity_name": e.get("entity_name"),
                "description": e.get("description", "")
            }
            for e in batch
        ]

        entities_json = json.dumps(entities_data, ensure_ascii=False, indent=2)
        prompt = PROMPTS["recognition_entity_filter"].format(
            query=query,
            entities_json=entities_json
        )

        try:
            # 調用 OpenAI (不使用 response_format，而是透過 prompt 要求 JSON)
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
            logger.debug(f"Entity filter LLM raw response: {response_text[:500]}")

            # 使用專門的 JSON 提取函數
            result = extract_json_from_response(response_text)

            # 如果提取失敗，保留所有實體作為 fallback
            if result is None or not isinstance(result, dict):
                logger.warning(f"Failed to extract JSON from LLM response for batch {i//batch_size + 1}, keeping all entities")
                logger.warning(f"LLM response was: {response_text[:200]}")
                filtered.extend(batch)
                continue

            # 獲取要移除的實體 IDs（新格式）
            removed_ids = set(result.get("irrelevant_entity_ids", []))

            # 保留所有不在移除列表中的實體（完整保留原始 dict）
            for entity in batch:
                if entity.get("entity_name") not in removed_ids:
                    filtered.append(entity)  # 完整保留原始 dict

        except Exception as e:
            logger.warning(
                f"Recognition entity filter failed for batch {i//batch_size + 1}: {e}. "
                f"Keeping all {len(batch)} entities in this batch."
            )
            filtered.extend(batch)  # Fallback: 保留所有

    # Fallback: 如果全部被過濾，保留 top-3
    if not filtered and entities:
        logger.warning("Recognition filtered all entities, keeping top 3 as fallback")
        filtered = entities[:3]

    return filtered


async def _batch_recognize_relations(
    query: str,
    relations: list[dict],
    batch_size: int,
) -> list[dict]:
    """批次過濾關係，完整保留原始數據"""
    if not relations:
        return []

    filtered = []

    for i in range(0, len(relations), batch_size):
        batch = relations[i:i+batch_size]

        # 為每個 relation 生成臨時 ID（用於 LLM 回傳）
        id_mapping = {}
        relations_data = []

        for idx, r in enumerate(batch):
            rel_id = f"rel_{i+idx}"
            id_mapping[rel_id] = r  # 儲存 ID 到原始數據的映射

            relations_data.append({
                "id": rel_id,
                "src_id": r.get("src_id", ""),
                "tgt_id": r.get("tgt_id", ""),
                "description": r.get("description", "")
            })

        relations_json = json.dumps(relations_data, ensure_ascii=False, indent=2)
        prompt = PROMPTS["recognition_relation_filter"].format(
            query=query,
            relations_json=relations_json
        )

        try:
            # 調用 OpenAI (不使用 response_format，而是透過 prompt 要求 JSON)
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
            logger.debug(f"Relation filter LLM raw response: {response_text[:500]}")

            # 使用專門的 JSON 提取函數
            result = extract_json_from_response(response_text)

            # 如果提取失敗，保留所有關係作為 fallback
            if result is None or not isinstance(result, dict):
                logger.warning(f"Failed to extract JSON from LLM response for batch {i//batch_size + 1}, keeping all relations")
                logger.warning(f"LLM response was: {response_text[:200]}")
                filtered.extend(batch)
                continue

            # 獲取要移除的關係 IDs（新格式）
            removed_ids = set(result.get("irrelevant_relation_ids", []))

            # 保留所有不在移除列表中的關係（完整保留原始 dict）
            for rel_id, rel_data in id_mapping.items():
                if rel_id not in removed_ids:
                    filtered.append(rel_data)  # 完整保留原始 dict

        except Exception as e:
            logger.warning(
                f"Recognition relation filter failed for batch {i//batch_size + 1}: {e}. "
                f"Keeping all {len(batch)} relations in this batch."
            )
            filtered.extend(batch)  # Fallback: 保留所有

    # Fallback: 如果全部被過濾，保留 top-3
    if not filtered and relations:
        logger.warning("Recognition filtered all relations, keeping top 3 as fallback")
        filtered = relations[:3]

    return filtered
