#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entity 去重與合併工具（以 LLM 決策、Neo4j 實際合併）

功能概要：
1) 從 Neo4j 讀出每個節點的 (entity_id, description)
2) 兩兩比對，交給 LLM（有工具使用權）判斷是否為同一實體
3) 若 LLM「非常確定（>95%）」是同一實體，則呼叫 merge() 工具，
   以 rag.merge_entities(...) 將 B 併入 A
4) 合併策略：
   - created_at: keep_last
   - description: concatenate
   - entity_type: keep_first
   - source_id: join_unique
   - file_path: join_unique

注意：
- 需先在環境變數設定 Neo4j 與 LLM 的連線資訊
  NEO4J_URI, NEO4J_USER, NEO4J_PASS
  LLM_BASE_URL, LLM_API_KEY, LLM_MODEL（例如 qwen3-235b-fp8）
"""

import os
import json
from typing import List, Tuple

from neo4j import GraphDatabase
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 你專案裡的 RAG 初始化（照你的截圖）
# 若無此模組，請自行改成實際的初始化方式
from config import initialize_rag

# -------------------------
# 初始化 RAG / Neo4j / LLM
# -------------------------

rag = initialize_rag()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASS", "password"),
)

# LLM：改成走環境變數，避免把金鑰寫死在程式
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1/")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "dummy")
LLM_MODEL    = os.getenv("LLM_MODEL",    "qwen3-235b-fp8")

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)

# -------------------------------------------------
# 1) 從 Neo4j 將所有 node 的 (entity_id, description) 拉出
# -------------------------------------------------
def fetch_entities() -> List[Tuple[str, str]]:
    q = """
    MATCH (n)
    WHERE exists(n.entity_id) AND exists(n.description)
    RETURN n.entity_id AS entity_id, n.description AS description
    """
    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver, driver.session() as sess:
        return [(r["entity_id"], r["description"]) for r in sess.run(q)]

# -------------------------------------------------
# 2) 提供給 LLM 可呼叫的 merge 工具（真的執行合併）
# -------------------------------------------------
@tool
def merge(a_entity_id: str, b_entity_id: str) -> str:
    """
    若兩個 entity 代表相同實體，將 b_entity_id 併入 a_entity_id。
    """
    rag.merge_entities(
        source_entities=[a_entity_id, b_entity_id],
        target_entity=a_entity_id,
        merge_strategy={
            "created_at": "keep_last",
            "description": "concatenate",   # 將描述串接
            "entity_type": "keep_first",    # 以 A 的 entity_type 為主
            "source_id":   "join_unique",   # 合併不重複的來源 id
            "file_path":   "join_unique",   # 合併不重複的檔案路徑
        },
    )
    return f"Merge successfully: {a_entity_id} <- {b_entity_id}"

# 將工具綁到 LLM
llm_with_tools = llm.bind_tools([merge])

# -------------------------------------------------
# 3) LLM 判定是否合併（只在 >95% 確信下才呼叫工具）
# -------------------------------------------------
SYSTEM_INSTRUCT = """
You are a deduplication evaluator. Given two entities (A, B) with (entity_id, description),
you must carefully compare both entity_id and description.

Only when you are OVER 95% confident that A and B refer to the SAME real-world entity,
you should INVOKE the tool: merge(a_entity_id, b_entity_id).

If you are NOT over 95% confident (or they should not be merged), do NOT invoke any tool
and simply reply with the single token: NO_MERGE.
"""

EXAMPLE_BLOCK = """
Example
#### Merge ###
A.entity_id = Apple
A.description = A technology company known for designing and manufacturing consumer electronics, software, and services.

B.entity_id = Apple Inc.
B.description = An American multinational technology company headquartered in Cupertino, California.

#### No merge ###
A.entity_id = Apple
A.description = A technology company known for designing and manufacturing consumer electronics, software, and services.

B.entity_id = TSMC
B.description = Taiwan Semiconductor Manufacturing Company, the world's largest semiconductor foundry.
""".strip()

def decide_entity_and_merge(
    a_entity_id: str, a_description: str,
    b_entity_id: str, b_description: str,
) -> Tuple[bool, str]:
    """
    讓 LLM 決定是否合併；若合併則會呼叫 merge() 工具。
    回傳：(是否有合併, 訊息字串/NO_MERGE)
    """
    user_prompt = (
        "Determine whether the two entities are the same. "
        "Only when you are over 95% confident that A and B refer to the SAME entity, "
        "invoke the tool `merge(a_entity_id, b_entity_id)`. "
        "If you are not 95% confident, reply exactly with `NO_MERGE`.\n\n"
        f"A.entity_id = {a_entity_id}\n"
        f"A.description = {a_description}\n\n"
        f"B.entity_id = {b_entity_id}\n"
        f"B.description = {b_description}\n"
    )

    resp = llm_with_tools.invoke(
        [
            {"role": "system", "content": SYSTEM_INSTRUCT},
            {"role": "user",   "content": EXAMPLE_BLOCK},
            {"role": "user",   "content": user_prompt},
        ]
    )

    tool_calls = resp.additional_kwargs.get("tool_calls", [])
    if not tool_calls:
        # 沒有呼叫工具 => 不合併
        return False, "NO_MERGE"

    # 只取第一個工具呼叫（照你的截圖邏輯）
    call = tool_calls[0]
    args = json.loads(call["function"]["arguments"])
    # 執行真正的合併
    result_str = merge.invoke(args)
    return True, result_str

# -------------------------------------------------
# 4) 主流程：兩兩比對、由 LLM 決策、必要時執行合併
# -------------------------------------------------
def main():
    items = fetch_entities()  # List[Tuple[entity_id, description]]
    n = len(items)
    if n == 0:
        print("No entities found.")
        return

    # active[i] = True 代表尚未被「合併成別人」或「處理結束」
    active = [True] * n

    # 外層：逐一把每個節點視為 A
    for i in range(n):
        if not active[i]:
            continue  # 跳過已經被合併或處理過的

        a_entity_id, a_description = items[i]

        # 內層：與後面的節點逐一比對（視為 B）
        for j in range(i + 1, n):
            if not active[j]:
                continue

            b_entity_id, b_description = items[j]
            ok, msg = decide_entity_and_merge(
                a_entity_id, a_description, b_entity_id, b_description
            )
            if ok:
                # B 併到 A，標記 B 已被合併
                active[j] = False
                print(msg)

        # A 已經與所有後面元素比對完成，保持 active[i] = True（表示 A 還存在）

    # 收集仍然活著的（未被合併到別人身上的）items
    remaining = [items[i] for i in range(n) if active[i]]
    print(f"Remaining active entities: {len(remaining)}")

if __name__ == "__main__":
    main()
