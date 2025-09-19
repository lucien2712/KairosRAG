# experiment
import os
import json
import asyncio
from config import initialize_rag
from lightrag import QueryParam
from lightrag.small_tool import get_current_time

file_path = "./rag_storage/kv_store_llm_response_cache.json"
if os.path.exists(file_path):
    os.remove(file_path)


async def generate_answers_from_questions(json_file_path, mode="hybrid"):
    try:
        # 初始化 RAG object
        rag = None
        rag = await initialize_rag()
        print("Initialization sucess!!!")

        with open(json_file_path, "r", encoding="utf-8") as f:
            qa_dataset = json.load(f)

        # 遍歷每一題，產生答案
        for qa in qa_dataset:
            question = qa["question"]
            print(f"Processing question: {question}")

            # 使用 RAG 進行查詢
            try:
                current_date, current_quarter = get_current_time()
                response = rag.query(
                    f"{question}, The current date is **{current_date}**, and we are currently in **{current_quarter}**",
                    param=QueryParam(
                        mode=mode,
                        history_turns=0,
                        max_total_tokens=40960,
                        enable_rerank=False,
                        user_prompt="/no_think",
                        response_type="Single Paragraph",
                    ),
                )

                if mode == "hybrid":
                    qa["Answer_1"] = response
                else:
                    qa["Answer_2"] = response

            except Exception as e:
                qa["exception"] = (
                    f"Failed to process question: {question}. Cause: {e}"
                )

            print("=" * 50)
            print(response)
            print("=" * 50)

        # 將更新後的 JSON 資料集存回原檔案
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

        print("All questions processed and answers saved successfully!")

    except Exception as e:
        print(f"Fail: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    JSON_FILE_PATH = "qa_dataset_cross_company.json"

    asyncio.run(generate_answers_from_questions(JSON_FILE_PATH, mode="hybrid"))
    asyncio.run(generate_answers_from_questions(JSON_FILE_PATH, mode="naive"))

    print("All questions processed and answers saved successfully!")
