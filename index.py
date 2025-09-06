# index.py
import os
import asyncio
from pathlib import Path
from typing import List, Tuple
from config import initialize_rag


def extract_txt_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def process_files_to_list(folder_path: str) -> Tuple[List[str], List[str], List[str]]:
    texts = []
    file_names = []
    fiscal_years = []

    for file_name in os.listdir(folder_path):
        file_path = Path(folder_path) / file_name
        if file_path.suffix.lower() != ".txt":
            print(f"Skip (not TXT): {file_name}")
            continue
        try:
            text_content = extract_txt_text(file_path)
            if text_content:
                texts.append(text_content)
                file_names.append(file_name)

                # 從檔名抓出 "2025 q3"
                # 假設檔名格式像 "apple 2025 q3.txt"
                parts = file_name.replace(".txt", "").split()
                fiscal = None
                for i in range(len(parts) - 1):
                    if parts[i].isdigit() and parts[i+1].lower().startswith("q"):
                        fiscal = f"Fiscal Year {parts[i]} {parts[i+1].lower()}"
                        break
                fiscal_years.append(fiscal if fiscal else "Unknown")

                print(f"Success: {file_name}")
            else:
                print(f"Empty content: {file_name}")
        except Exception as e:
            print(f"Fail: {file_name}, cause: {e}")

    return texts, file_names, fiscal_years


async def main():
    rag = None
    try:
        # 初始化 RAG
        rag = await initialize_rag()
        print("Initialization success")

        # 讀取 PDF 並建立索引
        texts, file_paths, fiscal_years = process_files_to_list("./inputs")

        if texts:
            rag.insert(texts, file_paths=file_paths, timestamps=fiscal_years)
            rag.agentic_merging(threshold=0.8)
            print("Graph building success!")
        else:
            print("No valid PDF content found, graph not built.")
    except Exception as e:
        print(f"Graph building fail: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
