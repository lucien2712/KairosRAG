# index.py
import os
import asyncio
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader

from config import initialize_rag


def extract_pdf_text(path: Path) -> str:
    """使用 pypdf 抽取 PDF 文字"""
    reader = PdfReader(str(path))
    texts: List[str] = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts).strip()


def process_files_to_list(folder_path: str) -> Tuple[List[str], List[str]]:
    """讀取資料夾下所有 PDF 檔案，回傳 (文字內容列表, 檔名列表)"""
    texts = []
    file_names = []

    for file_name in os.listdir(folder_path):
        file_path = Path(folder_path) / file_name
        if file_path.suffix.lower() != ".pdf":
            print(f"Skip (not PDF): {file_name}")
            continue
        try:
            text_content = extract_pdf_text(file_path)
            if text_content:
                texts.append(text_content)
                file_names.append(file_name)
                print(f"Success: {file_name}")
            else:
                print(f"Empty content: {file_name}")
        except Exception as e:
            print(f"Fail: {file_name}, cause: {e}")

    return texts, file_names


async def main():
    rag = None
    try:
        # 初始化 RAG
        rag = await initialize_rag()
        print("Initialization success")

        # 讀取 PDF 並建立索引
        texts, file_paths = process_files_to_list("./input")

        if texts:
            rag.insert(texts, file_paths=file_paths)
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
