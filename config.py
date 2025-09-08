import os
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc

import nest_asyncio
nest_asyncio.apply()

os.environ["OPENAI_API_KEY"] = "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"

setup_logger("lightrag", level="INFO")

# vector 等資料存放目錄
WORKING_DIR = "./rag_storage"
os.environ["WORKING_DIR"] = WORKING_DIR

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        enable_node_embedding=True,
        enable_llm_cache= False
    )
    
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline
    return rag