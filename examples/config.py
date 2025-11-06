import os
from lightrag.utils import setup_logger

import nest_asyncio
nest_asyncio.apply()

os.environ["OPENAI_API_KEY"] = "Your-OpenAI-API-Key-Here"

setup_logger("lightrag", level="INFO")

# vector 等資料存放目錄
WORKING_DIR = "./rag_storage"
os.environ["WORKING_DIR"] = WORKING_DIR

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)



