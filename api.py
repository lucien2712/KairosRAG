from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from config import initialize_rag
from lightrag import QueryParam
from rewriter import rewriter
from utils import get_current_time
from contextlib import asynccontextmanager
from fastapi_mcp import FastApiMCP
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# 定義 user request for server
class QueryRequest(BaseModel):
    input: str = Field(
        ..., description="User query message to process"
    )
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        default="hybrid",
        description="""
        Specifies the retrieval mode:
        - **local**: Focuses on context-dependent information within a localized scope.
        - **global**: Utilizes global knowledge for retrieval.
        - **hybrid**: Combines local and global retrieval methods.
        - **naive**: Performs a basic search.

        The default mode is **hybrid**. If not necessary, please use **hybrid** mode.
        """,
    )
    conversation_history: list = Field(
        default=[],
        description="""
        A list of previous conversation history, used to maintain context.
        Format: [{"role": "user/assistant", "content": "message"}]

        If not necessary, please use **[]**.
        """,
    )
    history_turns: int = Field(
        default=0,
        description="""
        Specifies the number of conversation history turns to include when processing user queries.

        If not necessary, please use **0**.
        """,
    )

# 定義 server response for user
class QueryResponse(BaseModel):
    result: str

# FastAPI lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    try:
        rag = None
        rag = await initialize_rag()
        print("RAG initialized successfully!")
    except Exception as e:
        print(f"Fail during initialization: {e}")

    yield

    if rag:
        try:
            await rag.finalize_storages()
            print("RAG finalized successfully!")
        except Exception as e:
            print(f"Fail during shutdown: {e}")


# 建立 FastAPI app
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 允許所有來源訪問
    allow_credentials=True,
    allow_methods=["*"],   # 允許所有方法
    allow_headers=["*"],   # 允許所有標頭
)

# FinGAR Query System
async def system(query, mode, conversation_history, history_turns):
    current_date, quarter = get_current_time()
    response = None
    try:
        if rag is None:
            raise RuntimeError("RAG is not initialized.")

        rewritten_query = rewriter(query, current_date, quarter)
        response = rag.query(
            rewritten_query,
            param=QueryParam(
                mode=mode,
                conversation_history=conversation_history,
                history_turns=history_turns,
                max_total_tokens=40960,
                enable_rerank=False,
                user_prompt=[
                    "/no_think.",
                    f"The current date is **{current_date}**, and we are currently in **{quarter}**.",
                ],
                response_type="Single Paragraph",
            ),
        )
    except Exception as e:
        print(f"Fail: {e}")

    return response or "No valid response found."


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    This system utilizes a knowledge graph to structure and organize corporate information,
    making it accessible and contextually relevant for various use cases.

    ## Target Use Cases
    1. **Investor Research**: Empower investors to make informed decisions by providing detailed insights.
    2. **Financial Analysts**: Support in identifying trends, risks, and opportunities.
    3. **Corporate Strategy Teams**: Enhance strategic planning with actionable insights.
    4. **Market Researchers**: Aid in tracking industry trends and benchmarking.
    """
    result = await system(
        request.input,
        request.mode,
        request.conversation_history,
        request.history_turns,
    )
    return QueryResponse(result=result)


name = "FinGAR"
description = """
This system utilizes a knowledge graph to structure and organize corporate information,
"""

# MCP
mcp = FastApiMCP(app, name=name, description=description)
mcp.mount_http()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
