# query
from config import initialize_rag
from lightrag import QueryParam
from rewriter import rewriter
import os


async def main():
    try:
        # 初始化 RAG object
        rag = None
        rag = await initialize_rag()
        print("Initialization success!!")

        query = """
        Based on [Company]: Apple, [Target Calendar Year]: 2025 Q2, 
        [Theme]: Tariffs and Supply Chain Diversification, analyze the latest developments from an analyst’s perspective. 
        Summarize the content with a focus on cross-quarter, trend forecasting, and potential impacts on future revenue and financial reports. 
        Include the timing of key events and contrast the latest data with previous periods for enhanced analytical insights. 
        Additionally, identify any potential underlying issues that might affect TSMC.
        """

        # 將時間相關模糊問題進行 rewrite
        rewritten_query = rewriter(query)
        print("Rewrite query: ", query)

        response = rag.query(
            rewritten_query,
            param=QueryParam(
                mode="hybrid",
                # conversation_history
                history_turns=0,
                chunk_top_k= 20,
                max_total_tokens=120000,
                max_hop=2,
                max_neighbors= 30,
                enable_rerank=False,
                user_prompt="""
                    You have to answer the question following the format below:
                    ## Title
                    ### Overview
                    <description: Provide a high-level summary, including scope, purpose, and context.>

                    ### Key Themes
                    <description: Extract the main recurring themes, announcements, priorities, challenges, and opportunities.>

                    ### Comparative Insights
                    <description: Highlight similarities, differences, shifts in tone, and evolving trends.>

                    ### Actionable Insights
                    <description: Summarize practical implications or recommendations (e.g., for strategy, investment, or risk management).>

                    ### Reference
                    <description: List ALL sources used>
                    """,
                response_type="Single Paragraph",
                only_need_context=False,
            ),
        )

        print("===============================")
        print(response)
        print("===============================")

    except Exception as e:
        print(f"Fail: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
