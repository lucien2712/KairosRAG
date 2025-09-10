from config import initialize_rag
from lightrag import QueryParam
from rewriter import rewriter


async def main():
    try:
        # 初始化 RAG object
        rag = None
        rag = await initialize_rag()
        print("Initialization success!!")

        query = """
        Please summarize the latest quarterly earnings report of Apple Inc.
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
                top_k=20,
                chunk_top_k= 20,
                max_total_tokens=120000,
                max_entity_tokens=30000,
                max_relation_tokens=30000,
                max_hop=2,
                max_neighbors= 20,
                multi_hop_relevance_threshold=0.3,
                enable_rerank=False,
                top_ppr_nodes=5,       
                top_fastrp_nodes=5,     
                user_prompt="""
                    Let's think step by step to answer the question
                    ## Step 1: Question Analysis
                    Break down what exactly is being asked and identify key components.

                    ## Step 2: Evidence Review
                    Carefully examine all the provided context and identify relevant information.

                    ## Step 3: Logical Reasoning
                    Connect the evidence pieces and apply logical reasoning to draw conclusions.

                    ## Step 4: Answer Synthesis
                    Provide a comprehensive answer based on the reasoning above.

                    ===
                    
                    You have to answer the question following the format below:
                    ## <Title of the report>
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
"""
You have to answer the question following the format below:
## <Title of the report>
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
"""