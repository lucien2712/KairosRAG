
from openai import OpenAI
from lightrag.small_tool import extract_json_from_response, get_current_time
import os
import config
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
)

# 使用 LLM 改寫 Query（將時間相關模糊詞換成精確時間）
def rewriter(user_query: str) -> str:
    # 獲取當前時間和季度
    current_date, current_quarter = get_current_time()
    
    prompt_system = """
You are an expert in understanding vague queries and rewriting them with precise temporal references.
Rewrite vague temporal terms such as "recent", "latest", "this quarter", "last quarter", "this year", etc., and replace them with precise time ranges or specific dates based on the provided context.

Provide the output strictly in JSON format with the following structure:
{
  "rewritten_query": "<rewritten query>"
}

Example:
User Query: "What are the recent trends? The current date is **2023-10-17**, and we are currently in **Q4**"
Expected Output:
{
  "rewritten_query": "What are the trends from Q2 2023 to Q3 2023?"
}
"""

    prompt_user = f"""
                    User Query: '{user_query}'
                    The current date is **{current_date}**, and we are currently in **{current_quarter}**.
                    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()
    json_output = extract_json_from_response(content)

    if json_output is None:
        raise ValueError(f"Failed to extract valid JSON from content: {content}")

    return json_output.get("rewritten_query", "")


if __name__ == "__main__":
    user_query = (
        "With the recent stagnation in the advancement of AI features in Apple products, "
        "and the lack of significant differentiation from Android competitors, what strategies "
        "or countermeasures has Apple adopted internally to maintain competitive positioning "
        "and technological leadership?"
    )

    try:
        result = rewriter(user_query)
        print(f"Rewritten Query: {result}")
    except ValueError as e:
        print(f"Error: {e}")
