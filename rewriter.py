
from openai import OpenAI
from utils import extract_json_from_response

client = OpenAI(
    api_key="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJmaWQiOiI1NTE10TNkMSOWNtC3LTQxMGYtYWE3Zi1kMWZhM2U4MDA2YjkiLCJ2ZXIiOiI5IiwizXhwIjoxNzY1MjY5NzIyLCJpYXQiOjE3NDk3MTc3MjIsImFjY3QiOiJMSU5ZQ0FQIn0.HQAHJP4iBm_m8IR5NTAArtN-e-DWTQcQ-PvjgbxY5gH-dvA_CgIBHoVIvqN9WCgDqfhrYie9kOjTaBhuvUWd9A",
    base_url="http://aikm-qwen3-235b-fp8-1.mlp-dev.mldp1.dev.tsmc.com/v1/",
)

# 使用 LLM 改寫 Query（將時間相關模糊詞換成精確時間）
def rewriter(user_query: str) -> str:
    prompt_system = """
/no_think
You are an expert in understanding vague queries and rewriting them with precise temporal references.
Rewrite vague temporal terms such as "recent", "latest", "this quarter", "last quarter", "this year", etc., and replace them with precise time ranges or specific dates based on the provided context.

Provide the output strictly in JSON format with the following structure:
{{
  "rewritten_query": "<rewritten query>"
}}

Example:
User Query: "What are the recent trends? The current date is **2023-10-17**, and we are currently in **Q4**"
Expected Output:
{{
  "rewritten_query": "What are the trends from Q2 2023 to Q3 2023?"
}}
"""

    prompt_user = f"""
                    User Query: '{user_query}'
                    """

    response = client.chat.completions.create(
        model="qwen3-235b-fp8",
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
