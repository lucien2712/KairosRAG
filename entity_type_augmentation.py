import config
import os
import textract
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

ENTITY_TYPE_PATH = os.path.join(os.environ["WORKING_DIR"], "entity_type.json")
DATASET_STATUS_PATH = os.path.join(
    os.environ["WORKING_DIR"], "entity_check_status.json"
)

# 預設 Entity Types
DEFAULT_ENTITY_TYPES = [
    {
        "entity_type": "organization",
        "explanation": "An entity representing organizations, companies, or institutions.",
    },
    {
        "entity_type": "person",
        "explanation": "An entity representing individual persons.",
    },
    {
        "entity_type": "geo",
        "explanation": "An entity representing geographical locations.",
    },
    {
        "entity_type": "event",
        "explanation": "An entity representing events or activities.",
    },
    {
        "entity_type": "category",
        "explanation": "An entity representing general categories or classifications.",
    },
    {
        "entity_type": "temporal_range",
        "explanation": "An entity representing time periods, including specific dates, months, quarters, or years (e.g., '2024 Q1', '2024 July').",
    },
]


# 讀取 JSON 檔案
def load_json(file_path, default_data=None):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default_data if default_data else []


# 儲存 JSON 檔案
def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 掃描並處理檔案
def process_files(folder_path):
    processed_files = load_json(DATASET_STATUS_PATH, {})
    file_text_pairs = []

    all_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    for file_name in all_files:
        if file_name in processed_files:
            print(f"Skipped: {file_name} (already processed)")
            continue

        file_path = os.path.join(folder_path, file_name)
        try:
            text_content = textract.process(file_path)
            file_text_pairs.append((file_name, text_content.decode("utf-8")))
            print(f"Success: {file_name}")
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

    return file_text_pairs


# 從 LLM 回應提取 JSON
def extract_json_from_response(response_content):
    try:
        json_start = (
            response_content.find("[")
            if "[" in response_content
            else response_content.find("{")
        )
        json_end = (
            response_content.rfind("]") + 1
            if "]" in response_content
            else response_content.rfind("}") + 1
        )
        if json_start != -1 and json_end != -1:
            json_str = response_content[json_start:json_end]
            print(f"Extracted JSON String:\n{json_str}")
            return json.loads(json_str)
        print("Warning: No valid JSON found in the response.")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return None


# 使用 LLM 決定是否新增 Entity Types
def process_file_with_llm(file_content, current_entity_types):
    try:
        current_entity_types_json = json.dumps(
            [et["entity_type"] for et in current_entity_types]
        )
        response = client.chat.completions.create(
            model="qwen3-235b-fp8",
            messages=[
                {
                    "role": "system",
                    "content": """
                                You are an expert in Named Entity Recognition (NER). Your goal is to analyze the connections and relations between existing entity types and document content to provide meaningful refinements or additions.

                                ## Task Requirements:
                                - Avoid suggesting "other" or "unknown" types
                                - Do not suggest duplicates or overlapping entity types  
                                - Prioritize quality over quantity
                                - Provide concise yet clear explanations
                                - Respond in strict JSON array format only

                                ## Response Format:
                                [
                                {
                                    "entity_type": "<entity_type_name>",
                                    "explanation": "<detailed_explanation>"
                                }
                                ]

                                ## Example:
                                ### Current Entity Types:
                                [
                                    {
                                        "entity_type": "person",
                                        "explanation": "An entity representing individual persons."
                                    },
                                    {
                                        "entity_type": "temporal_range",
                                        "explanation": "An entity representing time periods, including specific dates, months, quarters, or years (e.g., '2024 Q1', '2024 July')."
                                    }
                                ]
                                ## Document Content:
                                Apple Inc. was founded in 1976. After that, it became one of the most successful companies in the world.

                                
                                ### Suggested New Entity Types:
                                [
                                    {
                                        "entity_type": "organization",
                                        "explanation": "An entity representing organizations, companies, or institutions."
                                    }
                                ]
                                """,
                },
                {
                    "role": "user",
                    "content": f"""
                                ## Current Entity Types:
                                {current_entity_types_json}

                                ## Task:
                                Based on the following document content, analyze and suggest new entity types with explanations if needed.

                                ## Document Content:
                                {file_content}

                                Please carefully analyze the entities that appear in the document and suggest appropriate new entity types (if any are needed).
                                """,
                },
            ],
        )
        llm_response = response.choices[0].message.content
        suggested_types = extract_json_from_response(llm_response)
        return suggested_types if suggested_types else []
    except Exception as e:
        print(f"Error processing file with LLM: {e}")
        return []


# 使用 LLM 去除重複或相似的 Entity Types
def refine_entity_types(entity_types):
    try:
        response = client.chat.completions.create(
            model="qwen3-235b-fp8",
            messages=[
                {
                    "role": "system",
                    "content": """
                        You are an advanced linguistic assistant with expertise in Named Entity Recognition (NER).

                        ## Task:
                        Refine a list of entity types by removing duplicates or semantically overlapping types.

                        ## Requirements:
                        - Ensure each type is distinct, meaningful, and concise
                        - Remove redundant or overlapping entity types
                        - Keep the most comprehensive and well-defined entity type when merging similar ones
                        - Maintain clear and accurate explanations
                        - Return the refined list in strict JSON array format only

                        ## Response Format:
                        [
                        {
                            "entity_type": "<entity_type_name>",
                            "explanation": "<clear_explanation>"
                        }
                        ]

                        ## Example:
                        ### Entity Types List to Refine:
                        [
                            {
                                "entity_type": "Company",
                                "explanation": "A company is a legal entity formed by a group of individuals or entities to engage in business activities."
                            },
                            {
                                "entity_type": "Organization", 
                                "explanation": "An organization is a group of individuals or entities that work together to achieve a common goal."
                            }
                        ]

                        ### Refined List:
                        [
                            {
                                "entity_type": "Organization",
                                "explanation": "An organization is a group of individuals or entities that work together to achieve a common goal."
                            }
                        ]
                        """,
                },
                {
                    "role": "user",
                    "content": f"""
                        ## Entity Types List to Refine:
                        {json.dumps(entity_types, ensure_ascii=False, indent=2)}

                        ## Task:
                        Please refine this list by removing duplicates or semantically similar entity types. Keep the most appropriate and comprehensive entity type when consolidating similar ones.
                    """,
                },
            ],
        )
        response_content = response.choices[0].message.content
        refined_entity_types = extract_json_from_response(response_content)
        return refined_entity_types if refined_entity_types else entity_types
    except Exception as e:
        print(f"Error refining entity types with LLM: {e}")
        return entity_types


if __name__ == "__main__":
    folder_path = "./transcript"

    # Step 1: 讀取尚未處理的檔案
    file_text_pairs = process_files(folder_path)

    # Step 2: 讀取現有 Entity Types
    current_entity_types = load_json(ENTITY_TYPE_PATH, DEFAULT_ENTITY_TYPES)

    # Step 3: 逐檔處理，生成新 Entity Types
    all_new_entity_types = []
    processed_files = load_json(DATASET_STATUS_PATH, {})
    if not isinstance(processed_files, dict):
        processed_files = {}

    for file_name, text_content in file_text_pairs:
        new_entity_types = process_file_with_llm(text_content, current_entity_types)
        all_new_entity_types.extend(new_entity_types)
        processed_files[file_name] = True

    # Step 4: 合併 Entity Types
    combined_entity_types = current_entity_types + all_new_entity_types

    # Step 5: 使用 LLM 清理重複
    refined_entity_types = refine_entity_types(combined_entity_types)

    # Step 6: 儲存更新後的 Entity Types 與狀態
    save_json(refined_entity_types, ENTITY_TYPE_PATH)
    print(f"Refined Entity Types: {refined_entity_types}")

    save_json(processed_files, DATASET_STATUS_PATH)
    print("Entity types refined and files marked as processed!")
