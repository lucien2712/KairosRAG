import os
import textract
import json
from openai import OpenAI

# 請在此處填寫您的 API 金鑰
# The long api_key is constructed from multiple images.
# For security, it has been replaced with a placeholder.
client = OpenAI(
    base_url="http://twsuat.tsmc.com/api/v2beta/openai/",
    api_key="YOUR_API_KEY_HERE"
)

# 請在此處定義您的 Prompt
PROMPT_cross_company = """
Your prompt content goes here. For example:
Based on the following text, generate a list of question-answer pairs in JSON format.
Each object in the list should have a "question" key and an "answer" key.
"""

# 從模型的回應中提取 JSON 格式的數據
def extract_json_from_response(response_content):
    """
    Extracts a JSON list from a string response.
    """
    try:
        # 尋找JSON 起始和結束位置 (包含 [] 的情況)
        json_start = response_content.find("[")
        json_end = response_content.rfind("]") + 1
        
        # 如果找到JSON 區塊
        if json_start != -1 and json_end != 0:
            json_str = response_content[json_start:json_end] # 提取JSON 字串
            return json.loads(json_str) # 將字串轉換成Python 資料結構
        else:
            # 如果未找到JSON，回傳 None
            return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during JSON extraction: {e}")
        return None

# 生成QA 資料集
def generate_qa_dataset(split_texts, prompt):
    """
    Generates a QA dataset by sending text batches to an AI model.
    """
    instruction = prompt
    qa_dataset_list = []
    
    for batch_texts in split_texts:
        batch_text = "\\n\\n".join(batch_texts)
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates QA datasets.",
                    },
                    {
                        "role": "user",
                        "content": f"{instruction}\\n\\n{batch_text}",
                    },
                ]
            )
            
            # 提取 JSON 部分
            qa_dataset_raw = response.choices[0].message.content
            qa_dataset = extract_json_from_response(qa_dataset_raw)
            
            if qa_dataset:
                qa_dataset_list.extend(qa_dataset)
            else:
                print(f"Error: Unable to parse JSON from response. Raw response:\n{qa_dataset_raw}")
        
        except Exception as e:
            print(f"Error generating QA dataset: {e}")
            
    return qa_dataset_list

def process_files_to_list(folder_path, mode):
    """
    Reads all files from a folder, extracts their text content, and returns a list of texts.
    """
    texts = []
    try:
        # 讀取所有檔案名稱
        all_files = os.listdir(folder_path)
        
        # 根據模式進行排序
        if mode == "cross_company":
            # 檔案名稱格式為 "amd 2023 q4.pdf", 根據年度, 季度排序
            sorted_files = sorted(all_files, key=lambda file_name: (
                int(file_name.split()[1]) if len(file_name.split()) > 2 else float("inf"), # 年份
                int(file_name.split()[2][1]) if len(file_name.split()) > 2 else float("inf") # 季度
            ))
        else:
            # 默認按檔案名稱字母順序排序 (cross year)
            sorted_files = sorted(all_files)
            
        for file_name in sorted_files:
            file_path = os.path.join(folder_path, file_name)
            try:
                # 使用 textract 提取檔案內容
                text_content = textract.process(file_path)
                texts.append(text_content.decode("utf-8"))
                print(f"Success: {file_name}")
            except Exception as e:
                print(f"Fail: {file_name}, cause: {e}")
                
    except Exception as e:
        print(f"Error accessing folder {folder_path}, cause: {e}")
        
    return texts

if __name__ == "__main__":
    folder_path = "./transcript"
    batch_size = 5

    # 步驟1: 處理檔案, 提取文本
    # Note: The code in the image uses "company" as the mode here.
    # The sorting logic for "cross_company" is defined above if you wish to use it.
    texts = process_files_to_list(folder_path, "company")

    # 步驟2: 分割文本到批次 (避免 token 爆掉)
    split_texts = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    # 步驟3: 生成QA 資料集
    qa_dataset = generate_qa_dataset(split_texts, PROMPT_cross_company)

    with open("qa_dataset_cross_company.json", "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

    print("Dataset building success!")