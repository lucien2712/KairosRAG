# utils.py
from datetime import datetime
import json

def get_current_time():
    now = datetime.now()
    current_month = now.month
    if current_month in [1, 2, 3]:
        quarter = "Q1"
    elif current_month in [4, 5, 6]:
        quarter = "Q2"
    elif current_month in [7, 8, 9]:
        quarter = "Q3"
    else:  # [10, 11, 12]
        quarter = "Q4"

    return now.strftime('%Y-%m-%d'), quarter


def extract_json_from_response(response_content: str):
    try:
        # 尋找 JSON
        json_start = response_content.find("{")
        json_end = response_content.rfind("}") + 1

        # 如果找到 JSON 區塊，提取並解析
        if json_start != -1 and json_end != -1:
            json_str = response_content[json_start:json_end]
            return json.loads(json_str)

        # 如果沒找到 JSON，回傳 None
        return None

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None
