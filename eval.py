#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-based Q&A evaluation script.

What it does
------------
1) evaluate_relevancy(question, model_output, reference_answer)
   -> returns (score 1..10, explanation)

2) evaluate_with_metrics(question, reference_answer, answer_1, answer_2)
   -> returns JSON with four dimensions for Answer_1 & Answer_2:
      Comprehensiveness, Diversity, Empowerment, Overall

3) evaluate_dataset(path)
   -> reads a JSON list with fields:
      { "question": ..., "answer": ..., "Answer_1": ..., "Answer_2": ... }
      and adds the evaluation results back into the items, then writes
      the updated list to the same file.

Environment variables
---------------------
OPENAI_BASE_URL  (default: http://localhost:8000/v1/)
OPENAI_API_KEY   (default: dummy)
OPENAI_MODEL     (default: gpt-4o)

Notes
-----
- If you have utils.extract_json_from_response, it will be used.
  Otherwise a fallback extractor is applied.
"""

import os
import json
from typing import Tuple, Dict, Any, List
from openai import OpenAI

try:
    from lightrag.small_tool import extract_json_from_response  # optional helper
except Exception:
    def extract_json_from_response(text: str):
        """Fallback: extract the first JSON object from a string."""
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError("No JSON object found in response")
        return json.loads(m.group(0))

# ---- OpenAI client ----
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1/"),
    api_key=os.getenv("OPENAI_API_KEY", "dummy"),
)
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


# -----------------------------
# Relevancy (single-answer) eval
# -----------------------------
def evaluate_relevancy(input_question: str, model_output: str, reference_answer: str, model: str = DEFAULT_MODEL) -> Tuple[int, str]:
    """Return (score, explanation) for relevancy vs reference answer."""
    relevancy_prompt = f"""
    ---Role---
    You are an evaluator tasked with assessing the quality and relevancy of a model-generated answer to a given question.
    Your evaluation should focus on both the precision and completeness of the answer, ensuring fairness across diverse response styles.

    ---Goal---
    Evaluate the Model Output based on the following criteria:

    1. **Direct Relevance (70%)**: Assess how accurately and completely the Model Output addresses the core question. This criterion is the primary factor in determining the quality of the answer. A concise and accurate response should be rated highly, even if it lacks supporting information.
    2. **Supporting Information (30%)**: Analyze whether additional context or supporting data enhances the response by providing clarity, depth, or reinforcement. Note that while supporting information can add value, it should not penalize answers that are already accurate and sufficient without extra context.
    3. **Explanation**: Provide a detailed justification for your scores, explaining how the Model Output meets (or fails to meet) the relevance and supporting information criteria. Clearly identify strengths and any areas for improvement.

    You must assign a score between 1 and 10 (1 being the worst and 10 being the best).

    Question: {input_question}

    === Reference Answer: ===
    {reference_answer}

    === Model Output: ===
    {model_output}

    Provide your evaluation in the structured JSON format below:
    ```json
    {{
        "Relevancy": {{
            "Score": score,
            "Explanation": explanation_for_the_score
        }}
    }}
    ```
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an expert evaluator of text relevancy."},
            {"role": "user", "content": relevancy_prompt},
        ],
    )
    result = response.choices[0].message.content
    data = extract_json_from_response(result)
    score = int(data["Relevancy"]["Score"])  # 1..10
    explanation = data["Relevancy"]["Explanation"]
    return score, explanation


# --------------------------------------------
# Comparison eval (two answers, four dimensions)
# --------------------------------------------
def evaluate_with_metrics(question: str, reference_answer: str, answer_1: str, answer_2: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Return per-dimension scores & explanations for Answer_1 and Answer_2."""
    comparison_prompt = f"""
    ---Role---
    You are a sophisticated evaluator tasked with comparing answers based on the following four dimensions: **Comprehensiveness**, **Diversity**, **Empowerment**, and **Overall**.

    ---Goal---
    You will evaluate two answers to the same question based on four criteria:
    **Comprehensiveness**, **Diversity**, **Empowerment**, and **Overall**.

    1. **Comprehensiveness**: How thoroughly does the answer address all aspects and details of the question?
    2. **Diversity**: How varied and rich is the answer in offering different perspectives and insights related to the question?
    3. **Empowerment**: How effectively does the answer enable the reader to understand the topic and make informed judgments?
    4. **Overall**: This dimension assesses the cumulative performance across the above three criteria.

    Question: {question}

    === Reference Answer: ===
    {reference_answer}

    === Answer_1: ===
    {answer_1}

    === Answer_2: ===
    {answer_2}

    Evaluate Answer_1 and Answer_2 across the four dimensions. For each dimension, assign a score between 1 and 10 (1 worst, 10 best) and provide a short explanation for each answer.

    Respond in JSON format exactly as below:
    ```json
    {{
        "Comprehensiveness": {{
            "Answer_1": score,
            "Answer_2": score,
            "Answer_1_explanation": "Explanation for this score",
            "Answer_2_explanation": "Explanation for this score"
        }},
        "Diversity": {{
            "Answer_1": score,
            "Answer_2": score,
            "Answer_1_explanation": "Explanation for this score",
            "Answer_2_explanation": "Explanation for this score"
        }},
        "Empowerment": {{
            "Answer_1": score,
            "Answer_2": score,
            "Answer_1_explanation": "Explanation for this score",
            "Answer_2_explanation": "Explanation for this score"
        }},
        "Overall": {{
            "Answer_1": score,
            "Answer_2": score,
            "Answer_1_explanation": "Explanation for this score",
            "Answer_2_explanation": "Explanation for this score"
        }}
    }}
    ```
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an expert evaluator of text answers."},
            {"role": "user", "content": comparison_prompt},
        ],
    )
    result = response.choices[0].message.content
    data = extract_json_from_response(result)
    return data


# ---------
# Integrate
# ---------
def evaluate_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    for item in dataset:
        question = item.get("question", "")
        reference_answer = item.get("answer", "")
        Answer_1 = item.get("Answer_1", "")
        Answer_2 = item.get("Answer_2", "")

        # Step 1: relevancy for each answer
        item["Answer_1_Relevancy"], item["Answer_1_Relevancy_explanation"] = evaluate_relevancy(
            input_question=question, model_output=Answer_1, reference_answer=reference_answer
        )
        item["Answer_2_Relevancy"], item["Answer_2_Relevancy_explanation"] = evaluate_relevancy(
            input_question=question, model_output=Answer_2, reference_answer=reference_answer
        )

        # Step 2: four-dimension comparison
        openai_result = evaluate_with_metrics(
            question=question,
            reference_answer=reference_answer,
            answer_1=Answer_1,
            answer_2=Answer_2,
        )

        # Fill metrics for Answer_1
        item["Answer_1_Comprehensiveness"] = openai_result.get("Comprehensiveness", {}).get("Answer_1")
        item["Answer_1_Comprehensiveness_explanation"] = openai_result.get("Comprehensiveness", {}).get("Answer_1_explanation")
        item["Answer_1_Diversity"] = openai_result.get("Diversity", {}).get("Answer_1")
        item["Answer_1_Diversity_explanation"] = openai_result.get("Diversity", {}).get("Answer_1_explanation")
        item["Answer_1_Empowerment"] = openai_result.get("Empowerment", {}).get("Answer_1")
        item["Answer_1_Empowerment_explanation"] = openai_result.get("Empowerment", {}).get("Answer_1_explanation")
        item["Answer_1_Overall"] = openai_result.get("Overall", {}).get("Answer_1")
        item["Answer_1_Overall_explanation"] = openai_result.get("Overall", {}).get("Answer_1_explanation")

        # Fill metrics for Answer_2
        item["Answer_2_Comprehensiveness"] = openai_result.get("Comprehensiveness", {}).get("Answer_2")
        item["Answer_2_Comprehensiveness_explanation"] = openai_result.get("Comprehensiveness", {}).get("Answer_2_explanation")
        item["Answer_2_Diversity"] = openai_result.get("Diversity", {}).get("Answer_2")
        item["Answer_2_Diversity_explanation"] = openai_result.get("Diversity", {}).get("Answer_2_explanation")
        item["Answer_2_Empowerment"] = openai_result.get("Empowerment", {}).get("Answer_2")
        item["Answer_2_Empowerment_explanation"] = openai_result.get("Empowerment", {}).get("Answer_2_explanation")
        item["Answer_2_Overall"] = openai_result.get("Overall", {}).get("Answer_2")
        item["Answer_2_Overall_explanation"] = openai_result.get("Overall", {}).get("Answer_2_explanation")

        # Optional: print each item
        print("============================")
        print(item)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"Evaluation completed! Results saved to {path}")


if __name__ == "__main__":
    path = "qa_dataset.json"
    evaluate_dataset(path)
