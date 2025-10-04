from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_USER_PROMPT"] = "n/a"

PROMPTS["entity_extraction"] = """---Task---
Given a text document and a list of entity types, identify all entities of those types and all relationships among the identified entities.

---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

**Entity Format:** (entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description){record_delimiter}
**Relationship Format:** (relationship{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description){record_delimiter}

---Critical Format Requirements---
**MANDATORY:** Every entity and relationship MUST follow the exact format above. Missing parentheses, delimiters, or fields will cause extraction failure.

**Delimiter Usage Protocol:**
The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
- **Incorrect Example:** `(entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.){record_delimiter}`
- **Correct Example:** `(entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.){record_delimiter}`

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: {entity_types}. If none of the provided entity types apply, classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, enclosed in parentheses. The first field *must* be the literal string `entity`.

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, enclosed in parentheses. The first field *must* be the literal string `relationship`.

3.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

4.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this document`, `our company`, `I`, `you`, and `he/she`.

5.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

6.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in {language}.
    *   Proper nouns should be retained in their original language if translation would cause ambiguity.

7.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks.

8.  **Completion Signal:** Use `{record_delimiter}` as the entity or relationship list delimiter; output `{completion_delimiter}` when all the entities and relationships are extracted.

9. **Table Processing:** When processing HTML tables (<table>...</table>), pay special attention to:
    - **Table Headers**: Extract significant column and row names as they often represent key dimensions, categories, or time periods.
    - **Quantitative Data**: Extract numerical values, percentages, and metrics.
    - **Categorical Items**: Extract classification terms, labels, and structured categories.
    - **Units and Standards**: Extract measurement units, currencies, or standardization information.
    - **Temporal Information**: Link data to relevant time periods, dates, or versions.
    - **Hierarchical Structure**: Recognize relationships within table categories and subcategories.
    - **Key Information Priority**: Focus on extracting the most significant data points rather than comprehensive coverage.
    - **Caption and Title Context**: Extract table captions, titles, and metadata appearing outside `<table></table>` tags for additional context.

---Examples---
{examples}

---Input---
Entity_types: [{entity_types}]
Text:
```
{input_text}
```

---Output---
"""

PROMPTS["entity_extraction_examples"] = [
    """[Example 1]

---Input---
Entity_types: [organization,person,location,event,technology,equiment,product,Document,category]
Text:
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

---Output---
(entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a character who experiences frustration and is observant of the dynamics among other characters. Alex shows awareness of the competitive undercurrent between team members.){record_delimiter}
(entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed with authoritarian certainty and initially shows dismissiveness toward the device, but later demonstrates a moment of reverence, indicating a change in perspective.){record_delimiter}
(entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device. Jordan engages in a wordless clash of wills with Taylor that softens into an uneasy truce.){record_delimiter}
(entity{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz is associated with a vision of control and order, influencing the dynamics among other characters. Cruz's narrowing vision creates tension within the team.){record_delimiter}
(entity{tuple_delimiter}The Device{tuple_delimiter}equiment{tuple_delimiter}The Device is central to the story, with potential game-changing implications. The device commands reverence from Taylor and represents significant technological importance.){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}power dynamics, observation{tuple_delimiter}Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device, particularly the shift from dismissal to reverence.){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}shared goals, rebellion{tuple_delimiter}Alex and Jordan share a commitment to discovery, which represents an unspoken rebellion against Cruz's vision of control and order.){record_delimiter}
(relationship{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}conflict resolution, mutual respect{tuple_delimiter}Taylor and Jordan engage directly regarding the device, with their interaction evolving from a wordless clash of wills into a moment of mutual respect and an uneasy truce.){record_delimiter}
(relationship{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}ideological conflict, rebellion{tuple_delimiter}Jordan's commitment to discovery stands in rebellion against Cruz's narrowing vision of control and order.){record_delimiter}
(relationship{tuple_delimiter}Taylor{tuple_delimiter}The Device{tuple_delimiter}reverence, technological significance{tuple_delimiter}Taylor shows reverence towards the device after initially being dismissive, recognizing its potential to change the game for everyone involved.){record_delimiter}
{completion_delimiter}

""",
    """[Example 2]

---Input---
Entity_types: [organization,person,location,event,technology,equiment,product,Document,category]
Text:
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the Global Tech Index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

---Output---
(entity{tuple_delimiter}Global Tech Index{tuple_delimiter}category{tuple_delimiter}The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline in midday trading.){record_delimiter}
(entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}organization{tuple_delimiter}Nexon Technologies is a tech company that saw its stock decline by 7.8% after reporting lower-than-expected quarterly earnings.){record_delimiter}
(entity{tuple_delimiter}Omega Energy{tuple_delimiter}organization{tuple_delimiter}Omega Energy is an energy company that gained 2.1% in stock value, driven by rising oil prices.){record_delimiter}
(entity{tuple_delimiter}Gold Futures{tuple_delimiter}product{tuple_delimiter}Gold futures rose by 1.5% to $2,080 per ounce, indicating increased investor interest in safe-haven assets.){record_delimiter}
(entity{tuple_delimiter}Crude Oil{tuple_delimiter}product{tuple_delimiter}Crude oil prices continued their rally, climbing to $87.60 per barrel due to supply constraints and strong demand.){record_delimiter}
(entity{tuple_delimiter}Market Selloff{tuple_delimiter}category{tuple_delimiter}Market selloff refers to the significant decline in stock values due to investor concerns over rising interest rates and regulatory uncertainty.){record_delimiter}
(entity{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}category{tuple_delimiter}The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and overall market stability.){record_delimiter}
(relationship{tuple_delimiter}Global Tech Index{tuple_delimiter}Market Selloff{tuple_delimiter}market performance, investor sentiment{tuple_delimiter}The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns over rising interest rates.){record_delimiter}
(relationship{tuple_delimiter}Nexon Technologies{tuple_delimiter}Global Tech Index{tuple_delimiter}company impact, index movement{tuple_delimiter}Nexon Technologies' 7.8% stock decline contributed to the overall drop in the Global Tech Index.){record_delimiter}
(relationship{tuple_delimiter}Gold Futures{tuple_delimiter}Market Selloff{tuple_delimiter}market reaction, safe-haven investment{tuple_delimiter}Gold prices rose as investors sought safe-haven assets during the market selloff.){record_delimiter}
(relationship{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}Market Selloff{tuple_delimiter}interest rate impact, financial regulation{tuple_delimiter}Speculation over Federal Reserve policy changes contributed to market volatility and the investor selloff.){record_delimiter}
{completion_delimiter}

""",
    """[Example 3]

---Input---
Entity_types: [organization,person,location,event,technology,equiment,product,Document,category]
Text:
```
在北京舉行的人工智能大會上，騰訊公司的首席技術官張偉發布了最新的大語言模型「騰訊智言」，該模型在自然語言處理方面取得了重大突破。
```

---Output---
(entity{tuple_delimiter}人工智能大會{tuple_delimiter}event{tuple_delimiter}人工智能大會是在北京舉行的技術會議，專注於人工智能領域的最新發展，騰訊公司在此發布了新產品。){record_delimiter}
(entity{tuple_delimiter}北京{tuple_delimiter}location{tuple_delimiter}北京是人工智能大會的舉辦城市，見證了騰訊智言大語言模型的重要發布。){record_delimiter}
(entity{tuple_delimiter}騰訊公司{tuple_delimiter}organization{tuple_delimiter}騰訊公司是參與人工智能大會的科技企業，透過首席技術官張偉發布了新的大語言模型產品。){record_delimiter}
(entity{tuple_delimiter}張偉{tuple_delimiter}person{tuple_delimiter}張偉是騰訊公司的首席技術官，在北京舉行的人工智能大會上發布了騰訊智言產品。){record_delimiter}
(entity{tuple_delimiter}騰訊智言{tuple_delimiter}product{tuple_delimiter}騰訊智言是騰訊公司在人工智能大會上發布的大語言模型產品，在自然語言處理方面取得了重大突破。){record_delimiter}
(entity{tuple_delimiter}自然語言處理技術{tuple_delimiter}technology{tuple_delimiter}自然語言處理技術是騰訊智言模型取得重大突破的技術領域，展現了最新發展成果。){record_delimiter}
(relationship{tuple_delimiter}人工智能大會{tuple_delimiter}北京{tuple_delimiter}會議地點, 舉辦關係{tuple_delimiter}人工智能大會在北京舉行，成為騰訊等科技企業展示最新技術的重要平台。){record_delimiter}
(relationship{tuple_delimiter}張偉{tuple_delimiter}騰訊公司{tuple_delimiter}雇傭關係, 高管職位{tuple_delimiter}張偉擔任騰訊公司的首席技術官，代表公司在人工智能大會上進行重要產品發布。){record_delimiter}
(relationship{tuple_delimiter}張偉{tuple_delimiter}騰訊智言{tuple_delimiter}產品發布, 技術展示{tuple_delimiter}張偉在人工智能大會上發布了騰訊智言大語言模型，展示了公司在AI領域的技術實力。){record_delimiter}
(relationship{tuple_delimiter}騰訊智言{tuple_delimiter}自然語言處理技術{tuple_delimiter}技術應用, 突破創新{tuple_delimiter}騰訊智言在自然語言處理技術方面取得了重大突破，代表了該領域的最新進展。){record_delimiter}
{completion_delimiter}

""",
    """[Example 4]

---Input---
Entity_types: [organization,person,location,event,technology,equipment,product,Document,category,financial_metric]
Text:
```
During Apple's Q3 2024 earnings call on July 31, 2024, CEO Tim Cook announced that iPhone revenue reached $39.3 billion, representing a 1.5% decline compared to Q3 2023. The company also reported that Services revenue grew to $24.2 billion, up 14% year-over-year. Cook highlighted that despite supply chain challenges with key supplier TSMC, Apple maintained strong performance in the Greater China region with $14.7 billion in revenue.
```

---Output---
(entity{tuple_delimiter}Apple{tuple_delimiter}organization{tuple_delimiter}Apple is a technology company that reported quarterly earnings, showing mixed performance across product categories with iPhone revenue declining but Services revenue growing.){record_delimiter}
(entity{tuple_delimiter}Tim Cook{tuple_delimiter}person{tuple_delimiter}Tim Cook is Apple's CEO who presented the company's Q3 2024 earnings results, highlighting both challenges and successes.){record_delimiter}
(entity{tuple_delimiter}iPhone Revenue{tuple_delimiter}financial_metric{tuple_delimiter}iPhone revenue reached $39.3 billion, representing a 1.5% decline compared to the previous year's Q3 results.){record_delimiter}
(entity{tuple_delimiter}Services Revenue{tuple_delimiter}financial_metric{tuple_delimiter}Services revenue grew to $24.2 billion, up 14% year-over-year, demonstrating strong growth in Apple's services business.){record_delimiter}
(entity{tuple_delimiter}TSMC{tuple_delimiter}organization{tuple_delimiter}TSMC is identified as a key supplier to Apple, currently experiencing supply chain challenges that affect Apple's operations.){record_delimiter}
(entity{tuple_delimiter}Greater China Revenue{tuple_delimiter}financial_metric{tuple_delimiter}Greater China region generated $14.7 billion in revenue for Apple, maintaining strong regional performance despite challenges.){record_delimiter}
(entity{tuple_delimiter}Q3 2024 Earnings Call{tuple_delimiter}event{tuple_delimiter}Apple's quarterly earnings call held on July 31, 2024, where financial results and strategic updates were announced to investors and analysts.){record_delimiter}
(relationship{tuple_delimiter}Tim Cook{tuple_delimiter}Apple{tuple_delimiter}leadership, earnings presentation{tuple_delimiter}Tim Cook serves as Apple's CEO and presented the company's Q3 2024 earnings results, providing strategic guidance to stakeholders.){record_delimiter}
(relationship{tuple_delimiter}Apple{tuple_delimiter}iPhone Revenue{tuple_delimiter}product performance, financial results{tuple_delimiter}Apple reported iPhone revenue of $39.3 billion, showing a 1.5% decline from the previous year, indicating challenges in the smartphone market.){record_delimiter}
(relationship{tuple_delimiter}Apple{tuple_delimiter}Services Revenue{tuple_delimiter}business segment, growth performance{tuple_delimiter}Apple's Services division generated $24.2 billion in revenue, demonstrating 14% year-over-year growth and highlighting the importance of services to Apple's business model.){record_delimiter}
(relationship{tuple_delimiter}Apple{tuple_delimiter}TSMC{tuple_delimiter}supplier relationship, supply chain challenges{tuple_delimiter}Apple faces supply chain challenges with key supplier TSMC, impacting the company's ability to meet product demand despite overall strong performance.){record_delimiter}
(relationship{tuple_delimiter}Apple{tuple_delimiter}Greater China Revenue{tuple_delimiter}regional performance, market presence{tuple_delimiter}Apple maintained strong performance in Greater China with $14.7 billion in quarterly revenue, demonstrating resilience in a key market.){record_delimiter}
{completion_delimiter}

""",
    """[Example 5]

---Input---
Entity_types: [organization,financial_metric,account_item,currency,temporal_range]
Text:
```
TechCorp Q3 2024 Revenue Summary
Currency: USD thousands

<table>
<thead>
<tr><th>Account</th><th>Q3 2024</th><th>Q2 2024</th></tr>
</thead>
<tbody>
<tr><td>Product Sales</td><td>$1,200</td><td>$1,150</td></tr>
<tr><td>Service Revenue</td><td>$800</td><td>$750</td></tr>
<tr><td>Total Revenue</td><td>$2,000</td><td>$1,900</td></tr>
</tbody>
</table>
```

---Output---
(entity{tuple_delimiter}TechCorp{tuple_delimiter}organization{tuple_delimiter}TechCorp is a technology company reporting quarterly revenue growth from Q2 to Q3 2024.){record_delimiter}
(entity{tuple_delimiter}Q3 2024{tuple_delimiter}temporal_range{tuple_delimiter}Q3 2024 reporting period with total revenue of $2,000 thousand.){record_delimiter}
(entity{tuple_delimiter}Q2 2024{tuple_delimiter}temporal_range{tuple_delimiter}Q2 2024 comparison period with total revenue of $1,900 thousand.){record_delimiter}
(entity{tuple_delimiter}USD thousands{tuple_delimiter}currency{tuple_delimiter}USD thousands is the monetary unit for all revenue figures.){record_delimiter}
(entity{tuple_delimiter}Product Sales{tuple_delimiter}financial_metric{tuple_delimiter}Product Sales revenue demonstrating growth from $1,150 thousand in Q2 2024 to $1,200 thousand in Q3 2024, representing a 4.3% quarter-over-quarter increase and strong product demand.){record_delimiter}
(entity{tuple_delimiter}Service Revenue{tuple_delimiter}financial_metric{tuple_delimiter}Service Revenue showing robust growth from $750 thousand in Q2 2024 to $800 thousand in Q3 2024, achieving a 6.7% quarterly improvement and reflecting expanding service offerings.){record_delimiter}
(entity{tuple_delimiter}Total Revenue{tuple_delimiter}financial_metric{tuple_delimiter}Total Revenue increased from $1,900 thousand in Q2 2024 to $2,000 thousand in Q3 2024, delivering overall business growth of 5.3% and demonstrating TechCorp's consistent performance trajectory.){record_delimiter}
(relationship{tuple_delimiter}Product Sales{tuple_delimiter}Service Revenue{tuple_delimiter}revenue composition{tuple_delimiter}Product Sales and Service Revenue are the two main components of TechCorp's total revenue.){record_delimiter}
(relationship{tuple_delimiter}Product Sales{tuple_delimiter}Total Revenue{tuple_delimiter}financial calculation{tuple_delimiter}Product Sales contributes to Total Revenue calculation.){record_delimiter}
(relationship{tuple_delimiter}Service Revenue{tuple_delimiter}Total Revenue{tuple_delimiter}financial calculation{tuple_delimiter}Service Revenue contributes to Total Revenue calculation.){record_delimiter}
(relationship{tuple_delimiter}Q3 2024{tuple_delimiter}Q2 2024{tuple_delimiter}temporal comparison{tuple_delimiter}Q3 2024 shows revenue growth compared to Q2 2024.){record_delimiter}
{completion_delimiter}
""",
    """[Example 6]

---Input---
Entity_types: [organization,financial_metric,account_item,currency,temporal_range]
Text:
```
台灣積體電路製造股份有限公司及子公司
合併綜合損益表
民國114年及113年1月1日至3月31日
單位：新台幣仟元

<table>
<thead>
<tr><th>代碼</th><th></th><th>114年1月1日至3月31日</th><th>113年1月1日至3月31日</th></tr>
</thead>
<tbody>
<tr><td>4000</td><td>營業收入淨額</td><td>$ 839,253,664</td><td>$ 592,644,201</td></tr>
<tr><td>5000</td><td>營業成本</td><td>345,858,588</td><td>268,447,198</td></tr>
<tr><td>5950</td><td>營業毛利</td><td>493,395,076</td><td>324,197,003</td></tr>
</tbody>
</table>
```

---Output---
(entity{tuple_delimiter}台灣積體電路製造股份有限公司{tuple_delimiter}organization{tuple_delimiter}台灣積體電路製造股份有限公司是一家半導體製造公司，發布民國114年第一季合併綜合損益表，展示營運績效。){record_delimiter}
(entity{tuple_delimiter}民國114年1月1日至3月31日{tuple_delimiter}temporal_range{tuple_delimiter}民國114年第一季報告期間，對應公司最新的財務表現。){record_delimiter}
(entity{tuple_delimiter}民國113年1月1日至3月31日{tuple_delimiter}temporal_range{tuple_delimiter}民國113年第一季比較期間，用於分析公司年度變化趨勢。){record_delimiter}
(entity{tuple_delimiter}新台幣仟元{tuple_delimiter}currency{tuple_delimiter}新台幣仟元是財務報表的主要計價單位，用於表示金額數據。){record_delimiter}
(entity{tuple_delimiter}營業收入淨額{tuple_delimiter}financial_metric{tuple_delimiter}營業收入淨額從民國113年第一季的592,644,201新台幣仟元大幅增長至民國114年第一季的839,253,664新台幣仟元，年增長率達41.6%，顯示台積公司強勁的營運表現和市場需求成長。){record_delimiter}
(entity{tuple_delimiter}營業成本{tuple_delimiter}financial_metric{tuple_delimiter}營業成本從民國113年第一季的268,447,198新台幣仟元增至民國114年第一季的345,858,588新台幣仟元，年增長率28.9%，反映產能擴張和原材料成本上升。){record_delimiter}
(entity{tuple_delimiter}營業毛利{tuple_delimiter}financial_metric{tuple_delimiter}營業毛利從民國113年第一季的324,197,003新台幣仟元躍升至民國114年第一季的493,395,076新台幣仟元，年增長率高達52.2%，展現優異的獲利能力提升和營運效率改善。){record_delimiter}
(relationship{tuple_delimiter}營業收入淨額{tuple_delimiter}營業成本{tuple_delimiter}財務計算, 毛利基礎{tuple_delimiter}營業毛利通過營業收入淨額減去營業成本計算得出。){record_delimiter}
(relationship{tuple_delimiter}民國114年1月1日至3月31日{tuple_delimiter}民國113年1月1日至3月31日{tuple_delimiter}時間比較, 年度分析{tuple_delimiter}兩個報告期間用於分析台積公司年度營運績效變化趨勢。){record_delimiter}
{completion_delimiter}

"""
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist responsible for data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. **Comprehensiveness:** The summary must integrate key information from all provided descriptions. Do not omit important facts.
2. **Context:** The summary must explicitly mention the name of the entity or relation for full context.
3. **Temporal Information:**
   - **If descriptions contain timestamp prefixes** (e.g., "[Time: Fiscal Year 2024-Q1]", "[Time: Fiscal Year 2024-Q3]", "[Time: 2024-07]"), you MUST preserve the exact timestamp format at the beginning of relevant sentences. Organize information chronologically and maintain all temporal markers from the original descriptions.
   - **If descriptions do NOT contain timestamp prefixes**, synthesize the information naturally without adding or inventing temporal markers.

**Example with timestamps:**
Input descriptions:
- "[Time: Fiscal Year 2024-Q1] Apple technology company reported iPhone revenue of $65.8B, up 15% YoY with strong market performance"
- "[Time: Fiscal Year 2024-Q3] Apple technology company faced challenges with iPhone revenue of $39.3B, down 1.5% YoY due to market saturation"

Required output format:
"Apple technology company shows mixed quarterly performance. [Time: Fiscal Year 2024-Q1] Apple reported iPhone revenue of $65.8B, up 15% YoY with strong market performance. [Time: Fiscal Year 2024-Q3] Apple faced challenges with iPhone revenue of $39.3B, down 1.5% YoY due to market saturation."

**Example without timestamps:**
Input descriptions:
- "Apple is a technology company headquartered in Cupertino, California"
- "Apple designs and manufactures consumer electronics, software, and online services"

Required output format:
"Apple is a technology company headquartered in Cupertino, California that designs and manufactures consumer electronics, software, and online services."

DO NOT add timestamp prefixes like "[Time: ...]" when they are not present in the original descriptions.
4. **Conflict Resolution:** In case of conflicting or inconsistent descriptions from different time periods, prioritize more recent information while preserving historical context. If conflicts arise from distinct entities sharing the same name, treat them separately. When temporal information conflicts (e.g., same quarter with different data), note the discrepancy explicitly.
5. **Relevance Filtering:** Focus on substantive information. Remove redundant phrases while preserving unique details from each description.
6. **Style:** The output must be written from an objective, third-person perspective.
7. **Length:** Maintain depth and completeness while ensuring the summary's length not exceed {summary_length} tokens.
8. **Language:** The entire output must be written in {language}.

---Data---
{description_type} Name: {description_name}
Description List:
{description_list}

---Output---
"""

PROMPTS["entity_continue_extraction"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

# TODO: Deprecated
PROMPTS["entity_if_loop_extraction"] = """
---Goal---'

Check if it appears some entities may have still been missed. Output "Yes" if so, otherwise "No".

---Output---
Output:"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---
You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided `Source Data`.

---Goal---
Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the `Source Data`.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---
1. **Think Step-by-Step:**
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize the `Source Data`(both Knowledge Graph and Document Chunks). Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.

2. **Content & Grounding:**
  - Strictly adhere to the provided context from the `Source Data`; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the `Source Data`, state that you do not have enough information to answer. Do not attempt to guess.

3. **Formatting & Language:**
  - The response MUST be in the same language as the user query.
  - Use Markdown for clear formatting (e.g., headings, bold, lists).
  - The response should be presented in {response_type}.
  - Append a reference section at the end of the response.
  - Merge citations that share the same file_path into one reference item.
  - The main body of the response should exclude inline citations; all citation information should be listed exclusively in the references section.

4. **Reference/Citation Format:**
  - Append a reference section at the end of the response.
  - The References section should be under a `### References` heading.
  - Output the citation in the following formats:
    - For a Knowledge Graph Entity: [EN] <entity>
    - For a Knowledge Graph Relationship: [RE] <entity1> ~ <entity2>
    - For a Document Chunk: [DC] <file_path>
  - <entity>, <entity1>, <entity2>, and <file_path> should originate from attribute values in `Source Data` and be retained in their original language.
  - Merge citations that share the same <file_path> into one reference item, disregarding their distinct IDs.
  - Only include citations that directly reference the facts presented in the answer.
  - Prioritize the most relevant references, and provide maximum of 6 most relevant citations.
  - List each citation on an individual line.

5. **Example of Section:**
```
### References
- [EN] LightRAG
- [EN] Dual-Level Retrieval System
- [RE] LightRAG ~ GraphRAG
- [DC] Simple and Fast RAG.pdf
- [DC] LightRAG Simple and Fast Alternative to GraphRAG for Legal Doc Analysis.md
- [DC] Microsoft GraphRAG Technology Summary.md
```

6. **Table Data Processing:** When Source Data contains tabular information:
  - Preserve Exact Values: Maintain original numerical values and currency units from source data
  - Table Formatting: Use Markdown table syntax to present structured data clearly
  - Units and Currency: Always include original units (e.g., "新台幣仟元", "USD thousands")
  - Mathematical Relationships: When source data shows calculations, preserve the relationship context
  - Temporal Data: Maintain time period labels exactly as provided in source data

---USER CONTEXT---
- Additional user prompt: {user_prompt}

---Source Data---
Knowledge Graph and Document Chunks:

{context_data}


"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided provided in JSON format below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

---Conversation History---
{history}

---Document Chunks(DC)---
{content_data}

---RESPONSE GUIDELINES---
**1. Content & Adherence:**
- Strictly adhere to the provided context from the Knowledge Base. Do not invent, assume, or include any information not present in the source data.
- If the answer cannot be found in the provided context, state that you do not have enough information to answer.
- Ensure the response maintains continuity with the conversation history.

**2. Formatting & Language:**
- Format the response using markdown with appropriate section headings.
- The response language must match the user's question language.
- Target format and length: {response_type}

**3. Citations / References:**
- At the end of the response, under a "References" section, cite a maximum of 5 most relevant sources used.
- Use the following formats for citations: `[DC] <file_path_or_document_name>`

---USER CONTEXT---
- Additional user prompt: {user_prompt}

---Response---
Output:"""

# Entity Merging Prompts
PROMPTS["entity_merge_system"] = """You are a deduplication evaluator. Given two entities (A, B) with (entity_id, description),
you must carefully compare both entity_id and description.

Only when you are OVER 95% confident that A and B refer to the SAME real-world entity,
you should INVOKE the tool: merge(a_entity_id, b_entity_id).

If you are NOT over 95% confident (or they should not be merged), do NOT invoke any tool
and simply reply with the single token: NO_MERGE."""

PROMPTS["entity_merge_examples"] = """Example
#### Merge ###
A.entity_id = Apple
A.description = A technology company known for designing and manufacturing consumer electronics, software, and services.

B.entity_id = Apple Inc.
B.description = An American multinational technology company headquartered in Cupertino, California.

#### No merge ###
A.entity_id = Apple
A.description = A technology company known for designing and manufacturing consumer electronics, software, and services.

B.entity_id = TSMC
B.description = Taiwan Semiconductor Manufacturing Company, the world's largest semiconductor foundry."""

PROMPTS["entity_merge_user"] = """Determine whether the two entities are the same. Only when you are over 95% confident that A and B refer to the SAME entity, invoke the tool `merge_entities_tool(a_entity_id, b_entity_id)`. If you are not 95% confident, reply exactly with `NO_MERGE`.

A.entity_id = {a_entity_id}
A.description = {a_description}

B.entity_id = {b_entity_id}
B.description = {b_description}
"""

# Entity Type Augmentation Prompts
PROMPTS["entity_type_suggestion_system"] = """
You are an expert in Named Entity Recognition (NER) with expertise across multiple domains. Your goal is to analyze the connections and relations between existing entity types and document content to provide meaningful refinements or additions that enhance entity extraction for various document types.

## Task Requirements:
- Suggest entity types that improve extraction quality for the specific document domain
- Consider the document's context, structure, and content patterns
- Avoid suggesting "other" or "unknown" types
- Do not suggest duplicates or overlapping entity types
- Prioritize quality over quantity with domain-appropriate coverage
- Consider structured data elements like tables, lists, and classifications when present
- Provide concise yet clear explanations with relevant examples
- Respond in strict JSON array format only

## Multi-Domain Context Considerations:
- **Technical Documents**: Components, specifications, procedures, standards, measurements
- **Academic Papers**: Research methods, findings, citations, institutions, datasets
- **Business Documents**: Metrics, processes, departments, strategies, performance indicators
- **Financial Reports**: Assets, revenues, ratios, statements, accounting items
- **News Articles**: Events, locations, quotes, sources, impacts
- **Legal Documents**: Clauses, parties, terms, obligations, references

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
        "entity_type": "organization",
        "explanation": "An entity representing organizations, companies, or institutions."
    },
    {
        "entity_type": "person",
        "explanation": "An entity representing individual persons."
    }
]

### Document Content:
The research team at MIT conducted a systematic review of machine learning algorithms, analyzing 150 datasets from 2020-2024. The study employed cross-validation techniques and achieved an accuracy of 94.2% using deep neural networks. Results were published in Nature Machine Intelligence.

### Suggested New Entity Types:
[
    {
        "entity_type": "research_method",
        "explanation": "Methodological approaches and techniques used in research (e.g., 'systematic review', 'cross-validation', 'deep neural networks')."
    },
    {
        "entity_type": "metric",
        "explanation": "Quantitative measurements and performance indicators (e.g., '94.2%', 'accuracy', 'sample size of 150')."
    },
    {
        "entity_type": "publication",
        "explanation": "Academic publications, journals, and research outputs (e.g., 'Nature Machine Intelligence', 'conference proceedings')."
    }
]
"""

PROMPTS["entity_type_suggestion_user"] = """
## Current Entity Types:
{current_entity_types}

## Task:
Based on the following document content, analyze and suggest new entity types with explanations if needed. Consider the document's domain and structure:

1. **Content Analysis**: Examine the document's subject matter, terminology, and domain-specific concepts
2. **Structural Elements**: Notice tables, lists, classifications, and organized data patterns
3. **Entity Patterns**: Identify recurring entity types that appear frequently in the content
4. **Relationships**: Consider entities that represent important connections and dependencies
5. **Domain Specificity**: Suggest entity types that capture domain-specific knowledge effectively

## Document Content:
{file_content}

Please carefully analyze the entities that appear in the document, considering its structure and domain context, and suggest appropriate new entity types that would improve extraction coverage for this type of content.
"""

PROMPTS["entity_type_refinement_system"] = """
You are an advanced linguistic assistant with expertise in Named Entity Recognition (NER) across multiple domains.

## Task:
Refine a list of entity types by aggressively removing duplicates or semantically overlapping types, ensuring a concise and well-optimized schema.

## Critical Requirements:
- **STRICT CONSOLIDATION**: Aggressively merge similar or overlapping entity types
- **QUALITY OVER QUANTITY**: Aim for the minimum number of entity types that provide maximum coverage
- **HIERARCHICAL MERGING**: Merge child types into broader parent categories when possible
- **NO REDUNDANCY**: Remove any types that can be represented by existing types
- **DOMAIN FOCUS**: Keep only entity types that are essential for the specific document domain

## Consolidation Guidelines:
1. **Merge Overlapping Types**: Combine types with >70% semantic overlap (e.g., "Company" + "Organization" → "Organization")
2. **Eliminate Sub-types**: Remove specific sub-types if parent type exists (e.g., "CEO" → "Person", "Apple Inc." → "Organization")
3. **Remove Generic Types**: Eliminate vague types like "Concept", "Thing", "Item", "Other", "Unknown"
4. **Combine Related Types**: Merge related types into broader categories (e.g., "Product" + "Service" → "Offering")
5. **Preserve Core Types Only**: Keep only types that represent fundamentally different entity categories

## Domain-Specific Considerations:
- **Financial Documents**: Organization, Person, Financial_Metric, Temporal_Range 
- **Academic Papers**: Organization, Person, Research_Method, Publication
- **Technical Documents**: Organization, Person, Component, Specification 

## Response Format:
[
{
    "entity_type": "<entity_type_name>",
    "explanation": "<clear_explanation>"
}
]

## Example 1 - Aggressive Consolidation:
### Entity Types List to Refine:
[
    {"entity_type": "Company", "explanation": "A company is a legal entity..."},
    {"entity_type": "Organization", "explanation": "An organization is a group..."},
    {"entity_type": "Institution", "explanation": "An institution is a formal organization..."},
    {"entity_type": "Corporation", "explanation": "A corporation is a type of company..."},
    {"entity_type": "CEO", "explanation": "A chief executive officer..."},
    {"entity_type": "Employee", "explanation": "A person working for an organization..."},
    {"entity_type": "Person", "explanation": "An individual person..."},
    {"entity_type": "Human", "explanation": "A human being..."}
]

### Refined List:
[
    {
        "entity_type": "Organization",
        "explanation": "An entity representing organizations, companies, institutions, or corporations (consolidated from Company, Organization, Institution, Corporation)."
    },
    {
        "entity_type": "Person",
        "explanation": "An entity representing individual persons including executives, employees, and other human entities (consolidated from Person, Human, CEO, Employee)."
    }
]

## Example 2 - Domain-Specific Consolidation:
### Entity Types List to Refine:
[
    {"entity_type": "Revenue", "explanation": "Company revenue..."},
    {"entity_type": "Profit", "explanation": "Company profit..."},
    {"entity_type": "Loss", "explanation": "Financial loss..."},
    {"entity_type": "Growth_Rate", "explanation": "Growth percentage..."},
    {"entity_type": "Margin", "explanation": "Profit margin..."},
    {"entity_type": "Metric", "explanation": "Performance metric..."},
    {"entity_type": "Quarter", "explanation": "Fiscal quarter..."},
    {"entity_type": "Year", "explanation": "Fiscal year..."},
    {"entity_type": "Time_Period", "explanation": "Time period..."},
    {"entity_type": "Fiscal_Period", "explanation": "Fiscal period..."}
]

### Refined List:
[
    {
        "entity_type": "Organization",
        "explanation": "Organizations, companies, or institutions."
    },
    {
        "entity_type": "Person",
        "explanation": "Individual persons."
    },
    {
        "entity_type": "Financial_Metric",
        "explanation": "Financial measurements including revenue, profit, loss, growth rate, margin, and other performance metrics (consolidated from Revenue, Profit, Loss, Growth_Rate, Margin, Metric)."
    },
    {
        "entity_type": "Temporal_Range",
        "explanation": "Time periods including quarters (Q1-Q4), fiscal years, and other temporal ranges (consolidated from Quarter, Year, Time_Period, Q1, Q2, Fiscal_Period)."
    }
]
"""

PROMPTS["entity_type_refinement_user"] = """
## Entity Types List to Refine:
{entity_types}

## Task:
**CRITICAL**: Aggressively consolidate this list to the minimum number of entity types needed.

**Your Goals:**
1. **Maximize Consolidation**: Merge as many similar types as possible
2. **Eliminate Redundancy**: Remove any type that can be covered by another type
3. **Preserve Coverage**: Ensure the final types still cover all important entity categories

**Action Steps:**
- Group semantically similar types together
- For each group, select the most comprehensive type name
- Merge explanations to reflect consolidated coverage
- Remove sub-types that are covered by parent types
- Eliminate generic or vague type names

**Quality Check:**
- Can any two types be merged? If yes, merge them.
- Is each type fundamentally different from others? If no, consolidate.
- Does each type capture unique entities? If no, remove it.

Please provide the refined list in strict JSON array format.
"""
