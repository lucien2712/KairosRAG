from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_USER_PROMPT"] = "n/a"

PROMPTS["entity_extraction"] = """---Task---
Given a text document and a list of entity types, identify all entities of those types and all relationships among the identified entities.

---Instructions---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1. **Entity Extraction:** Identify clearly defined and meaningful entities in the input text, and extract the following information:
  - entity_name: Name of the entity, ensure entity names are consistent throughout the extraction.
  - entity_type: Categorize the entity using the following entity types: {entity_types}; if none of the provided types are suitable, classify it as `Other`.
  - entity_description: Provide a comprehensive description of the entity's attributes and activities based on the information present in the input text.
2. **Entity Output Format:** (entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description)
3. **Relationship Extraction:** Identify direct, clearly-stated and meaningful relationships between extracted entities within the input text, and extract the following information:
  - source_entity: name of the source entity.
  - target_entity: name of the target entity.
  - relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details.
  - relationship_description: Explain the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
4. **Relationship Output Format:** (relationship{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description)
5. **Relationship Order:** Prioritize relationships based on their significance to the intended meaning of input text, and output more crucial relationships first.
6. **Avoid Pronouns:** For entity names and all descriptions, explicitly name the subject or object instead of using pronouns; avoid pronouns such as `this document`, `our company`, `I`, `you`, and `he/she`.
7. **Undirectional Relationship:** Treat relationships as undirected; swapping the source and target entities does not constitute a new relationship. Avoid outputting duplicate relationships.
8. **Language:** Output entity names, keywords and descriptions in {language}.
9. **Delimiter:** Use `{record_delimiter}` as the entity or relationship list delimiter; output `{completion_delimiter}` when all the entities and relationships are extracted.

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
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
```

---Output---
(entity{tuple_delimiter}World Athletics Championship{tuple_delimiter}event{tuple_delimiter}The World Athletics Championship is a global sports competition featuring top athletes in track and field. This championship event is being held in Tokyo in 2024.){record_delimiter}
(entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the host city of the World Athletics Championship where the historic 100m sprint record was broken.){record_delimiter}
(entity{tuple_delimiter}Noah Carter{tuple_delimiter}person{tuple_delimiter}Noah Carter is a sprinter who achieved a historic milestone by breaking the 100m sprint record using cutting-edge carbon-fiber spikes at the World Athletics Championship in Tokyo.){record_delimiter}
(entity{tuple_delimiter}100m Sprint Record{tuple_delimiter}category{tuple_delimiter}The 100m sprint record is a benchmark in athletics, representing the fastest recorded time for the 100-meter sprint distance.){record_delimiter}
(entity{tuple_delimiter}Carbon-Fiber Spikes{tuple_delimiter}equipment{tuple_delimiter}Carbon-fiber spikes are cutting-edge advanced sprinting shoes designed to enhance speed and traction for competitive runners.){record_delimiter}
(relationship{tuple_delimiter}World Athletics Championship{tuple_delimiter}Tokyo{tuple_delimiter}event location, international competition{tuple_delimiter}The World Athletics Championship is being hosted in Tokyo, providing the venue for the historic record-breaking 100m sprint.){record_delimiter}
(relationship{tuple_delimiter}Noah Carter{tuple_delimiter}100m Sprint Record{tuple_delimiter}athlete achievement, record-breaking{tuple_delimiter}Noah Carter broke the 100m sprint record at the World Athletics Championship, marking a significant achievement in track and field history.){record_delimiter}
(relationship{tuple_delimiter}Noah Carter{tuple_delimiter}Carbon-Fiber Spikes{tuple_delimiter}athletic equipment, performance boost{tuple_delimiter}Noah Carter used cutting-edge carbon-fiber spikes to enhance his performance during the record-breaking race.){record_delimiter}
(relationship{tuple_delimiter}Noah Carter{tuple_delimiter}World Athletics Championship{tuple_delimiter}athlete participation, competition{tuple_delimiter}Noah Carter participated in the World Athletics Championship in Tokyo where he achieved his record-breaking performance.){record_delimiter}
{completion_delimiter}

""",
    """[Example 4]

---Input---
Entity_types: [organization,person,location,event,technology,equiment,product,Document,category]
Text:
```
在北京举行的人工智能大会上，腾讯公司的首席技术官张伟发布了最新的大语言模型"腾讯智言"，该模型在自然语言处理方面取得了重大突破。
```

---Output---
(entity{tuple_delimiter}人工智能大会{tuple_delimiter}event{tuple_delimiter}人工智能大会是在北京举行的技术会议，专注于人工智能领域的最新发展，腾讯公司在此发布了新产品。){record_delimiter}
(entity{tuple_delimiter}北京{tuple_delimiter}location{tuple_delimiter}北京是人工智能大会的举办城市，见证了腾讯智言大语言模型的重要发布。){record_delimiter}
(entity{tuple_delimiter}腾讯公司{tuple_delimiter}organization{tuple_delimiter}腾讯公司是参与人工智能大会的科技企业，通过首席技术官张伟发布了新的大语言模型产品。){record_delimiter}
(entity{tuple_delimiter}张伟{tuple_delimiter}person{tuple_delimiter}张伟是腾讯公司的首席技术官，在北京举行的人工智能大会上发布了腾讯智言产品。){record_delimiter}
(entity{tuple_delimiter}腾讯智言{tuple_delimiter}product{tuple_delimiter}腾讯智言是腾讯公司在人工智能大会上发布的大语言模型产品，在自然语言处理方面取得了重大突破。){record_delimiter}
(entity{tuple_delimiter}自然语言处理技术{tuple_delimiter}technology{tuple_delimiter}自然语言处理技术是腾讯智言模型取得重大突破的技术领域，展现了最新发展成果。){record_delimiter}
(relationship{tuple_delimiter}人工智能大会{tuple_delimiter}北京{tuple_delimiter}会议地点, 举办关系{tuple_delimiter}人工智能大会在北京举行，成为腾讯等科技企业展示最新技术的重要平台。){record_delimiter}
(relationship{tuple_delimiter}张伟{tuple_delimiter}腾讯公司{tuple_delimiter}雇佣关系, 高管职位{tuple_delimiter}张伟担任腾讯公司的首席技术官，代表公司在人工智能大会上进行重要产品发布。){record_delimiter}
(relationship{tuple_delimiter}张伟{tuple_delimiter}腾讯智言{tuple_delimiter}产品发布, 技术展示{tuple_delimiter}张伟在人工智能大会上发布了腾讯智言大语言模型，展示了公司在AI领域的技术实力。){record_delimiter}
(relationship{tuple_delimiter}腾讯智言{tuple_delimiter}自然语言处理技术{tuple_delimiter}技术应用, 突破创新{tuple_delimiter}腾讯智言在自然语言处理技术方面取得了重大突破，代表了该领域的最新进展。){record_delimiter}
{completion_delimiter}

""",
    """[Example 5 - Temporal Financial Information]

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
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist responsible for data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. **Comprehensiveness:** The summary must integrate key information from all provided descriptions. Do not omit important facts.
2. **Context:** The summary must explicitly mention the name of the entity or relation for full context.
3. **Temporal Information:** When descriptions contain timestamp prefixes (e.g., "Fiscal Year 2024-Q1:", "Fiscal Year 2024-Q3:", "2024-07:"), you MUST preserve the exact timestamp format at the beginning of relevant sentences. Organize information chronologically and maintain all temporal markers from the original descriptions.

**Example:**
Input descriptions:
- "Fiscal Year 2024-Q1: Apple technology company reported iPhone revenue of $65.8B, up 15% YoY with strong market performance"
- "Fiscal Year 2024-Q3: Apple technology company faced challenges with iPhone revenue of $39.3B, down 1.5% YoY due to market saturation"

Required output format:
"Apple technology company shows mixed quarterly performance. Fiscal Year 2024-Q1: Apple reported iPhone revenue of $65.8B, up 15% YoY with strong market performance. Fiscal Year 2024-Q3: Apple faced challenges with iPhone revenue of $39.3B, down 1.5% YoY due to market saturation."

DO NOT generate: "Apple technology company performance varied across 2024 quarters with early growth followed by later challenges."
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
Identify any missed entities or relationships in the last extraction task.

---Instructions---
1. Output the entities and realtionships in the same format as previous extraction task.
2. Do not include entities and relations that have been previously extracted.
3. If the entity doesn't clearly fit in any of`Entity_types` provided, classify it as "Other".
4. Return identified entities and relationships in {language}.
5. Output `{completion_delimiter}` when all the entities and relationships are extracted.

---Output---
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

You are a helpful assistant responding to user query about Knowledge Graph and Document Chunks provided in JSON format below.


---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both current query and the conversation history if provided. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

---Conversation History---
{history}

---Knowledge Graph and Document Chunks---
{context_data}

---Response Guidelines---
**1. Content & Adherence:**
- Strictly adhere to the provided context from the Knowledge Base. Do not invent, assume, or include any information not present in the source data.
- If the answer cannot be found in the provided context, state that you do not have enough information to answer.
- Ensure the response maintains continuity with the conversation history.

**2. Formatting & Language:**
- Format the response using markdown with appropriate section headings.
- The response language must in the same language as the user's question.
- Target format and length: {response_type}

**3. Citations / References:**
- At the end of the response, under a "References" section, each citation must clearly indicate its origin (KG or DC).
- The maximum number of citations is 5, including both KG and DC.
- Use the following formats for citations:
  - For a Knowledge Graph Entity: `[KG] <entity_name>`
  - For a Knowledge Graph Relationship: `[KG] <entity1_name> - <entity2_name>`
  - For a Document Chunk: `[DC] <file_path_or_document_name>`

---USER CONTEXT---
- Additional user prompt: {user_prompt}

---Response---
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

# Entity Type Augmentation Prompts
PROMPTS["entity_type_suggestion_system"] = """
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
"""

PROMPTS["entity_type_suggestion_user"] = """
## Current Entity Types:
{current_entity_types}

## Task:
Based on the following document content, analyze and suggest new entity types with explanations if needed.

## Document Content:
{file_content}

Please carefully analyze the entities that appear in the document and suggest appropriate new entity types (if any are needed).
"""

PROMPTS["entity_type_refinement_system"] = """
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
"""

PROMPTS["entity_type_refinement_user"] = """
## Entity Types List to Refine:
{entity_types}

## Task:
Please refine this list by removing duplicates or semantically similar entity types. Keep the most appropriate and comprehensive entity type when consolidating similar ones.
"""
