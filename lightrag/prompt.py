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
1. Recognizing definitively conceptualized entities in text. For each identified entity, extract the following information:
  - entity_name: Name of the entity, use same language as input text. If English, capitalized the name
  - entity_type: Categorize the entity using the provided `Entity_types` list. If a suitable category cannot be determined, classify it as "Other".
  - entity_description: Provide a comprehensive description of the entity's attributes and activities based on the information present in the input text. IMPORTANT: If a timestamp is provided, begin the description with the specific time reference (e.g., "{timestamp}: "). If no timestamp is provided but temporal information is available in the text (dates, quarters, years, time periods), begin the description with the specific time reference found in the text. To ensure clarity and precision, all descriptions must replace pronouns and referential terms (e.g., "this document," "our company," "I," "you," "he/she") with the specific nouns they represent.
2. Format each entity as: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
3. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are directly and clearly related based on the text. Unsubstantiated relationships must be excluded from the output.
For each pair of related entities, extract the following information:
  - source_entity: name of the source entity, as identified in step 1
  - target_entity: name of the target entity, as identified in step 1
  - relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
  - relationship_description: Explain the nature of the relationship between the source and target entities, providing a clear rationale for their connection. IMPORTANT: If a timestamp is provided, begin the description with the specific time reference (e.g., "{timestamp}: "). If no timestamp is provided but temporal information is available in the text, begin the description with the specific time reference found in the text to indicate when this relationship was observed or reported.
4. Format each relationship as: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_description>)
5. Use `{tuple_delimiter}` as field delimiter. Use `{record_delimiter}` as the entity or relation list delimiter.
6. Return identified entities and relationships in {language}.
7. Output `{completion_delimiter}` when all the entities and relationships are extracted.

---Quality Guidelines---
- Only extract entities that are clearly defined and meaningful in the context
- Avoid over-interpretation; stick to what is explicitly stated in the text
- For all output content, explicitly name the subject or object rather than using pronouns
- Include specific numerical data in entity name when relevant
- Ensure entity names are consistent throughout the extraction

---Examples---
{examples}

---Input---
Entity_types: [{entity_types}]
Timestamp: {timestamp}
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
Timestamp: 2024-03-15
Text:
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

---Output---
(entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}2024-03-15: Alex is a character who experiences frustration and is observant of the dynamics among other characters.){record_delimiter}
(entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}2024-03-15: Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.){record_delimiter}
(entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}2024-03-15: Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.){record_delimiter}
(entity{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}2024-03-15: Cruz is associated with a vision of control and order, influencing the dynamics among other characters.){record_delimiter}
(entity{tuple_delimiter}The Device{tuple_delimiter}equiment{tuple_delimiter}2024-03-15: The Device is central to the story, with potential game-changing implications, and is revered by Taylor.){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}power dynamics, observation{tuple_delimiter}2024-03-15: Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device.){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}shared goals, rebellion{tuple_delimiter}2024-03-15: Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.){record_delimiter}
(relationship{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}conflict resolution, mutual respect{tuple_delimiter}2024-03-15: Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.){record_delimiter}
(relationship{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}ideological conflict, rebellion{tuple_delimiter}2024-03-15: Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.){record_delimiter}
(relationship{tuple_delimiter}Taylor{tuple_delimiter}The Device{tuple_delimiter}reverence, technological significance{tuple_delimiter}2024-03-15: Taylor shows reverence towards the device, indicating its importance and potential impact.){record_delimiter}
{completion_delimiter}

""",
    """[Example 2]

---Input---
Entity_types: [organization,person,location,event,technology,equiment,product,Document,category]
Timestamp: 2024-03-14
Text:
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the Global Tech Index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

---Output---
(entity{tuple_delimiter}Global Tech Index{tuple_delimiter}category{tuple_delimiter}2024-03-14: The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline in midday trading.){record_delimiter}
(entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}organization{tuple_delimiter}2024-03-14: Nexon Technologies is a tech company that saw its stock decline by 7.8% after reporting lower-than-expected quarterly earnings.){record_delimiter}
(entity{tuple_delimiter}Omega Energy{tuple_delimiter}organization{tuple_delimiter}2024-03-14: Omega Energy is an energy company that gained 2.1% in stock value, driven by rising oil prices.){record_delimiter}
(entity{tuple_delimiter}Gold Futures{tuple_delimiter}product{tuple_delimiter}2024-03-14: Gold futures rose by 1.5% to $2,080 per ounce, indicating increased investor interest in safe-haven assets.){record_delimiter}
(entity{tuple_delimiter}Crude Oil{tuple_delimiter}product{tuple_delimiter}2024-03-14: Crude oil prices continued their rally, climbing to $87.60 per barrel due to supply constraints and strong demand.){record_delimiter}
(entity{tuple_delimiter}Market Selloff{tuple_delimiter}category{tuple_delimiter}2024-03-14: Market selloff refers to the significant decline in stock values due to investor concerns over rising interest rates and regulatory uncertainty.){record_delimiter}
(entity{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}category{tuple_delimiter}2024-03-14: The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and overall market stability.){record_delimiter}
(relationship{tuple_delimiter}Global Tech Index{tuple_delimiter}Market Selloff{tuple_delimiter}market performance, investor sentiment{tuple_delimiter}2024-03-14: The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns over rising interest rates.){record_delimiter}
(relationship{tuple_delimiter}Nexon Technologies{tuple_delimiter}Global Tech Index{tuple_delimiter}company impact, index movement{tuple_delimiter}2024-03-14: Nexon Technologies' 7.8% stock decline contributed to the overall drop in the Global Tech Index.){record_delimiter}
(relationship{tuple_delimiter}Gold Futures{tuple_delimiter}Market Selloff{tuple_delimiter}market reaction, safe-haven investment{tuple_delimiter}2024-03-14: Gold prices rose as investors sought safe-haven assets during the market selloff.){record_delimiter}
(relationship{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}Market Selloff{tuple_delimiter}interest rate impact, financial regulation{tuple_delimiter}2024-03-14: Speculation over Federal Reserve policy changes contributed to market volatility and the investor selloff.){record_delimiter}
{completion_delimiter}

""",
    """[Example 3]

---Input---
Entity_types: [organization,person,location,event,technology,equiment,product,Document,category]
Timestamp: 2024-07-20
Text:
```
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
```

---Output---
(entity{tuple_delimiter}World Athletics Championship{tuple_delimiter}event{tuple_delimiter}2024-07-20: The World Athletics Championship is a global sports competition featuring top athletes in track and field, being held in Tokyo.){record_delimiter}
(entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}2024-07-20: Tokyo is the host city of the World Athletics Championship where the 100m sprint record was broken.){record_delimiter}
(entity{tuple_delimiter}Noah Carter{tuple_delimiter}person{tuple_delimiter}2024-07-20: Noah Carter is a sprinter who broke the 100m sprint record using cutting-edge carbon-fiber spikes at the World Athletics Championship in Tokyo.){record_delimiter}
(entity{tuple_delimiter}100m Sprint Record{tuple_delimiter}category{tuple_delimiter}2024-07-20: The 100m sprint record is a benchmark in athletics, broken by Noah Carter at the World Athletics Championship.){record_delimiter}
(entity{tuple_delimiter}Carbon-Fiber Spikes{tuple_delimiter}equipment{tuple_delimiter}2024-07-20: Carbon-fiber spikes are cutting-edge advanced sprinting shoes that Noah Carter used to enhance speed and traction for his record-breaking performance.){record_delimiter}
(relationship{tuple_delimiter}World Athletics Championship{tuple_delimiter}Tokyo{tuple_delimiter}event location, international competition{tuple_delimiter}2024-07-20: The World Athletics Championship is being hosted in Tokyo where the record-breaking 100m sprint took place.){record_delimiter}
(relationship{tuple_delimiter}Noah Carter{tuple_delimiter}100m Sprint Record{tuple_delimiter}athlete achievement, record-breaking{tuple_delimiter}2024-07-20: Noah Carter broke the 100m sprint record at the World Athletics Championship in Tokyo.){record_delimiter}
(relationship{tuple_delimiter}Noah Carter{tuple_delimiter}Carbon-Fiber Spikes{tuple_delimiter}athletic equipment, performance boost{tuple_delimiter}2024-07-20: Noah Carter used cutting-edge carbon-fiber spikes to enhance his performance during the record-breaking race.){record_delimiter}
(relationship{tuple_delimiter}Noah Carter{tuple_delimiter}World Athletics Championship{tuple_delimiter}athlete participation, competition{tuple_delimiter}2024-07-20: Noah Carter participated in the World Athletics Championship in Tokyo where he achieved his record-breaking performance.){record_delimiter}
{completion_delimiter}

""",
    """[Example 4]

---Input---
Entity_types: [organization,person,location,event,technology,equiment,product,Document,category]
Timestamp: 2024年3月10日
Text:
```
在北京举行的人工智能大会上，腾讯公司的首席技术官张伟发布了最新的大语言模型"腾讯智言"，该模型在自然语言处理方面取得了重大突破。
```

---Output---
(entity{tuple_delimiter}人工智能大会{tuple_delimiter}event{tuple_delimiter}2024年3月10日: 人工智能大会是在北京举行的技术会议，专注于人工智能领域的最新发展，腾讯公司在此发布了新产品。){record_delimiter}
(entity{tuple_delimiter}北京{tuple_delimiter}location{tuple_delimiter}2024年3月10日: 北京是人工智能大会的举办城市，见证了腾讯智言大语言模型的重要发布。){record_delimiter}
(entity{tuple_delimiter}腾讯公司{tuple_delimiter}organization{tuple_delimiter}2024年3月10日: 腾讯公司是参与人工智能大会的科技企业，通过首席技术官张伟发布了新的大语言模型产品。){record_delimiter}
(entity{tuple_delimiter}张伟{tuple_delimiter}person{tuple_delimiter}2024年3月10日: 张伟是腾讯公司的首席技术官，在北京举行的人工智能大会上发布了腾讯智言产品。){record_delimiter}
(entity{tuple_delimiter}腾讯智言{tuple_delimiter}product{tuple_delimiter}2024年3月10日: 腾讯智言是腾讯公司在人工智能大会上发布的大语言模型产品，在自然语言处理方面取得了重大突破。){record_delimiter}
(entity{tuple_delimiter}自然语言处理技术{tuple_delimiter}technology{tuple_delimiter}2024年3月10日: 自然语言处理技术是腾讯智言模型取得重大突破的技术领域，展现了最新发展成果。){record_delimiter}
(relationship{tuple_delimiter}人工智能大会{tuple_delimiter}北京{tuple_delimiter}会议地点, 举办关系{tuple_delimiter}2024年3月10日: 人工智能大会在北京举行，成为腾讯等科技企业展示最新技术的重要平台。){record_delimiter}
(relationship{tuple_delimiter}张伟{tuple_delimiter}腾讯公司{tuple_delimiter}雇佣关系, 高管职位{tuple_delimiter}2024年3月10日: 张伟担任腾讯公司的首席技术官，代表公司在人工智能大会上进行重要产品发布。){record_delimiter}
(relationship{tuple_delimiter}张伟{tuple_delimiter}腾讯智言{tuple_delimiter}产品发布, 技术展示{tuple_delimiter}2024年3月10日: 张伟在人工智能大会上发布了腾讯智言大语言模型，展示了公司在AI领域的技术实力。){record_delimiter}
(relationship{tuple_delimiter}腾讯智言{tuple_delimiter}自然语言处理技术{tuple_delimiter}技术应用, 突破创新{tuple_delimiter}2024年3月10日: 腾讯智言在自然语言处理技术方面取得了重大突破，代表了该领域的最新进展。){record_delimiter}
{completion_delimiter}

""",
    """[Example 5 - Temporal Financial Information]

---Input---
Entity_types: [organization,person,location,event,technology,equipment,product,Document,category,financial_metric]
Timestamp: 2024-07-31 Q3 2024
Text:
```
During Apple's Q3 2024 earnings call on July 31, 2024, CEO Tim Cook announced that iPhone revenue reached $39.3 billion, representing a 1.5% decline compared to Q3 2023. The company also reported that Services revenue grew to $24.2 billion, up 14% year-over-year. Cook highlighted that despite supply chain challenges with key supplier TSMC, Apple maintained strong performance in the Greater China region with $14.7 billion in revenue.
```

---Output---
(entity{tuple_delimiter}Apple{tuple_delimiter}organization{tuple_delimiter}Q3 2024: Apple is a technology company that reported quarterly earnings on July 31, 2024, showing mixed performance across product categories.){record_delimiter}
(entity{tuple_delimiter}Tim Cook{tuple_delimiter}person{tuple_delimiter}Q3 2024: Tim Cook is Apple's CEO who presented the company's Q3 2024 earnings results on July 31, 2024.){record_delimiter}
(entity{tuple_delimiter}iPhone Revenue{tuple_delimiter}financial_metric{tuple_delimiter}Q3 2024: iPhone revenue reached $39.3 billion, representing a 1.5% decline compared to Q3 2023.){record_delimiter}
(entity{tuple_delimiter}Services Revenue{tuple_delimiter}financial_metric{tuple_delimiter}Q3 2024: Services revenue grew to $24.2 billion, up 14% year-over-year from Q3 2023.){record_delimiter}
(entity{tuple_delimiter}TSMC{tuple_delimiter}organization{tuple_delimiter}Q3 2024: TSMC is identified as a key supplier to Apple experiencing supply chain challenges during Q3 2024.){record_delimiter}
(entity{tuple_delimiter}Greater China Revenue{tuple_delimiter}financial_metric{tuple_delimiter}Q3 2024: Greater China region generated $14.7 billion in revenue for Apple during Q3 2024.){record_delimiter}
(entity{tuple_delimiter}Q3 2024 Earnings Call{tuple_delimiter}event{tuple_delimiter}Q3 2024: Apple's quarterly earnings call held on July 31, 2024, where financial results and strategic updates were announced.){record_delimiter}
(relationship{tuple_delimiter}Tim Cook{tuple_delimiter}Apple{tuple_delimiter}leadership, earnings presentation{tuple_delimiter}Q3 2024: Tim Cook serves as Apple's CEO and presented the company's Q3 2024 earnings results.){record_delimiter}
(relationship{tuple_delimiter}Apple{tuple_delimiter}iPhone Revenue{tuple_delimiter}product performance, financial results{tuple_delimiter}Q3 2024: Apple reported iPhone revenue of $39.3 billion, showing a 1.5% decline from the previous year.){record_delimiter}
(relationship{tuple_delimiter}Apple{tuple_delimiter}Services Revenue{tuple_delimiter}business segment, growth performance{tuple_delimiter}Q3 2024: Apple's Services division generated $24.2 billion in revenue, demonstrating 14% year-over-year growth.){record_delimiter}
(relationship{tuple_delimiter}Apple{tuple_delimiter}TSMC{tuple_delimiter}supplier relationship, supply chain challenges{tuple_delimiter}Q3 2024: Apple faces supply chain challenges with key supplier TSMC during the third quarter.){record_delimiter}
(relationship{tuple_delimiter}Apple{tuple_delimiter}Greater China Revenue{tuple_delimiter}regional performance, market presence{tuple_delimiter}Q3 2024: Apple maintained strong performance in Greater China with $14.7 billion in quarterly revenue.){record_delimiter}
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
3. **Temporal Information:** When descriptions contain temporal information (dates, quarters, years, time periods), preserve and organize this information chronologically. Present the most recent information prominently while maintaining historical context.
4. **Conflict:** In case of conflicting or inconsistent descriptions, determine if they originate from multiple, distinct entities or relationships that share the same name. If so, summarize each entity or relationship separately and then consolidate all summaries.
5. **Style:** The output must be written from an objective, third-person perspective.
6. **Length:** Maintain depth and completeness while ensuring the summary's length not exceed {summary_length} tokens.
7. **Language:** The entire output must be written in {language}.

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
