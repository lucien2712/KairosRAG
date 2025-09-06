# KairosRAG

**KairosRAG** is an enhanced framework built on top of [LightRAG](https://github.com/HKUDS/LightRAG), introducing **time-awareness, multi-hop retrieval optimization, adaptive schema induction, and agentic entity management** for domain-specific and temporally-sensitive knowledge graphs.

## 🚀 Key Contributions

### ⏳ **Automatic Timestamp Integration**

* **Problem**: Time-sensitive documents lose context when temporal information is inconsistent.
* **Solution**: KairosRAG automatically prefixes user-provided timestamps to entity and relation descriptions, ensuring consistent temporal alignment without relying on LLM memory.

### 🔍 **Embedding-Guided Multi-hop Retrieval**

* **Problem**: LightRAG primarily supports one-hop expansion, limiting indirect but meaningful reasoning.
* **Solution**: KairosRAG explores candidate paths by jointly considering **entity similarity, relation similarity, and hop distance decay**, enabling richer multi-hop reasoning.
* *Currently implemented for the NanoVectorDB backend.*

### 🧠 **Adaptive Entity Type Discovery**

* **Problem**: Predefined static schemas (e.g., Person, Organization, Category) often misclassify domain-specific metrics (e.g., EPS, Operating Margin).
* **Solution**: KairosRAG dynamically suggests and augments **domain-specific entity types** during extraction, improving recognition accuracy and retrieval relevance.

### ⚡ **Agentic Entity Merging**

* **Problem**: Duplicate entities scatter knowledge and reduce retrieval precision.
* **Solution**: KairosRAG combines **vector similarity pre-filtering** with **LLM-based reasoning** for intelligent entity deduplication.
* *Currently implemented for the NanoVectorDB backend.*

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/KairosRAG.git
cd KairosRAG
pip install -e .
```

---

## 🚀 Quick Start

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    # Initialize KairosRAG
    rag = LightRAG(
        working_dir="./data",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()
    
    # Insert with automatic timestamp integration
    await rag.insert(
        input=["Apple Q3 2024 earnings: iPhone revenue $39.3B, down 1.5% YoY..."],
        timestamps=["2024-Q3"],  # Auto-prefixed to descriptions
        file_paths=["apple_q3_2024.pdf"]
    )
    
    # Query with enhanced multi-hop
    response = rag.query(
        "Apple iPhone revenue trends and supplier relationships",
        param=QueryParam(mode="hybrid", max_hop=2)
    )
    
    # Run agentic entity merging (NanoVectorDB only)
    result = await rag.aagentic_merging(threshold=0.8)
    # Example: "Tim Cook" merged into "Timothy D. Cook" (similarity: 0.903)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🏗️ Workflow

```bash
# 1. Discover entity types for your domain
python entity_type_augmentation.py

# 2. Insert documents with timestamps
await rag.insert(docs, timestamps=["2024-Q3"], file_paths=["sample.pdf"])

# 3. Run agentic entity merging (NanoVectorDB only)
result = await rag.agentic_merging(threshold=0.8)

# 4. Query with enhanced multi-hop retrieval
response = rag.query(query, param=QueryParam(max_hop=2))
```

---

## 🤝 Contributing

KairosRAG extends **LightRAG** with:

1. **Automatic Timestamp Integration** – Consistent temporal metadata injection.
2. **Embedding-Guided Multi-hop Retrieval** – Path scoring with relation/entity similarity and hop decay.
3. **Adaptive Entity Type Discovery** – Dynamic schema induction for domain-specific contexts.
4. **Agentic Entity Merging** – Hybrid vector+LLM pipeline for robust deduplication.

---

## 🙏 Acknowledgments

KairosRAG builds upon [LightRAG](https://github.com/HKUDS/LightRAG) by the HKUDS team (MIT License).
We extend it with **time-awareness, adaptive schema induction, embedding-guided multi-hop reasoning, and agentic entity management**.
