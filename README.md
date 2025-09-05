# KairosRAG

Enhanced version of [LightRAG](https://github.com/HKUDS/LightRAG) with automatic timestamp integration, improved multi-hop retrieval, and adaptive entity type discovery.

## 🚀 Key Enhancements

### ✨ **Automatic Timestamp Integration**
**Problem**: Time-sensitive documents lose context when temporal information is inconsistent. Traditional RAG relies on LLMs to remember adding timestamps.

**Solution**: System automatically prefixes user-provided timestamps to all entity/relation descriptions, ensuring consistent temporal context.

### 🔍 **Enhanced Multi-hop Retrieval** 
**Problem**: The original LightRAG primarily supports one-hop expansion, which limits the ability to capture indirect but meaningful connections across documents. 

**Solution**: Our system introduces embedding-guided multi-hop retrieval, where candidate paths are explored by jointly considering relation similarity, entity similarity, and graph distance. *Note: Currently implemented for NanoVectorDB backend only.*

### 🧠 **Adaptive Entity Type Discovery**
**Problem**: Predefined and incomplete entity type schemas often fail in domain-specific contexts. For example, when analyzing financial reports, metrics such as EPS or operating margin may be incorrectly forced into generic categories (e.g., "category"), reducing entity extraction accuracy and retrieval precision.

**Solution**: The system dynamically analyzes document content to discover and suggest new domain-specific entity types, extending beyond the static schema. This adaptive process improves entity recognition accuracy and ensures domain-relevant knowledge is preserved in the graph.

### ⚡ **Agentic Entity Merging**
**Problem**: Duplicate entities scatter knowledge and reduce retrieval precision.

**Innovation**: Combines vector similarity pre-filtering with LLM decision making for intelligent entity deduplication. *Note: Currently implemented for NanoVectorDB backend only.*

## 🛠️ Installation

```bash
git clone https://github.com/your-username/KairosRAG.git
cd KairosRAG
pip install -e .
```

## 🚀 Quick Start

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    # Initialize
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
    
    # Run agentic entity merging (requires NanoVectorDB)
    result = await rag.aagentic_merging(threshold=0.8)
    # Example: Merged "Timothy D. Cook" ← "Tim Cook" (similarity: 0.903)

if __name__ == "__main__":
    asyncio.run(main())
```

## 🏗️ Workflow

```python
# 1. Discover entity types for your domain
python entity_type_augmentation.py

# 2. Insert documents with timestamps
await rag.insert(documents, timestamps=["2024-Q3"], file_path)

# 3. Run agentic entity merging (requires NanoVectorDB)
result = await rag.agentic_merging(threshold=0.8)

# 4. Query with enhanced retrieval
response = rag.query(query, param=QueryParam(max_hop=2))
```

## 🤝 Contributing

This project extends **LightRAG** with:

1. **Automatic Timestamp Integration** – Adds consistent temporal metadata without LLM dependence.
2. **Enhanced Multi-hop Retrieval** – Embedding-guided path search with provenance, chunk balancing, and vector caching.
3. **Adaptive Entity Type Discovery** – Dynamically suggests domain-specific entity types beyond static schemas.
4. **Agentic Entity Merging** – Vector similarity + LLM merging for cleaner, deduplicated knowledge graphs.


## 🙏 Acknowledgments

Built upon [LightRAG](https://github.com/HKUDS/LightRAG) by the HKUDS team. Enhanced with automatic timestamp processing, performance-optimized multi-hop retrieval, adaptive entity type discovery, and intelligent entity merging.