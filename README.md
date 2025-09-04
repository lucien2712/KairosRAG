# LightRAG Enhanced

Enhanced version of [LightRAG](https://github.com/HKUDS/LightRAG) with automatic timestamp integration, improved multi-hop retrieval, and adaptive entity type discovery.

## 🚀 Key Enhancements

### ✨ **Automatic Timestamp Integration**
**Problem**: Time-sensitive documents lose context when temporal information is inconsistent. Traditional RAG relies on LLMs to remember adding timestamps.

**Solution**: System automatically prefixes user-provided timestamps to all entity/relation descriptions, ensuring consistent temporal context.

### 🔍 **Enhanced Multi-hop Retrieval** 
**Problem**: Multi-hop expansion loses metadata and source attribution, making insights untracenable.

**Solution**: Maintains complete provenance for expanded entities/relations and includes source chunks with round-robin balancing.

### 🧠 **Adaptive Entity Type Discovery**
**Problem**: Static schemas misclassify domain-specific entities (financial metrics → "category").

**Innovation**: `entity_type_augmentation.py` automatically discovers and suggests domain-specific entity types based on document content.

## 🛠️ Installation

```bash
git clone https://github.com/your-username/LightRAG-Enhanced.git
cd LightRAG-Enhanced
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

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔧 Entity Type Augmentation

Automatically discover domain-specific entity types to improve retrieval accuracy:

```bash
python entity_type_augmentation.py
```
Domain-specific types enable precise retrieval
```json
{"iPhone Revenue": "financial_metric", "Q3 2024": "temporal_range", "Tim Cook": "person"}
```

Now queries like *"find financial metrics"* or *"temporal analysis"* can accurately locate relevant entities instead of generic "category" results.

## 📋 Enhanced Features

### Context Generation
All data includes complete metadata:

```json
{
  "entities": [{
    "entity": "Apple",
    "description": "2024-Q3: Apple technology company...",  // Auto-prefixed
    "created_at": "2024-09-04 15:30:45",
    "file_path": "earnings.pdf"
  }],
  "chunks": [{
    "content": "Apple reported...",
    "The content is occured at": "2024-Q3"  // Descriptive field
  }]
}
```

### Multi-hop Expansion
- **Seed Entities**: Initial query results
- **Expanded Entities**: Multi-hop discoveries with complete metadata
- **Round-Robin Merging**: Fair distribution of initial vs expanded results

## 🏗️ Workflow

```python
# 1. Discover entity types for your domain
python entity_type_augmentation.py

# 2. Insert documents with timestamps
await rag.insert(documents, timestamps=["2024-Q3"])

# 3. Query with enhanced retrieval
response = rag.query(query, param=QueryParam(max_hop=2))
```

## 🤝 Contributing

This project enhances LightRAG with:
1. **Automatic Timestamp Integration**: Eliminates LLM dependency
2. **Enhanced Multi-hop Retrieval**: Complete metadata preservation  
3. **Adaptive Entity Type Discovery**: Content-driven schema evolution

## 🙏 Acknowledgments

Built upon [LightRAG](https://github.com/HKUDS/LightRAG) by the HKUDS team. Enhanced with automatic timestamp processing, improved multi-hop retrieval, and adaptive entity type discovery.