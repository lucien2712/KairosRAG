# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LightRAG is a Python-based Retrieval-Augmented Generation system that combines knowledge graphs with vector databases for enhanced document understanding and querying. The system provides both a core library and a server API with web UI for production use.

## Installation & Setup

### Core Installation
```bash
# Install from source (recommended)
pip install -e .

# Install with API support
pip install -e ".[api]"
```

### Server Setup
```bash
# Copy configuration
cp env.example .env

# Start server
lightrag-server

# Or with Docker
docker compose up
```

## Core Architecture

### Main Components

- **lightrag/lightrag.py**: Core LightRAG class and initialization logic
- **lightrag/base.py**: Base storage interfaces and abstract classes
- **lightrag/kg/**: Knowledge graph storage implementations (NetworkX, Neo4j, PostgreSQL AGE)
- **lightrag/llm/**: LLM integrations (OpenAI, Ollama, HuggingFace, LlamaIndex)
- **lightrag/api/**: Server implementation with FastAPI and web UI
- **lightrag/types.py**: Core data structures and type definitions

### Storage Layer

LightRAG uses 4 storage types:
- **KV_STORAGE**: LLM cache, text chunks, document info (JSON, PostgreSQL, Redis, MongoDB)
- **VECTOR_STORAGE**: Entity/relation/chunk embeddings (NanoVectorDB, PostgreSQL, Milvus, Faiss, Qdrant, MongoDB)
- **GRAPH_STORAGE**: Knowledge graph (NetworkX, Neo4j, PostgreSQL AGE, Memgraph)
- **DOC_STATUS_STORAGE**: Document processing status (JSON, PostgreSQL, MongoDB)

### Processing Pipeline

1. **Document Ingestion**: Text chunking and preprocessing
2. **Entity Extraction**: LLM-based entity and relation extraction from chunks
3. **Graph Construction**: Building knowledge graph with entities and relations
4. **Vector Indexing**: Creating embeddings for entities, relations, and chunks
5. **Query Processing**: Hybrid retrieval combining graph traversal, vector search, and multi-hop expansion

## Key Development Commands

### Code Quality
```bash
# Format code
ruff format .

# Lint code
ruff check . --fix

# Type checking (not configured, but can be added)
# mypy lightrag/
```

### Testing
```bash
# Run basic tests
python -m pytest tests/

# Test specific storage backends
python tests/test_graph_storage.py
python tests/test_lightrag_ollama_chat.py
```

### Examples & Demos
```bash
# Basic OpenAI demo
python examples/lightrag_openai_demo.py

# Ollama integration
python examples/lightrag_ollama_demo.py

# Custom knowledge graph insertion
python examples/insert_custom_kg.py

# Graph visualization
python examples/graph_visual_with_html.py
```

## Multi-hop Retrieval

LightRAG supports multi-hop graph expansion to discover deeper relationships beyond direct keyword matches.

### Multi-hop Query Parameters

```python
from lightrag import QueryParam

# Basic usage
response = await rag.aquery(
    "Apple iPhone revenue and manufacturing relationships",
    param=QueryParam(
        mode="hybrid",  # local/global/hybrid/mix
        max_hop=2,      # Number of hops (0=no expansion, 1=1 hop, 2=2 hops)
        max_neighbors=10, # Max neighbors per hop
        multi_hop_relevance_threshold=0.3, # Relevance threshold (0.0-1.0)
    )
)
```

### Multi-hop Parameters Explanation

- **max_hop** (int, default: 1): Number of expansion hops
  - `0`: Only initial search, no expansion
  - `1`: Expand to direct neighbors (1-hop)  
  - `2`: Expand to neighbors of neighbors (2-hop)
  - Higher values discover deeper relationships but increase complexity

- **max_neighbors** (int, default: 10): Maximum neighbors selected per hop
  - Controls expansion breadth at each hop
  - Higher values find more relationships but use more tokens

- **multi_hop_relevance_threshold** (float, default: 0.3): Minimum relevance score
  - Recommended range: 0.05-0.5
  - Lower values (0.05-0.1) for broad exploration
  - Higher values (0.3-0.5) for focused, high-relevance expansion

### 4-Dimensional Scoring System

Multi-hop expansion uses 4-dimensional relevance scoring:

1. **Low-level Similarity (40%)**: Entity-keyword semantic similarity using embeddings
2. **High-level Similarity (40%)**: Relationship-keyword semantic similarity using embeddings  
3. **Node Importance (10%)**: Based on node degree/rank in knowledge graph
4. **Distance Decay (10%)**: Penalty factor that decreases with hop distance (0.8^hop)

### Embedding Format Consistency

Multi-hop expansion maintains consistency with initial search:
- **Entities**: `"{entity_name} {description}"`
- **Relationships**: `"{keywords}\t{src_id}\n{tgt_id}\n{description}"`

### Use Cases

Multi-hop retrieval is particularly effective for:
- **Complex analytical queries**: Cross-quarter trend analysis, impact assessment
- **Relationship discovery**: Finding indirect connections between entities
- **Time-sensitive documents**: Earnings calls, financial reports with temporal relationships
- **Supply chain analysis**: Discovering manufacturing partnerships and dependencies

### Performance Considerations

- Multi-hop expansion increases query latency and token usage
- Use lower `max_hop` values (1-2) for most applications
- Adjust `multi_hop_relevance_threshold` based on result quality needs:
  - 0.05-0.1 for broad exploration (finds more relationships)
  - 0.3-0.5 for focused analysis (higher quality relationships)
- Monitor token limits when using high `max_neighbors` values
- Start with `max_hop=1` and increase only if needed for deeper analysis

## Timestamp Support

LightRAG supports timestamp metadata for time-sensitive document analysis.

### Adding Timestamps During Insertion

```python
# Insert documents with timestamps
await rag.ainsert(
    input=[
        "Apple Q3 2024 Earnings Call - July 31, 2024. iPhone revenue reached $39.3 billion...",
        "Apple Q2 2024 Earnings Call - April 30, 2024. iPhone revenue was $45.9 billion..."
    ],
    timestamps=[
        "2024-07-31 Q3 2024",
        "2024-04-30 Q2 2024"
    ],
    file_paths=[
        "apple_q3_2024_earnings.txt",
        "apple_q2_2024_earnings.txt"
    ]
)
```

### Timestamp Integration

Timestamps are automatically integrated into entity and relationship descriptions during extraction:
- Entity descriptions include temporal context: `"Q3 2024: Apple is a technology company..."`
- Relationship descriptions include time references: `"Q3 2024: Apple's iPhone revenue reached $39.3B..."`
- Enables time-aware querying and cross-temporal analysis

### Time-Aware Queries

```python
# Query with temporal context
response = await rag.aquery(
    "Compare Apple's iPhone revenue performance across 2024 quarters",
    param=QueryParam(mode="hybrid", max_hop=2)
)

# Specific time period queries
response = await rag.aquery(
    "What were Apple's key financial highlights in Q3 2024?",
    param=QueryParam(mode="hybrid")
)
```

### Benefits

- **Temporal Analysis**: Compare metrics across time periods
- **Trend Discovery**: Identify patterns and changes over time
- **Context Preservation**: Maintain time-sensitive information in knowledge graph
- **Cross-Quarter Insights**: Multi-hop retrieval can discover temporal relationships

## Configuration

### Environment Variables

Key environment variables (see env.example):
- `OPENAI_API_KEY`: OpenAI API key
- `WORKING_DIR`: Data storage directory
- `TOP_K`: Number of entities/relations to retrieve
- `MAX_ASYNC`: Maximum concurrent LLM requests
- `COSINE_THRESHOLD`: Vector similarity threshold

### LLM and Embedding Setup

```python
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

rag = LightRAG(
    working_dir="./storage",
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embed,
)

# CRITICAL: Always initialize before use
await rag.initialize_storages()
await initialize_pipeline_status()
```

## API Development

The API server (`lightrag/api/`) provides:
- Document upload and indexing endpoints
- Query interface with multiple modes (local, global, hybrid, mix)
- Knowledge graph visualization
- User management and authentication
- Ollama-compatible chat interface

Start development server:
```bash
uvicorn lightrag.api.lightrag_server:app --reload --port 8000
```

## Important Notes

### Initialization Requirements
LightRAG requires explicit initialization:
1. `await rag.initialize_storages()` - Initialize storage backends
2. `await initialize_pipeline_status()` - Initialize processing pipeline

Missing either will cause `AttributeError: __aenter__` or `KeyError: 'history_messages'` errors.

### Storage Backends
- Production deployments should use PostgreSQL, Neo4j, or other enterprise storage
- Default file-based storage is suitable for development only
- Embedding model must remain consistent across indexing and querying
- When changing embedding models, clear existing data directory

### Performance Considerations
- Use at least 32B parameter LLM with 32KB+ context length
- Configure appropriate `max_parallel_insert` (default: 2, max recommended: 10)
- Enable LLM caching for better performance during development
- Consider reranker models for improved retrieval quality

## File Structure Patterns

- Core classes in `lightrag/` root
- Storage implementations in `lightrag/<storage_type>/`
- LLM integrations in `lightrag/llm/`
- API components in `lightrag/api/`
- Tools and utilities in `lightrag/tools/`
- Examples for different use cases in `examples/`
- Community-contributed examples in `examples/unofficial-sample/`