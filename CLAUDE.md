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
5. **Query Processing**: Hybrid retrieval combining graph traversal and vector search

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