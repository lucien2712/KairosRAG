# KairosRAG

**KairosRAG** is an enhanced framework built on top of [LightRAG](https://github.com/HKUDS/LightRAG), introducing **time-awareness, three-perspective retrieval expansion, adaptive schema induction, and agentic entity management** for domain-specific and temporally-sensitive knowledge graphs.

## 🚀 Key Contributions

### ⏳ **Automatic Timestamp Integration**

* **Problem**: Time-sensitive documents lose context when temporal information is inconsistent.
* **Solution**: KairosRAG automatically prefixes user-provided timestamps to entity and relation descriptions, ensuring consistent temporal alignment without relying on LLM memory.

**Mathematical Framework:**

$$Entity'(e, t) = T(t) \oplus Entity(e)$$
$$Relation'(r, t) = T(t) \oplus Relation(r)$$

where $T(t)$ is the timestamp prefix function, $\oplus$ denotes string concatenation

### 🔍 **Three-Perspective Expansion Architecture**

* **Problem**: LightRAG primarily supports one-hop expansion, limiting indirect but meaningful reasoning.
* **Solution**: KairosRAG employs a **three-perspective expansion approach** that processes queries from complementary angles: semantic multi-hop traversal, global importance ranking via Query-aware Personalized PageRank, and structural similarity through Adaptive FastRP embeddings with edge weights.
* *Currently implemented for the NanoVectorDB backend.*

**Architecture Overview:**

The system processes each query through three independent perspectives and merges results:

1. **Multi-hop Semantic Expansion**: Traditional graph traversal based on query-entity similarity
2. **Query-aware Global Importance Analysis**: Enhanced PageRank with query-sensitive seed weighting and edge reweighting  
3. **Adaptive Structural Analysis**: Enhanced FastRP with edge weights for structure-aware embeddings

**Mathematical Framework:**

$$R_{final} = \text{merge}(R_{multihop}, R_{ppr}, R_{fastrp})$$

**Multi-hop Expansion:**
$$S_{multihop}(e, q, h) = \alpha \cdot sim_{entity}(e, q) + \beta \cdot sim_{relation}(r, q) + \gamma \cdot \delta^h$$

where $\delta = 0.8$ (distance decay), $h$ = hop count

**Query-aware Personalized PageRank:**
$$PPR_{query}(v; S, W, E) = (1-d) \cdot p_{S,W}(v) + d \cdot \sum_{u \to v} \frac{PPR_{query}(u; S, W, E) \cdot e_{reweight}(u,v)}{|out(u)|}$$

where:
- $S$ = seed entities, $W$ = query-aware seed weights (Phase 1)
- $E$ = query-aware edge weights (Phase 2), $d = 0.85$ (damping factor)
- $p_{S,W}(v)$ = personalization vector with weighted seeds
- $e_{reweight}(u,v)$ = temporarily adjusted edge weights based on query-relation similarity

**Adaptive FastRP with Edge Weights:**
$$X = \sum_{k=0}^{K} w_k \cdot D^r \cdot S^k \cdot R$$

where $S = D^{-1/2}A_{weighted}D^{-1/2}$, $A_{weighted}$ uses original Graph edge weights

**Edge Weight Integration:**
The Adaptive FastRP algorithm uses edge weights from the original Graph knowledge graph construction, preserving relationship importance established during insertion time.

### 🧠 **Adaptive Entity Type Discovery**

* **Problem**: Predefined static schemas (e.g., Person, Organization, Category) often misclassify domain-specific metrics (e.g., EPS, Operating Margin).
* **Solution**: KairosRAG dynamically suggests and augments **domain-specific entity types** during extraction, improving recognition accuracy and retrieval relevance.

### ⚡ **Agentic Entity Canonicalization**

* **Problem**: Duplicate entities scatter knowledge and reduce retrieval precision, especially when Adaptive FastRP/PPR compute embeddings before entity merging.
* **Solution**: KairosRAG combines **vector similarity pre-filtering** with **LLM-based reasoning** for intelligent entity deduplication. The system now supports **automatic merging during insertion** to ensure Adaptive FastRP/PPR operate on cleaned entity graphs.
* *Currently implemented for the NanoVectorDB backend.*

**Mathematical Framework:**

**Vector Similarity Pre-filtering:**
$$C = \{(e_i, e_j) | cosine(emb(e_i), emb(e_j)) > \theta, i < j\}$$

where $\theta$ = similarity threshold (e.g., 0.8)

**LLM Decision Function:**
$$M(e_i, e_j) = \mathbb{I}[LLM_{confidence}("Should\ merge?", context(e_i, e_j)) > 0.95]$$

**Merge Operation:**
$$entity_{final} = Merge(e_i, e_j) \text{ if } M(e_i, e_j) = \text{True}$$

### 📊 **Table-Aware Document Processing**

* **Problem**: HTML tables in documents get fragmented during traditional chunking, causing loss of structural relationships and data integrity that are crucial for understanding financial reports, research papers, and structured documents.
* **Solution**: KairosRAG implements intelligent **table-aware chunking** that preserves complete `<table>...</table>` structures within single chunks, automatically includes surrounding context, and employs **table-enhanced entity extraction** with specialized prompts for tabular data.

---

## 🛠️ Installation

```bash
git clone https://github.com/lucien2712/KairosRAG.git
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
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()
    
    # Adaptive entity type discovery
    rag.entity_type_aug("input_folder/")

    # Insert with automatic timestamp integration and agentic merging
    await rag.insert(
        input=[document],
        timestamps=["2024-Q3"],  # Auto-prefixed to descriptions
        file_paths=["apple_q3_2024.pdf"],
        agentic_merging=True,           # Enable automatic entity merging
        agentic_merging_threshold=0.8   # Similarity threshold for merging
    )
    
    # Query with three-perspective expansion
    response = rag.query(
        "Apple iPhone revenue trends and supplier relationships",
        param=QueryParam(
            mode="hybrid", 
            max_hop=2,              # Multi-hop traversal depth
            top_neighbors=30,       # Max neighbor per node
            top_ppr_nodes=5,        # Top PageRank entities
            top_fastrp_nodes=5      # Top Adaptive FastRP structural entities
        )
    )

if __name__ == "__main__":
    asyncio.run(main())
```
---

## 🏗️ Workflow

```bash
# 1. Discover entity types for your domain
rag.entity_type_aug("input_folder/")

# 2. Insert documents with timestamps (tables processed automatically)
rag.insert(docs_with_tables, timestamps=["2024-Q3"], file_paths=["report.pdf"], agentic_merging=True, agentic_merging_threshold=0.8)
# Note: HTML tables are automatically detected and processed with context preservation

# 3. Query with three-perspective expansion
response = rag.query(query, param=QueryParam(
    max_hop=2,           # Multi-hop traversal
    top_neighbors=30,    # Top neighbor per node
    top_ppr_nodes=5,     # Personalized PageRank entities
    top_fastrp_nodes=5   # Adaptive FastRP structural entities
))
```

---

## 🤝 Contributing

KairosRAG extends **LightRAG** with:

1. **Automatic Timestamp Integration** – Consistent temporal metadata injection with mathematical prefix functions.
2. **Three-Perspective Expansion** – Multi-hop traversal, Query-aware Personalized PageRank, and Adaptive FastRP structural similarity providing complementary retrieval perspectives.
3. **Adaptive Entity Type Discovery** – Dynamic schema induction for domain-specific contexts.
4. **Agentic Entity Canonicalization** – Hybrid vector+LLM pipeline with cosine similarity pre-filtering and confidence thresholding.
5. **Table-Aware Document Processing** – Intelligent table detection, structure preservation, and context-aware chunking with specialized entity extraction for tabular data.

---

## 🙏 Acknowledgments

KairosRAG builds upon [LightRAG](https://github.com/HKUDS/LightRAG) by the HKUDS team (MIT License).
We extend it with **time-awareness, three-perspective expansion architecture, adaptive schema induction, and agentic entity management** using rigorous mathematical frameworks.
