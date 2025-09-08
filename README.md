# KairosRAG

**KairosRAG** is an enhanced framework built on top of [LightRAG](https://github.com/HKUDS/LightRAG), introducing **time-awareness, multi-hop retrieval optimization, adaptive schema induction, and agentic entity management** for domain-specific and temporally-sensitive knowledge graphs.

## 🚀 Key Contributions

### ⏳ **Automatic Timestamp Integration**

* **Problem**: Time-sensitive documents lose context when temporal information is inconsistent.
* **Solution**: KairosRAG automatically prefixes user-provided timestamps to entity and relation descriptions, ensuring consistent temporal alignment without relying on LLM memory.

**Mathematical Framework:**

$$Entity'(e, t) = T(t) \oplus Entity(e)$$
$$Relation'(r, t) = T(t) \oplus Relation(r)$$

where $T(t)$ is the timestamp prefix function, $\oplus$ denotes string concatenation

### 🔍 **Three-Way Parallel Expansion Architecture**

* **Problem**: LightRAG primarily supports one-hop expansion, limiting indirect but meaningful reasoning.
* **Solution**: KairosRAG employs a **three-way parallel expansion** combining multi-hop traversal, Personalized PageRank, and FastRP structural similarity for comprehensive retrieval.
* *Currently implemented for the NanoVectorDB backend.*

**Mathematical Framework:**

$$R_{final} = R_{multihop} \cup R_{ppr} \cup R_{fastrp}$$

**Multi-hop Expansion:**
$$S_{multihop}(e, q, h) = \alpha \cdot sim_{entity}(e, q) + \beta \cdot sim_{relation}(r, q) + \gamma \cdot \delta^h$$

where $\delta = 0.8$ (distance decay), $h$ = hop count

**Personalized PageRank:**
$$PPR(v; S) = (1-d) \cdot p_S(v) + d \cdot \sum_{u \to v} \frac{PPR(u; S)}{|out(u)|}$$

where $S$ = seed entities, $d = 0.85$ (damping factor)

**FastRP Structural Similarity:**
$$FastRP(G) = \Phi(A^k \cdot R)$$

where $A$ = adjacency matrix, $R$ = random projection, $k$ = iteration count

### 🧠 **Adaptive Entity Type Discovery**

* **Problem**: Predefined static schemas (e.g., Person, Organization, Category) often misclassify domain-specific metrics (e.g., EPS, Operating Margin).
* **Solution**: KairosRAG dynamically suggests and augments **domain-specific entity types** during extraction, improving recognition accuracy and retrieval relevance.

### ⚡ **Agentic Entity Canonicalization**

* **Problem**: Duplicate entities scatter knowledge and reduce retrieval precision.
* **Solution**: KairosRAG combines **vector similarity pre-filtering** with **LLM-based reasoning** for intelligent entity deduplication.
* *Currently implemented for the NanoVectorDB backend.*

**Mathematical Framework:**

**Vector Similarity Pre-filtering:**
$$C = \{(e_i, e_j) | cosine(emb(e_i), emb(e_j)) > \theta, i < j\}$$

where $\theta$ = similarity threshold (e.g., 0.8)

**LLM Decision Function:**
$$M(e_i, e_j) = \mathbb{I}[LLM_{confidence}("Should\ merge?", context(e_i, e_j)) > 0.95]$$

**Merge Operation:**
$$entity_{final} = Merge(e_i, e_j) \text{ if } M(e_i, e_j) = \text{True}$$

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
    
    # Query with three-way parallel expansion
    response = rag.query(
        "Apple iPhone revenue trends and supplier relationships",
        param=QueryParam(
            mode="hybrid", 
            max_hop=2,              # Multi-hop traversal depth
            max_neighbors=30,       # Max neighbor per node
            top_ppr_nodes=5,        # Top PageRank entities
            top_fastrp_nodes=5      # Top FastRP structural entities
        )
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
rag.insert(docs, timestamps=["2024-Q3"], file_paths=["sample.pdf"])

# 3. Run agentic entity merging (NanoVectorDB only)
result = await rag.agentic_merging(threshold=0.8)

# 4. Query with three-way parallel expansion
response = rag.query(query, param=QueryParam(
    max_hop=2,           # Multi-hop traversal
    max_neighbors=30,    # Max neighbor per node
    top_ppr_nodes=5,     # Personalized PageRank entities
    top_fastrp_nodes=5   # FastRP structural entities
))
```

---

## 🤝 Contributing

KairosRAG extends **LightRAG** with:

1. **Automatic Timestamp Integration** – Consistent temporal metadata injection with mathematical prefix functions.
2. **Three-Way Parallel Expansion** – Multi-hop traversal, Personalized PageRank, and FastRP structural similarity operating independently.
3. **Adaptive Entity Type Discovery** – Dynamic schema induction for domain-specific contexts.
4. **Agentic Entity Canonicalization** – Hybrid vector+LLM pipeline with cosine similarity pre-filtering and confidence thresholding.

---

## 🙏 Acknowledgments

KairosRAG builds upon [LightRAG](https://github.com/HKUDS/LightRAG) by the HKUDS team (MIT License).
We extend it with **time-awareness, three-way parallel expansion architecture, adaptive schema induction, and agentic entity management** using rigorous mathematical frameworks.
