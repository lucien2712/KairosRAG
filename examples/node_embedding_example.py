"""
Example demonstrating FastRP + Personalized PageRank enhanced retrieval in LightRAG.

This example shows how to:
1. Enable node embedding enhancement during LightRAG initialization
2. Insert documents with enhanced embedding computation
3. Query with enhanced embeddings for improved retrieval

Requirements:
    pip install lightrag-hku[gnn]  # Installs scikit-learn for FastRP
"""

import asyncio
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# Set up API keys
OPENAI_API_KEY = "your-openai-api-key-here"
WORKING_DIR = "./enhanced_rag_storage"

# Sample documents
documents = [
    """Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
    Tim Cook serves as the CEO of Apple. The company is known for innovative products like iPhone, iPad, and Mac computers.
    Apple reported strong iPhone revenue in Q3 2024.""",
    
    """Tesla, Inc. is an American electric vehicle and clean energy company based in Austin, Texas. 
    Elon Musk is the CEO of Tesla. Tesla is known for electric vehicles, energy storage, and solar panels.
    Tesla has partnerships with various suppliers for battery manufacturing.""",
    
    """Google LLC is an American multinational technology corporation focusing on online advertising, 
    search engine technology, cloud computing, computer software, quantum computing, e-commerce, 
    artificial intelligence, and consumer electronics. Sundar Pichai is the CEO of Google."""
]

async def main():
    print("🚀 LightRAG Node Embedding Enhancement Example")
    print("=" * 50)
    
    # Initialize LightRAG with node embedding enhancement enabled
    print("📋 Initializing LightRAG with FastRP + Personalized PageRank...")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
        enable_node_embedding=True,  # 🔥 Enable FastRP + PageRank enhancement
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
    )
    
    # Initialize storage
    await rag.initialize_storages()
    
    print("✅ LightRAG initialized with enhanced embeddings enabled")
    
    # Insert documents (this will compute FastRP + PageRank embeddings)
    print("\n📄 Inserting documents with enhanced embedding computation...")
    
    for i, doc in enumerate(documents, 1):
        print(f"   Inserting document {i}/3...")
        await rag.ainsert(doc)
    
    print("✅ All documents inserted with enhanced embeddings computed")
    
    # Query examples with and without enhanced embeddings
    queries = [
        "Who are the CEOs of major tech companies?",
        "Tell me about Apple's financial performance",
        "What partnerships does Tesla have?",
        "Compare the business models of Apple, Tesla, and Google"
    ]
    
    print("\n🔍 Testing Enhanced vs Regular Retrieval")
    print("=" * 50)
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        print("-" * 40)
        
        # Regular retrieval
        print("📊 Regular retrieval:")
        regular_response = await rag.aquery(
            query, 
            QueryParam(
                mode="hybrid", 
                use_node_embedding=False  # Regular retrieval
            )
        )
        print(f"   {regular_response[:150]}...")
        
        # Enhanced retrieval with semantic expansion + structural analysis (balanced mode)
        print("\n🧠 Enhanced retrieval (Semantic Expansion + Structural Analysis - Balanced):")
        enhanced_response = await rag.aquery(
            query, 
            QueryParam(
                mode="hybrid", 
                use_node_embedding=True,  # 🔥 Enable enhanced retrieval
                expansion_strategy="balanced",  # 50% semantic expansion, 50% structural analysis
                pagerank_importance_threshold=0.005,  # 🔧 Configurable PageRank threshold
                fastrp_similarity_threshold=0.1     # 🔧 Configurable FastRP threshold
            )
        )
        print(f"   {enhanced_response[:150]}...")
        
        # Enhanced retrieval with semantic-focused mode
        print("\n📚 Enhanced retrieval (Semantic Focused):")
        semantic_focused_response = await rag.aquery(
            query,
            QueryParam(
                mode="hybrid",
                use_node_embedding=True,
                expansion_strategy="semantic_focused"  # 70% semantic expansion, 30% structural analysis
            )
        )
        print(f"   {semantic_focused_response[:150]}...")
        
        # Enhanced retrieval with structure-focused mode  
        print("\n🔗 Enhanced retrieval (Structure Focused):")
        structure_focused_response = await rag.aquery(
            query,
            QueryParam(
                mode="hybrid", 
                use_node_embedding=True,
                expansion_strategy="structure_focused"  # 30% semantic expansion, 70% structural analysis
            )
        )
        print(f"   {structure_focused_response[:150]}...")
    
    print("\n🎯 Personalized PageRank Example")
    print("=" * 40)
    
    # Example of using personalized PageRank for specific entity-focused queries
    if hasattr(rag, 'node_embedding') and rag.node_embedding:
        try:
            # Compute personalized PageRank scores for Apple-focused analysis
            seed_entities = ["Apple Inc.", "Tim Cook", "iPhone"]
            ppr_scores = rag.node_embedding.compute_personalized_pagerank(seed_entities)
            
            if ppr_scores:
                print("📈 Personalized PageRank scores (Apple-focused):")
                sorted_scores = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
                for entity, score in sorted_scores[:10]:  # Top 10
                    print(f"   {entity}: {score:.4f}")
            
        except Exception as e:
            print(f"   ⚠️  Note: Personalized PageRank computation requires graph data: {e}")
    
    print("\n✨ Example completed!")
    print("\n📚 Key Benefits of Semantic Expansion + Structural Analysis:")
    print("   • Semantic expansion: Multi-hop similarity-based exploration")  
    print("   • Structural analysis: Global graph importance assessment")
    print("   • Complementary approaches: Local relevance + global significance")
    print("   • Configurable resource allocation between expansion strategies")
    print("   • Personalized PageRank for query-focused entity ranking")
    
    print(f"\n💾 Enhanced embeddings stored in: {WORKING_DIR}")


def demo_configuration():
    """Show different configuration options for node embedding."""
    print("\n⚙️  Configuration Options:")
    print("=" * 30)
    
    # Example 1: Basic enablement
    print("1️⃣  Basic Enhancement:")
    print("""
    rag = LightRAG(
        working_dir="./storage",
        llm_model_func=your_llm_function,
        embedding_func=your_embedding_function,
        enable_node_embedding=True,  # Enable FastRP + PageRank
    )
    """)
    
    # Example 2: Query-time control
    print("2️⃣  Query-time Control:")
    print("""
    # Use enhanced retrieval for complex queries
    response = await rag.aquery(
        "Complex multi-entity question", 
        QueryParam(
            mode="hybrid",
            use_node_embedding=True,  # Enable enhanced retrieval
            expansion_strategy="balanced",  # Balance semantic + structural
            top_k=10
        )
    )
    
    # Use regular retrieval for simple queries
    response = await rag.aquery(
        "Simple factual question",
        QueryParam(
            mode="local", 
            use_node_embedding=False  # Standard retrieval only
        )
    )
    """)
    
    # Example 3: Installation
    print("3️⃣  Installation:")
    print("""
    # Install with GNN dependencies
    pip install lightrag-hku[gnn]
    
    # Or install dependencies separately
    pip install lightrag-hku scikit-learn
    """)


if __name__ == "__main__":
    # Show configuration examples
    demo_configuration()
    
    # Run the main example
    if OPENAI_API_KEY != "your-openai-api-key-here":
        asyncio.run(main())
    else:
        print("⚠️  Please set your OpenAI API key in the OPENAI_API_KEY variable to run the example.")
        print("   You can also modify this example to use other LLM providers supported by LightRAG.")