"""
Debug script to check graph content
"""

import asyncio
import os
from config import initialize_rag

async def debug_graph():
    """Debug knowledge graph content"""
    try:
        # Initialize RAG
        rag = await initialize_rag()
        print("RAG initialized successfully")
        
        # Get graph instance
        graph = rag.chunk_entity_relation_graph
        
        # Check if TSMC exists in the graph
        print("\n=== Checking for TSMC-related entities ===")
        
        # Get all nodes
        all_nodes = await graph.get_all_nodes()
        print(f"Total nodes in graph: {len(all_nodes)}")
        
        # Search for TSMC related entities
        tsmc_related = []
        apple_related = []
        manufacturing_related = []
        
        for node_data in all_nodes:
            entity_name = node_data.get('entity_name', '')
            description = node_data.get('description', '')
            
            if 'tsmc' in entity_name.lower() or 'tsmc' in description.lower():
                tsmc_related.append((entity_name, description[:100]))
            
            if 'apple' in entity_name.lower() or 'apple' in description.lower():
                apple_related.append((entity_name, description[:100]))
                
            if 'manufacturing' in entity_name.lower() or 'manufacturing' in description.lower():
                manufacturing_related.append((entity_name, description[:100]))
        
        print(f"\nTSMC-related entities found: {len(tsmc_related)}")
        for name, desc in tsmc_related[:5]:
            print(f"  - {name}: {desc}")
            
        print(f"\nApple-related entities found: {len(apple_related)}")
        for name, desc in apple_related[:5]:
            print(f"  - {name}: {desc}")
            
        print(f"\nManufacturing-related entities found: {len(manufacturing_related)}")
        for name, desc in manufacturing_related[:5]:
            print(f"  - {name}: {desc}")
        
        # Check relationships
        print(f"\n=== Checking relationships ===")
        all_edges = await graph.get_all_edges()
        print(f"Total edges in graph: {len(all_edges)}")
        
        # Look for edges connecting Apple and TSMC or manufacturing
        relevant_edges = []
        for edge_data in all_edges:
            src = edge_data.get('src_id', '')
            tgt = edge_data.get('tgt_id', '')
            desc = edge_data.get('description', '')
            
            if (('apple' in src.lower() or 'apple' in tgt.lower()) and 
                ('tsmc' in desc.lower() or 'manufacturing' in desc.lower())):
                relevant_edges.append((src, tgt, desc[:100]))
        
        print(f"Relevant Apple-TSMC/manufacturing edges: {len(relevant_edges)}")
        for src, tgt, desc in relevant_edges[:3]:
            print(f"  {src} -> {tgt}: {desc}")
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'rag' in locals():
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(debug_graph())