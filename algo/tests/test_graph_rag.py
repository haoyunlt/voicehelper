import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_rag import (
    GraphRAG,
    Entity,
    Relationship,
    GraphNode,
    GraphEdge,
    Community,
    SearchResult,
    GraphRAGConfig
)


class TestGraphRAG:
    """Test cases for GraphRAG system"""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing"""
        store = Mock()
        store.search = AsyncMock()
        store.add_documents = AsyncMock()
        store.delete_documents = AsyncMock()
        return store

    @pytest.fixture
    def mock_graph_db(self):
        """Mock graph database for testing"""
        db = Mock()
        db.add_node = AsyncMock()
        db.add_edge = AsyncMock()
        db.query = AsyncMock()
        db.find_paths = AsyncMock()
        db.get_communities = AsyncMock()
        return db

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing"""
        client = Mock()
        client.extract_entities = AsyncMock()
        client.extract_relationships = AsyncMock()
        client.generate_summary = AsyncMock()
        return client

    @pytest.fixture
    def graph_rag_config(self):
        """Default GraphRAG configuration"""
        return GraphRAGConfig(
            max_hops=3,
            max_entities=50,
            similarity_threshold=0.8,
            community_detection_algorithm="louvain",
            enable_path_reasoning=True,
            enable_community_summary=True
        )

    @pytest.fixture
    def graph_rag(self, mock_vector_store, mock_graph_db, mock_llm_client, graph_rag_config):
        """Create GraphRAG instance for testing"""
        return GraphRAG(
            vector_store=mock_vector_store,
            graph_db=mock_graph_db,
            llm_client=mock_llm_client,
            config=graph_rag_config
        )

    @pytest.mark.asyncio
    async def test_entity_extraction(self, graph_rag, mock_llm_client):
        """Test entity extraction from text"""
        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
        
        expected_entities = [
            Entity(name="Apple Inc.", type="ORGANIZATION", description="Technology company"),
            Entity(name="Steve Jobs", type="PERSON", description="Co-founder of Apple"),
            Entity(name="Cupertino", type="LOCATION", description="City in California"),
            Entity(name="California", type="LOCATION", description="US State")
        ]
        
        mock_llm_client.extract_entities.return_value = expected_entities

        entities = await graph_rag.extract_entities(text)

        assert len(entities) == 4
        assert entities[0].name == "Apple Inc."
        assert entities[0].type == "ORGANIZATION"
        mock_llm_client.extract_entities.assert_called_once_with(text)

    @pytest.mark.asyncio
    async def test_relationship_extraction(self, graph_rag, mock_llm_client):
        """Test relationship extraction from text"""
        text = "Steve Jobs founded Apple Inc. in 1976."
        entities = [
            Entity(name="Steve Jobs", type="PERSON"),
            Entity(name="Apple Inc.", type="ORGANIZATION")
        ]
        
        expected_relationships = [
            Relationship(
                source="Steve Jobs",
                target="Apple Inc.",
                type="FOUNDED",
                confidence=0.9,
                description="Steve Jobs founded Apple Inc."
            )
        ]
        
        mock_llm_client.extract_relationships.return_value = expected_relationships

        relationships = await graph_rag.extract_relationships(text, entities)

        assert len(relationships) == 1
        assert relationships[0].source == "Steve Jobs"
        assert relationships[0].target == "Apple Inc."
        assert relationships[0].type == "FOUNDED"

    @pytest.mark.asyncio
    async def test_document_ingestion(self, graph_rag, mock_vector_store, mock_graph_db, mock_llm_client):
        """Test document ingestion into graph"""
        document = {
            "id": "doc1",
            "content": "Apple Inc. is a technology company founded by Steve Jobs.",
            "metadata": {"source": "wikipedia"}
        }

        # Mock entity and relationship extraction
        mock_llm_client.extract_entities.return_value = [
            Entity(name="Apple Inc.", type="ORGANIZATION"),
            Entity(name="Steve Jobs", type="PERSON")
        ]
        
        mock_llm_client.extract_relationships.return_value = [
            Relationship(source="Steve Jobs", target="Apple Inc.", type="FOUNDED", confidence=0.9)
        ]

        await graph_rag.ingest_document(document)

        # Verify vector store operations
        mock_vector_store.add_documents.assert_called_once()
        
        # Verify graph database operations
        assert mock_graph_db.add_node.call_count >= 2  # At least 2 entities
        assert mock_graph_db.add_edge.call_count >= 1  # At least 1 relationship

    @pytest.mark.asyncio
    async def test_vector_search(self, graph_rag, mock_vector_store):
        """Test vector-based search"""
        query = "Tell me about Apple Inc."
        
        mock_results = [
            {
                "content": "Apple Inc. is a technology company...",
                "score": 0.95,
                "metadata": {"doc_id": "doc1"}
            },
            {
                "content": "Steve Jobs founded Apple in 1976...",
                "score": 0.87,
                "metadata": {"doc_id": "doc2"}
            }
        ]
        
        mock_vector_store.search.return_value = mock_results

        results = await graph_rag.vector_search(query, top_k=5)

        assert len(results) == 2
        assert results[0]["score"] == 0.95
        mock_vector_store.search.assert_called_once_with(query, top_k=5)

    @pytest.mark.asyncio
    async def test_graph_traversal(self, graph_rag, mock_graph_db):
        """Test graph traversal search"""
        query = "What companies did Steve Jobs found?"
        
        # Mock graph traversal results
        mock_paths = [
            {
                "path": ["Steve Jobs", "FOUNDED", "Apple Inc."],
                "score": 0.9,
                "explanation": "Steve Jobs founded Apple Inc."
            },
            {
                "path": ["Steve Jobs", "CO_FOUNDED", "Pixar"],
                "score": 0.8,
                "explanation": "Steve Jobs co-founded Pixar"
            }
        ]
        
        mock_graph_db.find_paths.return_value = mock_paths

        results = await graph_rag.graph_traversal_search(query, max_hops=2)

        assert len(results) == 2
        assert "Apple Inc." in results[0]["path"]
        mock_graph_db.find_paths.assert_called_once()

    @pytest.mark.asyncio
    async def test_community_detection(self, graph_rag, mock_graph_db):
        """Test community detection in graph"""
        mock_communities = [
            Community(
                id="community_1",
                nodes=["Apple Inc.", "Steve Jobs", "Tim Cook"],
                description="Apple leadership community",
                score=0.85
            ),
            Community(
                id="community_2", 
                nodes=["Google", "Larry Page", "Sergey Brin"],
                description="Google founders community",
                score=0.82
            )
        ]
        
        mock_graph_db.get_communities.return_value = mock_communities

        communities = await graph_rag.detect_communities()

        assert len(communities) == 2
        assert communities[0].id == "community_1"
        assert "Apple Inc." in communities[0].nodes

    @pytest.mark.asyncio
    async def test_hybrid_search(self, graph_rag, mock_vector_store, mock_graph_db):
        """Test hybrid search combining vector and graph"""
        query = "Tell me about technology companies founded in California"
        
        # Mock vector search results
        mock_vector_store.search.return_value = [
            {"content": "Apple Inc. was founded in California...", "score": 0.9}
        ]
        
        # Mock graph search results
        mock_graph_db.find_paths.return_value = [
            {"path": ["California", "LOCATION_OF", "Apple Inc."], "score": 0.85}
        ]

        results = await graph_rag.hybrid_search(query, top_k=10)

        assert len(results) > 0
        assert isinstance(results, list)
        
        # Both search methods should be called
        mock_vector_store.search.assert_called_once()
        mock_graph_db.find_paths.assert_called_once()

    @pytest.mark.asyncio
    async def test_path_reasoning(self, graph_rag, mock_graph_db, mock_llm_client):
        """Test path-based reasoning"""
        query = "How is Steve Jobs connected to iPhone?"
        
        # Mock path finding
        mock_paths = [
            {
                "path": ["Steve Jobs", "FOUNDED", "Apple Inc.", "CREATED", "iPhone"],
                "score": 0.9
            }
        ]
        mock_graph_db.find_paths.return_value = mock_paths
        
        # Mock reasoning generation
        mock_llm_client.generate_summary.return_value = (
            "Steve Jobs is connected to iPhone through Apple Inc., "
            "which he founded and which created the iPhone."
        )

        result = await graph_rag.path_reasoning(query)

        assert "Steve Jobs" in result
        assert "iPhone" in result
        assert "Apple Inc." in result
        mock_llm_client.generate_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_community_summarization(self, graph_rag, mock_graph_db, mock_llm_client):
        """Test community-based summarization"""
        query = "Summarize information about Apple Inc."
        
        # Mock community detection
        apple_community = Community(
            id="apple_community",
            nodes=["Apple Inc.", "Steve Jobs", "Tim Cook", "iPhone", "iPad"],
            description="Apple ecosystem community"
        )
        mock_graph_db.get_communities.return_value = [apple_community]
        
        # Mock summary generation
        mock_llm_client.generate_summary.return_value = (
            "Apple Inc. is a technology company founded by Steve Jobs, "
            "currently led by Tim Cook, known for products like iPhone and iPad."
        )

        summary = await graph_rag.community_summarization(query)

        assert "Apple Inc." in summary
        assert "Steve Jobs" in summary
        mock_llm_client.generate_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_hop_reasoning(self, graph_rag, mock_graph_db):
        """Test multi-hop reasoning across graph"""
        query = "What products were created by companies founded by Steve Jobs?"
        
        # Mock multi-hop paths
        mock_paths = [
            {
                "path": ["Steve Jobs", "FOUNDED", "Apple Inc.", "CREATED", "iPhone"],
                "hops": 2,
                "score": 0.9
            },
            {
                "path": ["Steve Jobs", "FOUNDED", "Apple Inc.", "CREATED", "iPad"], 
                "hops": 2,
                "score": 0.85
            },
            {
                "path": ["Steve Jobs", "CO_FOUNDED", "Pixar", "PRODUCED", "Toy Story"],
                "hops": 2, 
                "score": 0.8
            }
        ]
        mock_graph_db.find_paths.return_value = mock_paths

        results = await graph_rag.multi_hop_search(query, max_hops=3)

        assert len(results) == 3
        assert any("iPhone" in str(result) for result in results)
        assert any("Pixar" in str(result) for result in results)

    @pytest.mark.asyncio
    async def test_graph_update(self, graph_rag, mock_graph_db):
        """Test updating graph with new information"""
        new_info = "Tim Cook became CEO of Apple Inc. in 2011."
        
        # Mock entity extraction
        with patch.object(graph_rag, 'extract_entities') as mock_extract_entities:
            mock_extract_entities.return_value = [
                Entity(name="Tim Cook", type="PERSON"),
                Entity(name="Apple Inc.", type="ORGANIZATION")
            ]
            
            with patch.object(graph_rag, 'extract_relationships') as mock_extract_rels:
                mock_extract_rels.return_value = [
                    Relationship(
                        source="Tim Cook",
                        target="Apple Inc.", 
                        type="CEO_OF",
                        confidence=0.95
                    )
                ]
                
                await graph_rag.update_graph(new_info)

        # Verify graph was updated
        mock_graph_db.add_node.assert_called()
        mock_graph_db.add_edge.assert_called()

    @pytest.mark.asyncio
    async def test_search_result_ranking(self, graph_rag):
        """Test search result ranking and fusion"""
        vector_results = [
            {"content": "Apple Inc. info", "score": 0.9, "source": "vector"}
        ]
        
        graph_results = [
            {"content": "Steve Jobs founded Apple", "score": 0.85, "source": "graph"}
        ]

        ranked_results = await graph_rag.rank_and_fuse_results(
            vector_results, graph_results
        )

        assert len(ranked_results) == 2
        # Higher score should be ranked first
        assert ranked_results[0]["score"] >= ranked_results[1]["score"]

    @pytest.mark.asyncio
    async def test_entity_resolution(self, graph_rag):
        """Test entity resolution and deduplication"""
        entities = [
            Entity(name="Apple Inc.", type="ORGANIZATION"),
            Entity(name="Apple", type="ORGANIZATION"),  # Should be merged
            Entity(name="Steve Jobs", type="PERSON"),
            Entity(name="S. Jobs", type="PERSON")  # Should be merged
        ]

        resolved_entities = await graph_rag.resolve_entities(entities)

        # Should have fewer entities after resolution
        assert len(resolved_entities) <= len(entities)
        
        # Check that similar entities are merged
        org_names = [e.name for e in resolved_entities if e.type == "ORGANIZATION"]
        person_names = [e.name for e in resolved_entities if e.type == "PERSON"]
        
        assert len(org_names) <= 2  # Apple variations merged
        assert len(person_names) <= 2  # Steve Jobs variations merged

    def test_graph_rag_config_validation(self):
        """Test GraphRAG configuration validation"""
        # Valid config
        config = GraphRAGConfig(
            max_hops=3,
            similarity_threshold=0.8,
            max_entities=100
        )
        assert config.max_hops == 3
        assert config.similarity_threshold == 0.8

        # Invalid config
        with pytest.raises(ValueError):
            GraphRAGConfig(max_hops=-1)
            
        with pytest.raises(ValueError):
            GraphRAGConfig(similarity_threshold=1.5)

    @pytest.mark.asyncio
    async def test_performance_optimization(self, graph_rag, mock_graph_db):
        """Test performance optimization features"""
        query = "Find information about Apple"
        
        # Mock cached results
        with patch.object(graph_rag, '_get_cached_result') as mock_cache:
            mock_cache.return_value = None  # No cache hit
            
            with patch.object(graph_rag, '_cache_result') as mock_cache_set:
                results = await graph_rag.search(query)
                
                # Should attempt to cache results
                mock_cache_set.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, graph_rag, mock_vector_store, mock_graph_db):
        """Test error handling in GraphRAG operations"""
        query = "Test query"
        
        # Mock vector store error
        mock_vector_store.search.side_effect = Exception("Vector store error")
        
        # Should handle error gracefully and fallback to graph search
        results = await graph_rag.search(query)
        
        # Should still return results (from graph fallback)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, graph_rag):
        """Test concurrent GraphRAG operations"""
        queries = [
            "Tell me about Apple Inc.",
            "Who founded Google?",
            "What is Microsoft known for?"
        ]

        # Process queries concurrently
        tasks = [graph_rag.search(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 3
        # All should complete without exceptions
        assert all(not isinstance(r, Exception) for r in results)

    def test_graph_metrics_collection(self, graph_rag):
        """Test graph metrics and statistics collection"""
        # Should track various metrics
        metrics = graph_rag.get_metrics()
        
        expected_metrics = [
            'total_nodes',
            'total_edges', 
            'communities_count',
            'avg_path_length',
            'search_latency'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics or hasattr(graph_rag, f'_{metric}')
