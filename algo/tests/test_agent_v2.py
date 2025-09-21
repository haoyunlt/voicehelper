import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent_v2 import (
    AgentV2,
    AgentState,
    AgentConfig,
    ReasoningType,
    PlanningStrategy,
    MemoryType,
    Tool,
    ToolCall,
    AgentResponse
)


class TestAgentV2:
    """Test cases for AgentV2 class"""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing"""
        client = Mock()
        client.generate = AsyncMock()
        client.generate_stream = AsyncMock()
        return client

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager for testing"""
        memory = Mock()
        memory.get_context = AsyncMock(return_value="")
        memory.add_interaction = AsyncMock()
        memory.get_relevant_memories = AsyncMock(return_value=[])
        return memory

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry for testing"""
        registry = Mock()
        registry.get_tool = Mock()
        registry.list_available_tools = Mock(return_value=[])
        return registry

    @pytest.fixture
    def agent_config(self):
        """Default agent configuration for testing"""
        return AgentConfig(
            max_iterations=5,
            reasoning_types=[ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE],
            planning_strategy=PlanningStrategy.HIERARCHICAL,
            memory_types=[MemoryType.SHORT_TERM, MemoryType.WORKING],
            confidence_threshold=0.8,
            enable_self_correction=True,
            max_tool_calls=10
        )

    @pytest.fixture
    def agent(self, mock_llm_client, mock_memory_manager, mock_tool_registry, agent_config):
        """Create agent instance for testing"""
        return AgentV2(
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            tool_registry=mock_tool_registry,
            config=agent_config
        )

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent, agent_config):
        """Test agent initialization"""
        assert agent.config == agent_config
        assert agent.state.status == "idle"
        assert agent.state.current_iteration == 0
        assert len(agent.state.reasoning_chain) == 0

    @pytest.mark.asyncio
    async def test_simple_query_processing(self, agent, mock_llm_client):
        """Test processing a simple query without tool calls"""
        query = "What is the capital of France?"
        expected_response = "The capital of France is Paris."
        
        mock_llm_client.generate.return_value = expected_response

        response = await agent.process(query)

        assert isinstance(response, AgentResponse)
        assert response.content == expected_response
        assert response.confidence > 0
        assert len(response.tool_calls) == 0
        mock_llm_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_reasoning_chain_construction(self, agent, mock_llm_client):
        """Test reasoning chain construction"""
        query = "If all birds can fly, and penguins are birds, can penguins fly?"
        
        # Mock reasoning steps
        mock_llm_client.generate.side_effect = [
            "Premise 1: All birds can fly",
            "Premise 2: Penguins are birds", 
            "Conclusion: Penguins can fly (but this contradicts reality)"
        ]

        response = await agent.process(query, reasoning_type=ReasoningType.DEDUCTIVE)

        assert len(agent.state.reasoning_chain) > 0
        assert agent.state.reasoning_chain[0].type == ReasoningType.DEDUCTIVE
        assert mock_llm_client.generate.call_count >= 1

    @pytest.mark.asyncio
    async def test_tool_call_execution(self, agent, mock_llm_client, mock_tool_registry):
        """Test tool call execution"""
        query = "What's the weather like in New York?"
        
        # Mock tool
        weather_tool = Mock()
        weather_tool.name = "get_weather"
        weather_tool.execute = AsyncMock(return_value="Sunny, 25°C")
        mock_tool_registry.get_tool.return_value = weather_tool

        # Mock LLM response with tool call
        mock_llm_client.generate.return_value = """
        I need to check the weather. Let me use the weather tool.
        
        Tool Call: get_weather
        Parameters: {"location": "New York"}
        """

        with patch.object(agent, '_parse_tool_calls') as mock_parse:
            mock_parse.return_value = [
                ToolCall(
                    name="get_weather",
                    parameters={"location": "New York"},
                    id="call_1"
                )
            ]
            
            response = await agent.process(query)

        assert len(response.tool_calls) > 0
        weather_tool.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_planning_strategy_hierarchical(self, agent, mock_llm_client):
        """Test hierarchical planning strategy"""
        query = "Plan a trip to Japan for 7 days"
        
        mock_llm_client.generate.side_effect = [
            "Main goal: Plan 7-day Japan trip",
            "Sub-goal 1: Choose destinations",
            "Sub-goal 2: Book accommodations", 
            "Sub-goal 3: Plan daily activities"
        ]

        response = await agent.process(query, planning_strategy=PlanningStrategy.HIERARCHICAL)

        assert len(agent.state.plan.steps) > 0
        assert agent.state.plan.strategy == PlanningStrategy.HIERARCHICAL

    @pytest.mark.asyncio
    async def test_memory_integration(self, agent, mock_memory_manager):
        """Test memory system integration"""
        query = "Remember that I like Italian food"
        
        await agent.process(query)

        # Verify memory interactions
        mock_memory_manager.get_context.assert_called()
        mock_memory_manager.add_interaction.assert_called()

    @pytest.mark.asyncio
    async def test_self_correction_mechanism(self, agent, mock_llm_client):
        """Test self-correction mechanism"""
        query = "What is 2 + 2?"
        
        # First response is wrong, second is corrected
        mock_llm_client.generate.side_effect = [
            "2 + 2 = 5",  # Wrong answer
            "Let me recalculate: 2 + 2 = 4"  # Corrected answer
        ]

        with patch.object(agent, '_evaluate_response') as mock_evaluate:
            # First evaluation fails, second passes
            mock_evaluate.side_effect = [
                {"confidence": 0.3, "needs_correction": True},
                {"confidence": 0.9, "needs_correction": False}
            ]
            
            response = await agent.process(query)

        assert response.confidence > agent.config.confidence_threshold
        assert mock_llm_client.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self, agent, mock_llm_client):
        """Test maximum iterations limit"""
        query = "Complex multi-step problem"
        
        # Mock responses that always require more iterations
        mock_llm_client.generate.return_value = "Need more analysis..."
        
        with patch.object(agent, '_should_continue') as mock_continue:
            mock_continue.return_value = True  # Always continue
            
            response = await agent.process(query)

        # Should stop at max_iterations
        assert agent.state.current_iteration <= agent.config.max_iterations

    @pytest.mark.asyncio
    async def test_confidence_threshold_enforcement(self, agent, mock_llm_client):
        """Test confidence threshold enforcement"""
        query = "Uncertain question"
        
        mock_llm_client.generate.return_value = "I'm not sure about this..."
        
        with patch.object(agent, '_calculate_confidence') as mock_confidence:
            mock_confidence.return_value = 0.5  # Below threshold
            
            response = await agent.process(query)

        # Should indicate low confidence
        assert response.confidence < agent.config.confidence_threshold
        assert "uncertain" in response.content.lower() or response.confidence < 0.8

    @pytest.mark.asyncio
    async def test_streaming_response(self, agent, mock_llm_client):
        """Test streaming response generation"""
        query = "Tell me a story"
        
        # Mock streaming response
        async def mock_stream():
            chunks = ["Once", " upon", " a", " time..."]
            for chunk in chunks:
                yield chunk
        
        mock_llm_client.generate_stream.return_value = mock_stream()

        response_chunks = []
        async for chunk in agent.process_stream(query):
            response_chunks.append(chunk)

        assert len(response_chunks) > 0
        full_response = "".join(response_chunks)
        assert "Once upon a time" in full_response

    @pytest.mark.asyncio
    async def test_error_handling(self, agent, mock_llm_client):
        """Test error handling in agent processing"""
        query = "Test query"
        
        # Mock LLM error
        mock_llm_client.generate.side_effect = Exception("LLM service unavailable")

        response = await agent.process(query)

        # Should handle error gracefully
        assert response is not None
        assert "error" in response.content.lower() or response.confidence == 0

    @pytest.mark.asyncio
    async def test_context_preservation(self, agent, mock_memory_manager):
        """Test context preservation across interactions"""
        queries = [
            "My name is Alice",
            "What is my name?",
            "I like pizza",
            "What food do I like?"
        ]

        # Mock memory to return previous context
        mock_memory_manager.get_context.side_effect = [
            "",  # First query - no context
            "User's name is Alice",  # Second query - has name context
            "User's name is Alice",  # Third query - still has name
            "User's name is Alice. User likes pizza"  # Fourth query - has both
        ]

        responses = []
        for query in queries:
            response = await agent.process(query)
            responses.append(response)

        # Verify memory was queried for each interaction
        assert mock_memory_manager.get_context.call_count == len(queries)
        assert mock_memory_manager.add_interaction.call_count == len(queries)

    @pytest.mark.asyncio
    async def test_tool_call_parsing(self, agent):
        """Test tool call parsing from LLM response"""
        response_text = """
        I need to search for information and then calculate something.
        
        Tool Call: search_web
        Parameters: {"query": "Python programming", "limit": 5}
        
        Tool Call: calculator
        Parameters: {"expression": "10 * 5 + 3"}
        """

        tool_calls = agent._parse_tool_calls(response_text)

        assert len(tool_calls) == 2
        assert tool_calls[0].name == "search_web"
        assert tool_calls[0].parameters["query"] == "Python programming"
        assert tool_calls[1].name == "calculator"
        assert tool_calls[1].parameters["expression"] == "10 * 5 + 3"

    def test_agent_state_management(self, agent):
        """Test agent state management"""
        # Initial state
        assert agent.state.status == "idle"
        assert agent.state.current_iteration == 0

        # Update state
        agent.state.status = "processing"
        agent.state.current_iteration = 1
        agent.state.add_reasoning_step(
            type=ReasoningType.DEDUCTIVE,
            content="Test reasoning step",
            confidence=0.8
        )

        assert agent.state.status == "processing"
        assert agent.state.current_iteration == 1
        assert len(agent.state.reasoning_chain) == 1

    def test_agent_config_validation(self):
        """Test agent configuration validation"""
        # Valid config
        config = AgentConfig(
            max_iterations=10,
            confidence_threshold=0.7,
            max_tool_calls=5
        )
        assert config.max_iterations == 10
        assert config.confidence_threshold == 0.7

        # Invalid config should raise error or use defaults
        with pytest.raises(ValueError):
            AgentConfig(max_iterations=-1)

        with pytest.raises(ValueError):
            AgentConfig(confidence_threshold=1.5)

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, agent, mock_llm_client):
        """Test concurrent query processing"""
        queries = [
            "What is 1+1?",
            "What is 2+2?", 
            "What is 3+3?"
        ]

        mock_llm_client.generate.side_effect = ["2", "4", "6"]

        # Process queries concurrently
        tasks = [agent.process(query) for query in queries]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 3
        assert all(isinstance(r, AgentResponse) for r in responses)

    @pytest.mark.asyncio
    async def test_reasoning_type_selection(self, agent, mock_llm_client):
        """Test automatic reasoning type selection"""
        test_cases = [
            ("All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?", ReasoningType.DEDUCTIVE),
            ("I've seen 100 swans and they were all white. Are all swans white?", ReasoningType.INDUCTIVE),
            ("The grass is wet. What might have caused this?", ReasoningType.ABDUCTIVE),
            ("Learning to ride a bike is like learning to drive a car.", ReasoningType.ANALOGICAL)
        ]

        for query, expected_type in test_cases:
            with patch.object(agent, '_select_reasoning_type') as mock_select:
                mock_select.return_value = expected_type
                
                await agent.process(query)
                
                mock_select.assert_called_once_with(query)

    def test_performance_metrics(self, agent):
        """Test performance metrics collection"""
        # Agent should track performance metrics
        assert hasattr(agent.state, 'metrics')
        
        # Metrics should include timing, token usage, etc.
        expected_metrics = ['processing_time', 'token_usage', 'tool_calls_count']
        for metric in expected_metrics:
            assert hasattr(agent.state.metrics, metric) or metric in agent.state.metrics
