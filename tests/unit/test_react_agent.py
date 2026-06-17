"""Unit tests for the ReAct agent loop.

Uses a mock LLM client to verify the Think → Act → Observe cycle
without calling any real API.
"""

from collections.abc import AsyncIterator

import pytest

from app.domain.agent.react_agent import ReActAgent, _parse_llm_output
from app.domain.llm.llm_client import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    StreamChunk,
    TokenUsage,
)
from app.domain.tool.base import BaseTool, ToolResult
from app.domain.tool.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Mock LLM — returns pre-scripted responses in sequence
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """LLM client that returns canned responses in order."""

    provider = "mock"
    model_name = "mock-model"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def invoke(self, messages: list[LLMMessage]) -> LLMResponse:
        if self._call_count < len(self._responses):
            content = self._responses[self._call_count]
        else:
            content = "Thought: I'm done.\nFinal Answer: No more responses."
        self._call_count += 1
        return LLMResponse(content=content, model="mock", usage=TokenUsage())

    async def stream_invoke(self, messages: list[LLMMessage]) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta="mock")
        return

    async def think(self, messages: list[LLMMessage]) -> LLMResponse:
        return await self.invoke(messages)

    async def embedding(self, text: str) -> list[float]:
        return [0.0] * 8


# ---------------------------------------------------------------------------
# Mock Tool
# ---------------------------------------------------------------------------

class EchoTool(BaseTool):
    """Echoes the input back."""

    name = "echo"
    description = "Echoes the input"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    async def execute(self, text: str = "", **kwargs) -> ToolResult:
        return ToolResult(output=f"Echo: {text}")


class AddTool(BaseTool):
    """Adds two numbers."""

    name = "add"
    description = "Add two numbers"
    parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
    }

    async def execute(self, a: int = 0, b: int = 0, **kwargs) -> ToolResult:
        return ToolResult(output=str(a + b))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReActAgent:
    """Test the ReAct agent loop."""

    async def test_direct_answer_no_tool(self) -> None:
        """Agent gives final answer without calling any tool."""
        llm = MockLLMClient([
            "Thought: This is a simple greeting.\nFinal Answer: Hello! How can I help you?",
        ])
        registry = ToolRegistry()
        agent = ReActAgent(llm_client=llm, tool_registry=registry)

        result = await agent.run("Hi")

        assert result.answer == "Hello! How can I help you?"
        assert len(result.steps) == 1
        assert result.steps[0].action == "respond"

    async def test_single_tool_call(self) -> None:
        """Agent calls a tool once, then gives final answer."""
        llm = MockLLMClient([
            'Thought: I need to echo the input.\n Action: echo\n Action Input: {"text": "hello"}',
            "Thought: Got the echo result.\nFinal Answer: The echo is Echo: hello",
        ])
        registry = ToolRegistry()
        registry.register(EchoTool())
        agent = ReActAgent(llm_client=llm, tool_registry=registry)

        result = await agent.run("echo hello")

        assert "Echo: hello" in result.steps[0].observation
        assert result.answer == "The echo is Echo: hello"
        assert len(result.steps) == 2

    async def test_multiple_tool_calls(self) -> None:
        """Agent calls two tools in sequence."""
        llm = MockLLMClient([
            'Thought: First add.\n Action: add\n Action Input: {"a": 2, "b": 3}',
            'Thought: Now echo the result.\n Action: echo\n Action Input: {"text": "sum is 5"}',
            "Thought: Done.\nFinal Answer: 2 + 3 = 5",
        ])
        registry = ToolRegistry()
        registry.register(AddTool())
        registry.register(EchoTool())
        agent = ReActAgent(llm_client=llm, tool_registry=registry)

        result = await agent.run("What is 2+3?")

        assert result.steps[0].observation == "5"
        assert "sum is 5" in result.steps[1].observation
        assert "2 + 3 = 5" in result.answer

    async def test_max_iterations_stops(self) -> None:
        """Agent stops after max iterations."""
        # Always produces an action, never a final answer
        llm = MockLLMClient([
            'Thought: loop\n Action: echo\n Action Input: {"text": "again"}',
        ] * 10)
        registry = ToolRegistry()
        registry.register(EchoTool())
        agent = ReActAgent(llm_client=llm, tool_registry=registry, max_iterations=3)

        result = await agent.run("loop forever")

        assert len(result.steps) == 3

    async def test_tool_error_handled(self) -> None:
        """Agent handles tool execution error gracefully."""
        class FailTool(BaseTool):
            name = "fail"
            description = "Always fails"
            parameters = {}

            async def execute(self, **kwargs) -> ToolResult:
                return ToolResult(output="", success=False, error="Something broke")

        llm = MockLLMClient([
            'Thought: Try it.\n Action: fail\n Action Input: {}',
            "Thought: It failed.\nFinal Answer: Sorry, the tool failed: Something broke",
        ])
        registry = ToolRegistry()
        registry.register(FailTool())
        agent = ReActAgent(llm_client=llm, tool_registry=registry)

        result = await agent.run("do something")

        assert "Something broke" in result.steps[0].observation
        assert "failed" in result.answer.lower()


class TestParseLLMOutput:
    """Test the ReAct output parser."""

    def test_parse_final_answer(self) -> None:
        text = "Thought: I know the answer.\nFinal Answer: 42"
        thought, action, action_input, final_answer = _parse_llm_output(text)
        assert thought == "I know the answer."
        assert final_answer == "42"
        assert action is None

    def test_parse_action(self) -> None:
        text = 'Thought: Need to calculate.\nAction: calculator\nAction Input: {"expression": "2+3"}'
        thought, action, action_input, final_answer = _parse_llm_output(text)
        assert action == "calculator"
        assert action_input == {"expression": "2+3"}
        assert final_answer is None

    def test_parse_action_input_string(self) -> None:
        """When action input is not valid JSON, wrap it as query."""
        text = "Thought: Search.\nAction: search\nAction Input: hello world"
        _, action, action_input, _ = _parse_llm_output(text)
        assert action == "search"
        assert action_input == {"query": "hello world"}
