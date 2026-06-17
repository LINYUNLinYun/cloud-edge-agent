"""Unit tests for the calculator tool."""

import pytest

from tools.calculator_tool import CalculatorTool


@pytest.mark.asyncio
class TestCalculator:
    """Test safe math evaluation."""

    async def test_basic_addition(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute(expression="2 + 3")
        assert result.success is True
        assert result.output == "5"

    async def test_complex_expression(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute(expression="(10 + 5) * 2 - 3")
        assert result.success is True
        assert result.output == "27"

    async def test_power(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute(expression="2 ** 10")
        assert result.success is True
        assert result.output == "1024"

    async def test_empty_expression(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute(expression="")
        assert result.success is False

    async def test_unsafe_expression_rejected(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute(expression="__import__('os').system('ls')")
        assert result.success is False
