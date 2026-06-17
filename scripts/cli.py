"""Minimal CLI — chat with the agent in terminal.

Usage:
    1. Copy configs/.env.example to .env and fill in CLOUD_LLM_API_KEY
    2. python scripts/cli.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from app.core.config.settings import get_settings
from app.core.logger.logger import get_logger, setup_logging
from app.domain.agent.react_agent import ReActAgent
from app.domain.tool.registry import ToolRegistry
from app.infrastructure.llm.client_factory import create_cloud_llm_client
from tools.calculator_tool import CalculatorTool
from tools.search_tool import SearchTool
from tools.time_tool import TimeTool

logger = get_logger("cli")

# ── Banner ──────────────────────────────────────────────────────────────
BANNER = r"""
  ╔═══════════════════════════════════════════════════╗
  ║   CloudEdgeAgent — Minimal Prototype (CLI)       ║
  ║   Powered by DeepSeek API                        ║
  ╚═══════════════════════════════════════════════════╝

  Type your message and press Enter.
  Type 'quit' or 'exit' to leave.
  Type 'trace' to toggle step-by-step reasoning display.
"""


async def main() -> None:
    """Interactive chat loop."""
    # Load .env from project root
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"  ✓ Loaded .env from {env_path}")
    else:
        print(f"  ⚠ No .env found at {env_path}")
        print("  Copy configs/.env.example to .env and fill in CLOUD_LLM_API_KEY")
        return

    settings = get_settings()
    setup_logging(settings.log_level)

    # Validate API key
    if not settings.cloud_llm.api_key or settings.cloud_llm.api_key == "":
        print("  ✗ CLOUD_LLM_API_KEY is empty. Please set it in .env")
        return

    # Create LLM client
    llm_client = create_cloud_llm_client(settings.cloud_llm)
    print(f"  ✓ LLM: {settings.cloud_llm.provider} / {settings.cloud_llm.model_name}")

    # Register tools
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TimeTool())
    registry.register(SearchTool())
    print(f"  ✓ Tools: {', '.join(t['name'] for t in registry.list_tools())}")

    # Create agent
    agent = ReActAgent(llm_client=llm_client, tool_registry=registry)
    print(f"  ✓ Agent ready (max {agent._max_iterations} iterations)")

    print(BANNER)

    show_trace = False

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("  Bye!")
            break
        if user_input.lower() == "trace":
            show_trace = not show_trace
            print(f"  Trace mode: {'ON' if show_trace else 'OFF'}")
            continue

        # Run agent
        print("  Thinking...")
        result = await agent.run(user_input)

        # Show trace if enabled
        if show_trace and result.steps:
            print("  ── Trace ─────────────────────────────────")
            for _i, step in enumerate(result.steps):
                if step.thought:
                    print(f"  💭 Thought: {step.thought}")
                if step.action and step.action != "respond":
                    print(f"  🔧 Action:  {step.action}({step.action_input})")
                    print(f"  👁 Observe: {step.observation[:200]}")
            print("  ───────────────────────────────────────────")

        # Show answer
        print(f"  Agent: {result.answer}")
        print(f"  [{result.latency_ms}ms, {len(result.steps)} steps]")
        print()


if __name__ == "__main__":
    asyncio.run(main())
