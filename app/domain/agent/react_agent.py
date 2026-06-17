"""ReAct Agent — Think → Act → Observe loop.

Minimal implementation: LLM decides when to call tools and when to give
a final answer. Uses OpenAI function-calling protocol for tool dispatch.
"""

import json
import time

from app.core.logger.logger import get_logger
from app.domain.agent.agent import AgentResult, AgentRole, AgentStep, BaseAgent
from app.domain.llm.llm_client import LLMClient, LLMMessage
from app.domain.tool.base import ToolResult
from app.domain.tool.registry import ToolRegistry

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a helpful AI assistant. You have access to the following tools:

{tool_descriptions}

To use a tool, respond EXACTLY in this format (no extra text):

 Thought: <your reasoning>
 Action: <tool_name>
 Action Input: <JSON arguments>

After receiving a tool result, continue reasoning or give your final answer:

 Thought: <your reasoning>
 Final Answer: <your answer to the user>

Rules:
- You may call multiple tools in sequence before giving a final answer.
- Always start with a Thought.
- If you don't need a tool, go directly to Final Answer.
- Be concise and helpful."""

MAX_ITERATIONS = 8


def _build_tool_descriptions(registry: ToolRegistry) -> str:
    """Format tool metadata for the system prompt."""
    lines = []
    for tool_meta in registry.list_tools():
        params = tool_meta.get("parameters", {})
        props = params.get("properties", {})
        param_str = ", ".join(
            f"{k}: {v.get('type', 'any')}" for k, v in props.items()
        )
        lines.append(f"- **{tool_meta['name']}**({param_str}): {tool_meta['description']}")
    return "\n".join(lines) if lines else "(no tools available)"


class ReActAgent(BaseAgent):
    """Agent that reasons using Think → Act → Observe loop."""

    role = AgentRole.EDGE  # default; caller can override

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self._llm = llm_client
        self._tools = tool_registry
        self._max_iterations = max_iterations

    async def run(self, query: str, context: dict | None = None) -> AgentResult:
        """Execute the ReAct loop until a Final Answer is produced."""
        start_time = time.monotonic()
        steps: list[AgentStep] = []

        system_prompt = _SYSTEM_PROMPT.format(
            tool_descriptions=_build_tool_descriptions(self._tools)
        )
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=query),
        ]

        for i in range(self._max_iterations):
            logger.info("react_iteration", step=i + 1, query=query[:50])
            response = await self._llm.invoke(messages)
            raw_text = response.content

            # Parse the response
            thought, action, action_input, final_answer = _parse_llm_output(raw_text)

            if final_answer is not None:
                # Agent decided it's done
                steps.append(
                    AgentStep(
                        thought=thought,
                        action="respond",
                        action_input={},
                        observation=final_answer,
                    )
                )
                elapsed_ms = (time.monotonic() - start_time) * 1000
                return AgentResult(
                    answer=final_answer,
                    steps=steps,
                    total_tokens=_total_tokens(steps),
                    latency_ms=round(elapsed_ms, 1),
                )

            if action:
                # Execute the tool
                tool_result: ToolResult = await self._tools.execute(
                    action, **(action_input or {})
                )
                observation = tool_result.output if tool_result.success else f"Error: {tool_result.error}"

                steps.append(
                    AgentStep(
                        thought=thought,
                        action=action,
                        action_input=action_input or {},
                        observation=observation,
                    )
                )

                # Feed observation back to LLM
                messages.append(LLMMessage(role="assistant", content=raw_text))
                messages.append(
                    LLMMessage(
                        role="user",
                        content=f"Observation: {observation}\n\nContinue reasoning or give your Final Answer.",
                    )
                )
            else:
                # LLM didn't produce a recognizable action or final answer
                # Treat the whole response as the final answer
                steps.append(
                    AgentStep(
                        thought=thought,
                        action="respond",
                        action_input={},
                        observation=raw_text,
                    )
                )
                elapsed_ms = (time.monotonic() - start_time) * 1000
                return AgentResult(
                    answer=raw_text,
                    steps=steps,
                    total_tokens=_total_tokens(steps),
                    latency_ms=round(elapsed_ms, 1),
                )

        # Max iterations reached — return what we have
        elapsed_ms = (time.monotonic() - start_time) * 1000
        last_answer = steps[-1].observation if steps else "I couldn't complete the task."
        return AgentResult(
            answer=last_answer,
            steps=steps,
            total_tokens=_total_tokens(steps),
            latency_ms=round(elapsed_ms, 1),
        )


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def _parse_llm_output(text: str) -> tuple[str, str | None, dict | None, str | None]:
    """Parse ReAct-formatted LLM output.

    Returns:
        (thought, action, action_input, final_answer)
        - If final_answer is set, the agent is done.
        - If action is set, the agent wants to call a tool.
    """
    thought = ""
    action = None
    action_input = None
    final_answer = None

    lines = text.strip().split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Thought:"):
            thought = stripped[len("Thought:"):].strip()
        elif stripped.startswith("Action:"):
            action = stripped[len("Action:"):].strip()
        elif stripped.startswith("Action Input:"):
            raw_input = stripped[len("Action Input:"):].strip()
            try:
                action_input = json.loads(raw_input)
            except json.JSONDecodeError:
                action_input = {"query": raw_input}
        elif stripped.startswith("Final Answer:"):
            final_answer = stripped[len("Final Answer:"):].strip()

    return thought, action, action_input, final_answer


def _total_tokens(steps: list[AgentStep]) -> int:
    """Placeholder for token counting."""
    return 0
