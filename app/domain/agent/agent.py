"""ReAct Agent abstraction — Think → Act → Observe → Reflect loop."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class AgentRole(str, Enum):
    EDGE = "edge"
    CLOUD = "cloud"


@dataclass
class AgentStep:
    """One step in the ReAct loop."""

    thought: str
    action: str  # tool name or "respond"
    action_input: dict
    observation: str = ""
    reflection: str = ""


@dataclass
class AgentResult:
    """Final result from an agent run."""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_tokens: int = 0
    latency_ms: float = 0.0


class BaseAgent(ABC):
    """Abstract agent that follows the ReAct reasoning pattern.

    Subclasses: EdgeAgent, CloudAgent.
    The orchestrator selects which agent to invoke.
    """

    role: AgentRole

    @abstractmethod
    async def run(self, query: str, context: dict | None = None) -> AgentResult:
        """Execute the ReAct loop until a final answer is produced.

        Args:
            query: user question.
            context: optional extra context (memory, prior state, etc.).

        Returns:
            AgentResult with answer and reasoning trace.
        """
        ...
