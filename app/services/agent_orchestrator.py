"""Agent orchestrator — the core collaborative engine.

Implements 4 modes:
  Mode A: Direct Local    — user → edge agent → answer
  Mode B: Direct Cloud    — user → cloud agent → answer
  Mode C: Sanitize-Cloud  — user → sanitize → cloud → restore → answer
  Mode D: Sketch-Refine   — edge sketch → cloud refine → edge restore
"""

import time
from dataclasses import dataclass

from app.core.config.settings import Settings
from app.core.logger.logger import get_logger
from app.domain.llm.llm_client import LLMClient, LLMMessage
from app.domain.privacy.policy import (
    CollaborateMode,
    ComplexityLevel,
    RoutingResult,
    route,
)
from app.domain.privacy.privacy import (
    PrivacyBudgetTracker,
    PrivacyDetection,
    PrivacyDetector,
    Sanitizer,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Task complexity analyzer (simple LLM-based)
# ---------------------------------------------------------------------------

_COMPLEXITY_PROMPT = """Classify the complexity of the following task.

Levels:
- L1: FAQ / simple lookup
- L2: single-step reasoning
- L3: multi-step reasoning
- L4: agent task (requires tool use)
- L5: long-chain complex task

Respond with ONLY a JSON object: {"level": "L1|L2|L3|L4|L5"}

Task:
{text}"""


async def analyze_complexity(text: str, edge_client: LLMClient) -> ComplexityLevel:
    """Use the edge SLM to classify task complexity."""
    messages = [LLMMessage(role="user", content=_COMPLEXITY_PROMPT.format(text=text))]
    try:
        response = await edge_client.invoke(messages)
        import json

        parsed = json.loads(response.content.strip())
        return ComplexityLevel[parsed["level"]]
    except Exception as exc:
        logger.warning("complexity_analysis_failed", error=str(exc))
        return ComplexityLevel.L3  # safe default


# ---------------------------------------------------------------------------
# Orchestrator result
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResult:
    """Final result from the orchestrator."""

    answer: str
    mode: CollaborateMode
    routing: RoutingResult
    privacy_detection: PrivacyDetection
    latency_ms: float
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Collaborative Orchestrator
# ---------------------------------------------------------------------------

class CollaborativeOrchestrator:
    """The main orchestrator that coordinates edge + cloud agents.

    Flow:
    1. Detect privacy level (PrivacyDetector)
    2. Analyze task complexity (edge SLM)
    3. Apply routing policy (route matrix)
    4. Execute the chosen collaborate mode
    """

    def __init__(
        self,
        settings: Settings,
        edge_client: LLMClient,
        cloud_client: LLMClient,
        privacy_detector: PrivacyDetector,
        sanitizer: Sanitizer,
        budget_tracker: PrivacyBudgetTracker,
    ) -> None:
        self._settings = settings
        self._edge = edge_client
        self._cloud = cloud_client
        self._detector = privacy_detector
        self._sanitizer = sanitizer
        self._budget = budget_tracker

    async def process(
        self, query: str, session_id: str
    ) -> OrchestratorResult:
        """Process a user query through the full routing pipeline."""
        start_time = time.monotonic()

        # Step 1: Privacy detection
        privacy_result = await self._detector.detect(query)

        # Step 2: Complexity analysis
        complexity = await analyze_complexity(query, self._edge)

        # Step 3: Check budget and route
        budget_exhausted = self._budget.is_exhausted(session_id)
        routing = route(
            privacy_level=privacy_result.level,
            complexity=complexity,
            budget_exhausted=budget_exhausted,
        )

        logger.info(
            "orchestrator_routing",
            session_id=session_id,
            mode=routing.mode.value,
            privacy=privacy_result.level.value,
            complexity=complexity.value,
            budget_remaining=self._budget.get_remaining(session_id),
        )

        # Step 4: Execute mode
        answer = await self._execute_mode(
            mode=routing.mode,
            query=query,
            privacy_result=privacy_result,
            session_id=session_id,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        return OrchestratorResult(
            answer=answer,
            mode=routing.mode,
            routing=routing,
            privacy_detection=privacy_result,
            latency_ms=round(elapsed_ms, 1),
        )

    async def _execute_mode(
        self,
        mode: CollaborateMode,
        query: str,
        privacy_result: PrivacyDetection,
        session_id: str,
    ) -> str:
        """Dispatch to the appropriate collaborate mode."""
        if mode == CollaborateMode.DIRECT_LOCAL:
            return await self._mode_direct_local(query)
        elif mode == CollaborateMode.DIRECT_CLOUD:
            return await self._mode_direct_cloud(query)
        elif mode == CollaborateMode.SANITIZE_CLOUD:
            cost = self._settings.privacy.sanitize_cost_epsilon
            return await self._mode_sanitize_cloud(query, privacy_result, session_id, cost)
        elif mode == CollaborateMode.SKETCH_REFINE:
            cost = self._settings.privacy.sketch_refine_cost_epsilon
            return await self._mode_sketch_refine(query, privacy_result, session_id, cost)
        else:
            return await self._mode_direct_local(query)

    # --- Mode A: Direct Local ---
    async def _mode_direct_local(self, query: str) -> str:
        """Simple local execution — edge agent handles everything."""
        messages = [LLMMessage(role="user", content=query)]
        response = await self._edge.invoke(messages)
        return response.content

    # --- Mode B: Direct Cloud ---
    async def _mode_direct_cloud(self, query: str) -> str:
        """Simple cloud execution — no privacy concerns."""
        messages = [LLMMessage(role="user", content=query)]
        response = await self._cloud.invoke(messages)
        return response.content

    # --- Mode C: Sanitize → Cloud → Restore ---
    async def _mode_sanitize_cloud(
        self,
        query: str,
        privacy_result: PrivacyDetection,
        session_id: str,
        cost: float,
    ) -> str:
        """Sanitize sensitive data, send to cloud, restore in result."""
        # Sanitize
        sanitize_result = await self._sanitizer.sanitize(query, privacy_result.entities)
        logger.info(
            "sanitized",
            entities_replaced=sanitize_result.entities_replaced,
            original_len=len(query),
            sanitized_len=len(sanitize_result.sanitized_text),
        )

        # Cloud inference on sanitized text
        messages = [LLMMessage(role="user", content=sanitize_result.sanitized_text)]
        response = await self._cloud.invoke(messages)

        # Restore original values in the response
        restored = await self._sanitizer.restore(
            response.content, sanitize_result.mapping
        )

        # Consume privacy budget
        self._budget.consume(session_id, cost)

        return restored

    # --- Mode D: Sketch-Refine (PBCR innovation) ---
    async def _mode_sketch_refine(
        self,
        query: str,
        privacy_result: PrivacyDetection,
        session_id: str,
        cost: float,
    ) -> str:
        """Edge generates a privacy-safe sketch, cloud refines it.

        This is the PBCR innovation: for S3 (confidential) + complex tasks,
        instead of forcing pure local execution, we use a collaborative approach:
        1. Edge: sanitize + summarize into a "sketch" (privacy-safe)
        2. Cloud: refine/expand the sketch with deeper reasoning
        3. Edge: restore original context in the final answer
        """
        # Step 1: Edge generates a sanitized sketch
        sanitize_result = await self._sanitizer.sanitize(query, privacy_result.entities)

        sketch_prompt = (
            f"Based on the following sanitized context, provide a detailed analysis "
            f"or summary. The original data has been redacted for privacy.\n\n"
            f"Sanitized context:\n{sanitize_result.sanitized_text}"
        )
        messages = [LLMMessage(role="user", content=sketch_prompt)]
        sketch_response = await self._edge.invoke(messages)

        # Step 2: Cloud refines the sketch
        refine_prompt = (
            f"Refine and expand the following analysis with deeper reasoning:\n\n"
            f"{sketch_response.content}"
        )
        messages = [LLMMessage(role="user", content=refine_prompt)]
        refined_response = await self._cloud.invoke(messages)

        # Step 3: Edge restores context
        restored = await self._sanitizer.restore(
            refined_response.content, sanitize_result.mapping
        )

        # Consume privacy budget
        self._budget.consume(session_id, cost)

        return restored
