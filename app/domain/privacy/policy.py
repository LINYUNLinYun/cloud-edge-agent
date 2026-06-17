"""Policy engine — decides routing based on privacy level and task complexity."""

from dataclasses import dataclass
from enum import Enum

from app.domain.privacy.privacy import PrivacyLevel


class ComplexityLevel(int, Enum):
    """Task complexity levels (L1 = simplest, L5 = hardest)."""

    L1 = 1  # FAQ
    L2 = 2  # single-step reasoning
    L3 = 3  # multi-step reasoning
    L4 = 4  # agent task
    L5 = 5  # long-chain complex task


class RouteDecision(str, Enum):
    """Where to route the task."""

    EDGE = "edge"
    CLOUD = "cloud"
    SANITIZED_CLOUD = "sanitized_cloud"
    SKETCH_REFINE = "sketch_refine"


class CollaborateMode(str, Enum):
    """Collaborative orchestrator mode."""

    DIRECT_LOCAL = "direct_local"        # Mode A
    DIRECT_CLOUD = "direct_cloud"        # Mode B
    SANITIZE_CLOUD = "sanitize_cloud"    # Mode C
    SKETCH_REFINE = "sketch_refine"      # Mode D


@dataclass
class RoutingResult:
    """Output of the policy engine."""

    decision: RouteDecision
    mode: CollaborateMode
    privacy_level: PrivacyLevel
    complexity: ComplexityLevel
    reason: str


# ---------------------------------------------------------------------------
# Routing matrix (from README §4.4)
# ---------------------------------------------------------------------------
#  Privacy | Complexity | Route
#  S1      | L1-L2      | Edge
#  S1      | L3-L5      | Cloud
#  S2      | L1-L2      | Edge
#  S2      | L3-L5      | Sanitized Cloud
#  S3      | Any        | Edge

ROUTE_MATRIX: dict[tuple[PrivacyLevel, str], RouteDecision] = {
    (PrivacyLevel.S1, "low"): RouteDecision.EDGE,
    (PrivacyLevel.S1, "high"): RouteDecision.CLOUD,
    (PrivacyLevel.S2, "low"): RouteDecision.EDGE,
    (PrivacyLevel.S2, "high"): RouteDecision.SANITIZED_CLOUD,
    (PrivacyLevel.S3, "low"): RouteDecision.EDGE,
    (PrivacyLevel.S3, "high"): RouteDecision.SKETCH_REFINE,  # PBCR innovation
}

_DECISION_TO_MODE = {
    RouteDecision.EDGE: CollaborateMode.DIRECT_LOCAL,
    RouteDecision.CLOUD: CollaborateMode.DIRECT_CLOUD,
    RouteDecision.SANITIZED_CLOUD: CollaborateMode.SANITIZE_CLOUD,
    RouteDecision.SKETCH_REFINE: CollaborateMode.SKETCH_REFINE,
}


def route(
    privacy_level: PrivacyLevel,
    complexity: ComplexityLevel,
    budget_exhausted: bool = False,
) -> RoutingResult:
    """Apply the routing matrix and return a decision.

    When the privacy budget is exhausted, force everything to edge.
    """
    if budget_exhausted:
        return RoutingResult(
            decision=RouteDecision.EDGE,
            mode=CollaborateMode.DIRECT_LOCAL,
            privacy_level=privacy_level,
            complexity=complexity,
            reason="Privacy budget exhausted — forcing local execution",
        )

    complexity_group = "low" if complexity.value <= 2 else "high"
    decision = ROUTE_MATRIX.get(
        (privacy_level, complexity_group), RouteDecision.EDGE
    )
    mode = _DECISION_TO_MODE[decision]

    return RoutingResult(
        decision=decision,
        mode=mode,
        privacy_level=privacy_level,
        complexity=complexity,
        reason=f"Privacy={privacy_level.value}, Complexity={complexity.value}",
    )
