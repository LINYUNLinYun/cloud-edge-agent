"""Privacy engine abstractions — detection, sanitization, budget tracking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Privacy levels (S1 = safe, S2 = sensitive, S3 = confidential)
# ---------------------------------------------------------------------------

class PrivacyLevel(str, Enum):
    S1 = "S1"  # safe — can go to cloud directly
    S2 = "S2"  # sensitive — sanitize before cloud
    S3 = "S3"  # confidential — must stay local


@dataclass
class PrivacyDetection:
    """Result of privacy analysis on a text."""

    level: PrivacyLevel
    confidence: float
    entities: list["SensitiveEntity"] = field(default_factory=list)
    reason: str = ""


@dataclass
class SensitiveEntity:
    """A detected sensitive entity in text."""

    entity_type: str  # PHONE, ID_CARD, EMAIL, NAME, ADDRESS, etc.
    value: str
    start: int
    end: int


@dataclass
class SanitizeResult:
    """Output of the sanitization process."""

    sanitized_text: str
    mapping: dict[str, str]  # placeholder → original value
    entities_replaced: int


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class PrivacyDetector(ABC):
    """Detect privacy level of input text (Layer 1-3 pipeline)."""

    @abstractmethod
    async def detect(self, text: str) -> PrivacyDetection:
        """Analyze text and return privacy level + detected entities."""
        ...


class Sanitizer(ABC):
    """Sanitize (redact) sensitive information from text."""

    @abstractmethod
    async def sanitize(self, text: str, entities: list[SensitiveEntity]) -> SanitizeResult:
        """Replace sensitive entities with placeholders."""
        ...

    @abstractmethod
    async def restore(self, sanitized_text: str, mapping: dict[str, str]) -> str:
        """Restore original values from placeholder mapping."""
        ...


class PrivacyBudgetTracker(ABC):
    """Track cumulative privacy budget (ε) for a session."""

    @abstractmethod
    def get_remaining(self, session_id: str) -> float:
        """Return remaining privacy budget ε for the session."""
        ...

    @abstractmethod
    def consume(self, session_id: str, cost: float) -> float:
        """Consume budget and return remaining ε."""
        ...

    @abstractmethod
    def is_exhausted(self, session_id: str) -> bool:
        """True if the session has no remaining budget."""
        ...
