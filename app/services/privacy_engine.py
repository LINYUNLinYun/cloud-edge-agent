"""Privacy engine — three-layer detection pipeline: Regex → NER → SLM.

This is the concrete implementation of the privacy detection + sanitization
interfaces defined in domain/privacy.
"""

import re
import uuid

from app.core.logger.logger import get_logger
from app.domain.llm.llm_client import LLMClient, LLMMessage
from app.domain.privacy.privacy import (
    PrivacyDetection,
    PrivacyDetector,
    PrivacyLevel,
    Sanitizer,
    SanitizeResult,
    SensitiveEntity,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Layer 1 — Regex patterns for common PII
# ---------------------------------------------------------------------------
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "PHONE": re.compile(
        r"(?<!\d)1[3-9]\d{9}(?!\d)"
    ),
    "ID_CARD": re.compile(
        r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
    ),
    "EMAIL": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    ),
    "BANK_CARD": re.compile(
        r"(?<!\d)(?:6[0-9]{15,18}|4[0-9]{15}|5[1-5][0-9]{14}|3[47][0-9]{13})(?!\d)"
    ),
    "IP_ADDRESS": re.compile(
        r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\d)"
    ),
}


def _regex_detect(text: str) -> list[SensitiveEntity]:
    """Layer 1: Fast regex-based PII detection (~0ms)."""
    entities: list[SensitiveEntity] = []
    for entity_type, pattern in _PII_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(
                SensitiveEntity(
                    entity_type=entity_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
            )
    return entities


# ---------------------------------------------------------------------------
# Layer 2 — NER (Presidio or simple keyword-based fallback)
# ---------------------------------------------------------------------------

async def _ner_detect(text: str) -> list[SensitiveEntity]:
    """Layer 2: Named Entity Recognition via Presidio.

    Falls back to a no-op if Presidio is not installed.
    """
    try:
        from presidio_analyzer import AnalyzerEngine

        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language="en")
        return [
            SensitiveEntity(
                entity_type=result.entity_type,
                value=text[result.start : result.end],
                start=result.start,
                end=result.end,
            )
            for result in results
        ]
    except ImportError:
        logger.warning("presidio_not_installed", msg="Skipping NER layer")
        return []
    except Exception as exc:
        logger.warning("ner_failed", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Layer 3 — SLM privacy judge (Qwen 1.5B)
# ---------------------------------------------------------------------------

_SLM_JUDGE_PROMPT = """You are a privacy classifier. Analyze the following text and determine its privacy level.

Rules:
- S1 (Safe): No personal/sensitive information. General knowledge questions, public info.
- S2 (Sensitive): Contains some personal info that could be redacted (names, phone numbers, etc.)
- S3 (Confidential): Highly sensitive data that should NEVER leave the local device (bank statements, medical records, private keys, passwords)

Respond with ONLY a JSON object: {{"level": "S1|S2|S3", "confidence": 0.0-1.0, "reason": "brief explanation"}}

Text to analyze:
{text}"""


async def _slm_judge(text: str, slm_client: LLMClient) -> PrivacyDetection:
    """Layer 3: Use a small local LLM to judge privacy level."""
    messages = [
        LLMMessage(role="user", content=_SLM_JUDGE_PROMPT.format(text=text))
    ]
    try:
        response = await slm_client.invoke(messages)
        import json

        parsed = json.loads(response.content.strip())
        level = PrivacyLevel(parsed.get("level", "NA"))
        confidence = float(parsed.get("confidence", 0.5))
        reason = parsed.get("reason", "")
        return PrivacyDetection(
            level=level, confidence=confidence, reason=reason
        )
    except Exception as exc:
        logger.warning("slm_judge_failed", error=str(exc))
        return PrivacyDetection(
            level=PrivacyLevel.NA,
            confidence=0.0,
            reason=f"SLM judge failed, unable to determine: {exc}",
        )


# ---------------------------------------------------------------------------
# Composite Privacy Detector
# ---------------------------------------------------------------------------

class ThreeLayerPrivacyDetector(PrivacyDetector):
    """Three-layer privacy detection: Regex → NER → SLM.

    Uses short-circuit logic:
    - If regex finds high-confidence PII → S2 immediately
    - If NER finds entities → S2
    - Otherwise, ask SLM for judgment → S1/S2/S3/NA

    SLM client MUST be local — never cloud. If SLM is unavailable,
    Layer 3 returns NA (unknown), never S2 (sensitive) since no
    detection actually ran.
    """

    def __init__(self, slm_client: LLMClient | None) -> None:
        self._slm_client = slm_client

    async def detect(self, text: str) -> PrivacyDetection:
        """Run the three-layer detection pipeline."""
        # Layer 1: Regex
        regex_entities = _regex_detect(text)
        if regex_entities:
            logger.info(
                "privacy_regex_hit",
                count=len(regex_entities),
                types=[e.entity_type for e in regex_entities],
            )
            return PrivacyDetection(
                level=PrivacyLevel.S2,
                confidence=0.95,
                entities=regex_entities,
                reason=f"Regex detected {len(regex_entities)} PII entities",
            )

        # Layer 2: NER
        ner_entities = await _ner_detect(text)
        if ner_entities:
            logger.info(
                "privacy_ner_hit",
                count=len(ner_entities),
                types=[e.entity_type for e in ner_entities],
            )
            return PrivacyDetection(
                level=PrivacyLevel.S2,
                confidence=0.8,
                entities=ner_entities,
                reason=f"NER detected {len(ner_entities)} entities",
            )

        # Layer 3: SLM judge — only when edge is available
        if self._slm_client is not None:
            slm_result = await _slm_judge(text, self._slm_client)
            slm_result.entities = []
            logger.info(
                "privacy_slm_judge",
                level=slm_result.level.value,
                confidence=slm_result.confidence,
            )
            return slm_result

        # No local SLM — unable to judge, no false S2
        logger.info("privacy_slm_skipped", reason="edge_unavailable")
        return PrivacyDetection(
            level=PrivacyLevel.NA,
            confidence=0.0,
            reason="SLM unavailable (edge down), privacy level unknown",
        )


# ---------------------------------------------------------------------------
# Sanitizer implementation
# ---------------------------------------------------------------------------

class RegexSanitizer(Sanitizer):
    """Replace sensitive entities with typed placeholders.

    Example: "我的手机是13812345678" → "我的手机是[REDACTED:PHONE:a3f2]"
    The mapping allows exact restoration.
    """

    async def sanitize(
        self, text: str, entities: list[SensitiveEntity]
    ) -> SanitizeResult:
        """Replace each entity with a placeholder."""
        if not entities:
            return SanitizeResult(sanitized_text=text, mapping={}, entities_replaced=0)

        # Sort by start position descending to replace from end to start
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
        sanitized = text
        mapping: dict[str, str] = {}

        for entity in sorted_entities:
            tag = uuid.uuid4().hex[:6]
            placeholder = f"[REDACTED:{entity.entity_type}:{tag}]"
            sanitized = sanitized[: entity.start] + placeholder + sanitized[entity.end :]
            mapping[placeholder] = entity.value

        return SanitizeResult(
            sanitized_text=sanitized,
            mapping=mapping,
            entities_replaced=len(entities),
        )

    async def restore(self, sanitized_text: str, mapping: dict[str, str]) -> str:
        """Restore original values from the mapping."""
        result = sanitized_text
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        return result
