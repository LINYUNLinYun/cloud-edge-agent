"""Unit tests for the privacy detection pipeline."""

import pytest

from app.services.privacy_engine import (
    RegexSanitizer,
    _regex_detect,
)


class TestRegexDetection:
    """Test Layer 1 regex-based PII detection."""

    def test_detects_phone_number(self) -> None:
        text = "我的手机号是13812345678"
        entities = _regex_detect(text)
        assert len(entities) == 1
        assert entities[0].entity_type == "PHONE"
        assert entities[0].value == "13812345678"

    def test_detects_id_card(self) -> None:
        text = "身份证号110101199001011234"
        entities = _regex_detect(text)
        assert len(entities) == 1
        assert entities[0].entity_type == "ID_CARD"

    def test_detects_email(self) -> None:
        text = "联系我 test@example.com"
        entities = _regex_detect(text)
        assert len(entities) == 1
        assert entities[0].entity_type == "EMAIL"

    def test_no_pii_in_general_text(self) -> None:
        text = "今天天气怎么样"
        entities = _regex_detect(text)
        assert len(entities) == 0

    def test_detects_multiple_entities(self) -> None:
        text = "手机13812345678，邮箱test@example.com"
        entities = _regex_detect(text)
        assert len(entities) == 2


@pytest.mark.asyncio
class TestSanitizer:
    """Test sanitization and restoration."""

    async def test_sanitize_and_restore(self) -> None:
        sanitizer = RegexSanitizer()
        text = "我的手机号是13812345678"
        entities = _regex_detect(text)

        result = await sanitizer.sanitize(text, entities)
        assert "[REDACTED:" in result.sanitized_text
        assert "13812345678" not in result.sanitized_text
        assert result.entities_replaced == 1

        # Restore
        restored = await sanitizer.restore(result.sanitized_text, result.mapping)
        assert restored == text

    async def test_sanitize_no_entities(self) -> None:
        sanitizer = RegexSanitizer()
        text = "今天天气好"
        result = await sanitizer.sanitize(text, [])
        assert result.sanitized_text == text
        assert result.entities_replaced == 0
