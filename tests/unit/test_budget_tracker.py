"""Unit tests for the privacy budget tracker."""

from app.services.privacy_engine import EpsilonBudgetTracker


class TestBudgetTracker:
    """Test epsilon budget tracking."""

    def test_initial_budget(self) -> None:
        tracker = EpsilonBudgetTracker(default_epsilon=8.0)
        assert tracker.get_remaining("session-1") == 8.0

    def test_consume_budget(self) -> None:
        tracker = EpsilonBudgetTracker(default_epsilon=8.0)
        remaining = tracker.consume("session-1", 2.0)
        assert remaining == 6.0

    def test_budget_exhausted(self) -> None:
        tracker = EpsilonBudgetTracker(default_epsilon=2.0)
        tracker.consume("session-1", 2.0)
        assert tracker.is_exhausted("session-1") is True

    def test_budget_not_negative(self) -> None:
        tracker = EpsilonBudgetTracker(default_epsilon=1.0)
        remaining = tracker.consume("session-1", 5.0)
        assert remaining == 0.0

    def test_separate_sessions(self) -> None:
        tracker = EpsilonBudgetTracker(default_epsilon=8.0)
        tracker.consume("session-1", 5.0)
        assert tracker.get_remaining("session-2") == 8.0
