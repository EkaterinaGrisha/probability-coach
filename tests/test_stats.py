import uuid

import pytest

from coach import stats


@pytest.fixture
def user():
    user_id = f"test-{uuid.uuid4().hex[:8]}"
    yield user_id
    with stats.connect() as conn:
        conn.execute("DELETE FROM attempts WHERE user_id = %s", (user_id,))


def test_summary(user):
    stats.save(user, "crt_01", "crt", True, 90)
    stats.save(user, "crt_02", "crt", False, 80)
    stats.save(user, "birthday_01", "birthday", False, 50)

    result = stats.summary(user)
    assert result["attempts"] == 3
    assert result["brier"] == pytest.approx((0.1**2 + 0.8**2 + 0.5**2) / 3, abs=1e-3)
    assert [t["trap"] for t in result["by_trap"]] == ["birthday", "crt"]
    assert result["calibration"] == [
        {"confidence_range": "40–60%", "attempts": 1, "mean_confidence": 0.5, "accuracy": 0.0},
        {"confidence_range": "80–100%", "attempts": 2, "mean_confidence": 0.85, "accuracy": 0.5},
    ]
    assert sorted(stats.solved(user)) == ["birthday_01", "crt_01", "crt_02"]


def test_summary_without_attempts(user):
    assert stats.summary(user) == {"attempts": 0}
