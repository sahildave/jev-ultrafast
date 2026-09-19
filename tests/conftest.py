"""Keep the suite off the host environment.

A test that passes because TYPESAFE_API_KEY happens to be exported in the developer's
shell is red on any other machine. Clear every variable the package reads, so each test
states its own preconditions.
"""

import pytest

OWNED = (
    "AI_GATEWAY_API_KEY",
    "TYPESAFE_API_KEY",
    "TEXT_MODEL_API_KEY",
    "TEXT_MODEL_BASE_URL",
    "TEXT_MODEL",
    "TEXT_MODEL_REASONING",
    "GATEWAY_MODEL",
    "TYPESAFE_MODEL",
    "JEV_ROUTE",
    "JEV_GUARD",
    "JEV_CLICK_ONLY",
    "JEV_REQUIRE_FREE",
    "JEV_ALLOW_PAID_GATEWAY",
    "JEV_GATEWAY_FREE_THROUGH",
    "JEV_MAX_COST_USD",
    "JEV_MIN_INTERVAL_SECONDS",
    "JEV_PACE_FILE",
    "JEV_RETRIES",
    "JEV_RETRY_BASE_SECONDS",
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch, tmp_path):
    for name in OWNED:
        monkeypatch.delenv(name, raising=False)
    # Never touch the shared pace file, and never sleep, from a unit test.
    monkeypatch.setenv("JEV_PACE_FILE", str(tmp_path / "pace"))
