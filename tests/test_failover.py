"""Ordered legs and failover. Offline, no paid APIs."""

import pytest

from jev_ultrafast import model
from jev_ultrafast.guards import Blocked, Budget

BODY = {"model": "jev-latest", "state": {}, "questions": {}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("JEV_ROUTE", "JEV_REQUIRE_FREE", "JEV_ALLOW_PAID_GATEWAY", "JEV_GATEWAY_FREE_THROUGH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JEV_GATEWAY_FREE_THROUGH", "2099-01-01")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")


def test_the_free_leg_is_tried_first():
    """Measured 0.57s against the direct API's 0.87s, and free -- there is no reason to go second."""
    assert [leg["name"] for leg in model.decision_legs(dict(BODY))] == ["gateway", "direct"]


def test_require_free_drops_the_paid_leg(monkeypatch):
    monkeypatch.setenv("JEV_REQUIRE_FREE", "1")
    assert [leg["name"] for leg in model.decision_legs(dict(BODY))] == ["gateway"]


def test_a_rate_limited_gateway_fails_over_to_the_direct_api(monkeypatch):
    tried = []

    def fake_post(url, _key, _body, _headers=None, retries=None):
        tried.append(url)
        if "ai-gateway" in url:
            raise model.ProviderError("HTTP 429 free tier", 429)
        return {"answers": {}, "usage": {"input_tokens": 1000}}

    monkeypatch.setattr(model, "post_json", fake_post)
    result, leg = model.post_decision(dict(BODY))
    assert leg == "direct"
    assert len(tried) == 2 and "api.typesafe.ai" in tried[1]
    assert result["usage"] == {"input_tokens": 1000}


def test_a_bad_request_does_not_fail_over(monkeypatch):
    """A 400 is our bug on every leg; failing over spends a second provider to learn the same thing."""
    tried = []

    def fake_post(url, _key, _body, _headers=None, retries=None):
        tried.append(url)
        raise model.ProviderError("HTTP 400 bad request", 400)

    monkeypatch.setattr(model, "post_json", fake_post)
    with pytest.raises(model.ProviderError, match="400"):
        model.post_decision(dict(BODY))
    assert len(tried) == 1, "a 400 must not be retried on the paid leg"


def test_a_connection_failure_fails_over(monkeypatch):
    calls = []

    def fake_post(url, _key, _body, _headers=None, retries=None):
        calls.append(url)
        if "ai-gateway" in url:
            raise model.ProviderError("Model connection failed", None)
        return {"answers": {}}

    monkeypatch.setattr(model, "post_json", fake_post)
    _result, leg = model.post_decision(dict(BODY))
    assert leg == "direct"


def test_the_last_leg_raises_rather_than_swallowing(monkeypatch):
    monkeypatch.setenv("JEV_ROUTE", "direct")

    def fake_post(*_a, **_k):
        raise model.ProviderError("HTTP 429", 429)

    monkeypatch.setattr(model, "post_json", fake_post)
    with pytest.raises(model.ProviderError, match="429"):
        model.post_decision(dict(BODY))


def test_a_direct_call_is_charged_at_the_published_rate(monkeypatch):
    """The direct API reports tokens, not money; a mixed run would otherwise under-report."""
    monkeypatch.setenv("JEV_MAX_COST_USD", "100")
    monkeypatch.delenv("JEV_REQUIRE_FREE", raising=False)
    budget = Budget()
    amount = budget.estimate({"input_tokens": 1000}, "decision:direct")
    assert amount == pytest.approx(0.000042)
    assert budget.spent == pytest.approx(0.000042)


def test_require_free_still_trips_if_a_paid_call_slips_through(monkeypatch):
    monkeypatch.setenv("JEV_REQUIRE_FREE", "1")
    monkeypatch.setenv("JEV_MAX_COST_USD", "100")
    with pytest.raises(Blocked, match="JEV_REQUIRE_FREE"):
        Budget().estimate({"input_tokens": 1000}, "decision:direct")


def test_a_leg_with_a_successor_does_not_back_off(monkeypatch):
    """Waiting 30s to reach a failover that was available immediately is 30s wasted."""
    seen = []

    def fake_post(url, _key, _body, _headers=None, retries=None):
        seen.append((("gateway" if "ai-gateway" in url else "direct"), retries))
        if "ai-gateway" in url:
            raise model.ProviderError("HTTP 429", 429)
        return {"answers": {}}

    monkeypatch.setattr(model, "post_json", fake_post)
    model.post_decision(dict(BODY))
    assert seen == [("gateway", 1), ("direct", None)], seen
