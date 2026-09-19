"""The call sites, not the units. Every test here failed a mutation that the rest of the
suite passed: a cap or a guard can be correct in isolation and wired to nothing."""

import pytest

from jev_ultrafast import model
from jev_ultrafast.guards import Blocked, Budget

STATE = {
    "url": "u", "title": "t", "text": "x",
    "actions": [{"kind": "click", "id": "a1", "node": 1, "role": "link", "label": "Roadmap"}],
}
ANSWER = {
    "answers": {
        "operation": {"type": "choice", "choice": "CLICK",
                      "probabilities": {"CLICK": 1.0, "DONE": 0.0, "BLOCKED": 0.0}, "confidence": 1.0},
        "click_target": {"type": "choice", "choice": "1", "probabilities": {"1": 1.0}, "confidence": 1.0},
    },
    "usage": {"inputTokens": 1000},
}


def test_choose_charges_the_budget(monkeypatch):
    """Deleting the BUDGET call in choose() left the whole suite green."""
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw")
    monkeypatch.setenv("JEV_GATEWAY_FREE_THROUGH", "2099-01-01")
    monkeypatch.setenv("JEV_MAX_COST_USD", "0.0001")
    monkeypatch.setattr(model, "BUDGET", Budget())
    monkeypatch.setattr(model, "post_json",
                        lambda *_a, **_k: {**ANSWER, "providerMetadata": {"gateway": {"cost": "0.001"}}})
    with pytest.raises(Blocked, match="JEV_MAX_COST_USD"):
        model.choose(STATE, "go", [])


def test_a_direct_decision_is_charged_too(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts")
    monkeypatch.setenv("JEV_MAX_COST_USD", "100")
    budget = Budget()
    monkeypatch.setattr(model, "BUDGET", budget)
    monkeypatch.setattr(model, "post_json", lambda *_a, **_k: ANSWER)
    model.choose(STATE, "go", [])
    assert budget.spent == pytest.approx(0.000042), "the direct leg must not be recorded as free"


def test_field_text_charges_the_budget(monkeypatch):
    """Deleting the BUDGET call in field_text() also left the suite green."""
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "tm")
    monkeypatch.setenv("JEV_MAX_COST_USD", "0.0001")
    monkeypatch.setattr(model, "BUDGET", Budget())
    monkeypatch.setattr(model, "post_json", lambda *_a, **_k: {
        "choices": [{"message": {"content": '{"text": "hello"}',
                                 "provider_metadata": {"gateway": {"cost": "0.001"}}}}]})
    with pytest.raises(Blocked, match="JEV_MAX_COST_USD"):
        model.field_text({"goal": "g"})


def test_route_gateway_never_touches_the_billed_api_under_a_rate_limit(monkeypatch):
    """The case that costs money: both keys present, the free leg throttled."""
    monkeypatch.setenv("JEV_ROUTE", "gateway")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts")
    monkeypatch.setenv("JEV_GATEWAY_FREE_THROUGH", "2099-01-01")
    tried = []

    def fake_post(url, *_a, **_k):
        tried.append(url)
        raise model.ProviderError("HTTP 429", 429)

    monkeypatch.setattr(model, "post_json", fake_post)
    with pytest.raises(model.ProviderError):
        model.post_decision({"model": "m", "state": {}, "questions": {}})
    assert not any("typesafe.ai" in url for url in tried), tried


def test_post_json_paces(monkeypatch):
    """The pacing tests called pace() directly, so the call site was never pinned."""
    monkeypatch.setattr(model, "pace", lambda: (_ for _ in ()).throw(RuntimeError("paced")))
    with pytest.raises(RuntimeError, match="paced"):
        model.post_json("https://example.invalid", "k", {})


def test_the_text_helper_reuses_the_gateway_key(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://ai-gateway.vercel.sh/v1")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")
    monkeypatch.setenv("JEV_MAX_COST_USD", "100")
    monkeypatch.setattr(model, "BUDGET", Budget())
    seen = {}

    def fake_post(_url, key, *_a, **_k):
        seen["key"] = key
        return {"choices": [{"message": {"content": '{"text": "v"}'}}]}

    monkeypatch.setattr(model, "post_json", fake_post)
    model.field_text({"goal": "g"})
    assert seen["key"] == "gw-key"


def test_route_direct_with_no_key_raises(monkeypatch):
    monkeypatch.setenv("JEV_ROUTE", "direct")
    with pytest.raises(Blocked, match="TYPESAFE_API_KEY"):
        model.decision_legs({"model": "m", "state": {}, "questions": {}})


def test_a_typo_in_the_route_is_not_silently_auto(monkeypatch):
    """JEV_ROUTE=Gateway used to mean auto, which bills the direct API."""
    monkeypatch.setenv("JEV_ROUTE", "getaway")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts")
    with pytest.raises(Blocked, match="not one of auto, gateway, direct"):
        model.decision_legs({"model": "m", "state": {}, "questions": {}})


def test_a_response_with_no_cost_field_is_not_assumed_free(monkeypatch):
    monkeypatch.setenv("JEV_REQUIRE_FREE", "1")
    monkeypatch.setenv("JEV_MAX_COST_USD", "100")
    with pytest.raises(Blocked, match="Unknown is not free"):
        Budget().record({"answers": {}, "usage": {"inputTokens": 10}}, "decision:gateway")


def test_tuning_env_is_read_after_dotenv_is_loaded(monkeypatch):
    """demo.py loads .env AFTER importing model, so import-time constants kept their
    defaults and JEV_MIN_INTERVAL_SECONDS=60 in .env did nothing under `uv run jev`."""
    monkeypatch.setenv("JEV_RETRIES", "9")
    monkeypatch.setenv("JEV_MIN_INTERVAL_SECONDS", "42")
    assert model.default_retries() == 9
    assert model.min_interval_seconds() == 42.0


def test_each_agent_starts_with_a_clean_budget(monkeypatch):
    """JEV_MAX_COST_USD is a per-run cap, and the inspector builds an Agent per run in one
    process. Without the reset the second run begins already spent."""
    from jev_ultrafast import agent as loop
    from jev_ultrafast.guards import BUDGET

    class FakeBrowser:
        def __init__(self, _url):
            pass

        def observe(self, screenshot=False):
            return {"url": "u", "title": "t", "text": "", "actions": [], "fingerprint": "f"}

    monkeypatch.setenv("JEV_MAX_COST_USD", "100")
    monkeypatch.setattr(loop, "Browser", FakeBrowser)
    BUDGET.charge(0.10, "a previous run")
    assert BUDGET.spent == pytest.approx(0.10)
    loop.Agent("https://example.invalid", "look at it")
    assert BUDGET.spent == 0.0, "the new run inherited the previous run's spend"


def test_post_json_resolves_its_own_retry_count(monkeypatch):
    """Every other test stubs post_json, so its body was never executed. A parameter named
    `retries` shadowed the module function of the same name: retries() called None."""
    monkeypatch.setenv("JEV_MIN_INTERVAL_SECONDS", "0")
    with pytest.raises(model.ProviderError, match="connection failed"):
        model.post_json("http://127.0.0.1:1/unreachable", "k", {}, None, retries=None)
