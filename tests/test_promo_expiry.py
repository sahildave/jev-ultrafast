"""The Gateway promotion ends on a date; the key does not. Offline, no paid APIs."""

import datetime

import pytest

from jev_ultrafast import model
from jev_ultrafast.guards import Blocked, Budget, gateway_still_free

BODY = {"model": "jev-latest", "state": {}, "questions": {}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("JEV_ROUTE", "JEV_ALLOW_PAID_GATEWAY", "JEV_GATEWAY_FREE_THROUGH", "JEV_REQUIRE_FREE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


def test_the_promotion_window_is_a_date_not_a_guess():
    assert gateway_still_free(datetime.date(2026, 9, 24)) is True
    assert gateway_still_free(datetime.date(2026, 9, 25)) is False


def test_after_the_promotion_with_no_other_key_there_is_no_route(monkeypatch):
    monkeypatch.setenv("JEV_GATEWAY_FREE_THROUGH", "2020-01-01")
    with pytest.raises(Blocked, match="No usable route"):
        model.decision_legs(dict(BODY))


def test_after_the_promotion_a_typesafe_key_takes_over(monkeypatch):
    monkeypatch.setenv("JEV_GATEWAY_FREE_THROUGH", "2020-01-01")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    leg = model.decision_legs(dict(BODY))[0]
    url, key, body, headers = leg["url"], leg["key"], leg["body"], leg["headers"]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert (key, headers, body["model"]) == ("ts-key", None, "jev-latest")


def test_route_gateway_does_not_silently_become_the_direct_api(monkeypatch):
    """JEV_ROUTE=gateway means gateway. Falling back would bill the key it was set to protect."""
    monkeypatch.setenv("JEV_GATEWAY_FREE_THROUGH", "2020-01-01")
    monkeypatch.setenv("JEV_ROUTE", "gateway")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    with pytest.raises(Blocked, match="Refusing to bill the direct API"):
        model.decision_legs(dict(BODY))


def test_paying_for_the_gateway_is_an_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("JEV_GATEWAY_FREE_THROUGH", "2020-01-01")
    monkeypatch.setenv("JEV_ALLOW_PAID_GATEWAY", "1")
    leg = model.decision_legs(dict(BODY))[0]
    url, headers = leg["url"], leg["headers"]
    assert url.endswith("/evaluation-model") and headers["ai-model-id"] == "typesafe-ai/jev"


def test_during_the_promotion_nothing_changes(monkeypatch):
    monkeypatch.setenv("JEV_GATEWAY_FREE_THROUGH", "2099-01-01")
    url = model.decision_legs(dict(BODY))[0]["url"]
    assert url.endswith("/evaluation-model")


def test_require_free_trips_on_the_first_billed_call(monkeypatch):
    """A promotion that ends early ends without a changelog entry; the cost field says so first."""
    monkeypatch.setenv("JEV_REQUIRE_FREE", "1")
    monkeypatch.setenv("JEV_MAX_COST_USD", "100")
    budget = Budget()
    budget.record({"providerMetadata": {"gateway": {"cost": "0"}}}, "decision")
    with pytest.raises(Blocked, match="promotion has ended or this model was never free"):
        budget.record({"providerMetadata": {"gateway": {"cost": "0.000014658"}}}, "decision")
