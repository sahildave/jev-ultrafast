"""Offline contracts for the Vercel AI Gateway route. No paid APIs."""

import pytest

from jev_ultrafast import model

BODY = {"model": "jev-latest", "state": {"page": {}}, "questions": {"operation": {}}}


def test_without_a_gateway_key_the_direct_api_keeps_the_model_in_the_body(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("JEV_ROUTE", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    leg = model.decision_legs(dict(BODY))[0]
    url, key, body, headers = leg["url"], leg["key"], leg["body"], leg["headers"]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert (key, body["model"], headers) == ("ts-key", "jev-latest", None)


def test_a_gateway_key_moves_the_model_id_into_the_headers(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")
    monkeypatch.delenv("GATEWAY_MODEL", raising=False)
    leg = model.decision_legs(dict(BODY))[0]
    url, key, body, headers = leg["url"], leg["key"], leg["body"], leg["headers"]
    assert url == "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
    assert key == "gw-key"
    assert set(body) == {"state", "questions"}, "the Gateway rejects a model key in the body"
    assert headers == {
        "ai-gateway-protocol-version": "0.0.1",
        "ai-gateway-auth-method": "api-key",
        "ai-evaluation-model-specification-version": "4",
        "ai-model-id": "typesafe-ai/jev",
    }


def test_route_gateway_refuses_to_fall_back_to_the_billed_api(monkeypatch):
    monkeypatch.setenv("JEV_ROUTE", "gateway")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key-that-must-not-be-used")
    with pytest.raises(Exception, match="Refusing to bill the direct API"):
        model.decision_legs(dict(BODY))


def test_a_gateway_answer_without_confidence_still_validates():
    """The Gateway response schema has no confidence field; the pre-patch code raised here."""
    answer = model.validate_choice(
        {"type": "choice", "choice": "CLICK", "probabilities": {"CLICK": 0.9, "DONE": 0.1}},
        {"CLICK", "DONE"},
    )
    assert answer["confidence"] == 0.9


def test_provider_metadata_confidence_beats_the_derived_fallback(monkeypatch):
    """The Gateway carries the real confidence in providerMetadata.typesafe, not on the answer."""
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")
    response = {
        "answers": {
            "operation": {
                "type": "choice",
                "choice": "CLICK",
                "probabilities": {"CLICK": 0.9, "DONE": 0.1, "BLOCKED": 0.0},
            },
            "click_target": {"type": "choice", "choice": "1", "probabilities": {"1": 1.0}},
        },
        "providerMetadata": {"typesafe": {"confidence": {"operation": 0.62, "click_target": 0.55}}},
        "usage": {"inputTokens": 349, "outputTokens": 31},
    }
    monkeypatch.setattr(model, "post_json", lambda *_args, **_kw: response)
    state = {
        "url": "u",
        "title": "t",
        "text": "x",
        "actions": [{"kind": "click", "id": "a1", "node": 1, "role": "link", "label": "Learn more"}],
    }
    decision = model.choose(state, "open it", [])
    assert decision["confidence"] == 0.62, "0.9 would mean the derived fallback overrode the real value"
    assert decision["target_confidence"] == 0.55
    assert decision["usage"] == {"input_tokens": 349, "output_tokens": 31}
