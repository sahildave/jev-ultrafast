"""TypeSafe makes choices; an optional small OpenAI-compatible model writes field values."""

import json
import math
import os
import pathlib
import tempfile
import time

import httpx

from .guards import (
    BUDGET,
    GATEWAY_FREE_THROUGH,
    Blocked,
    click_only,
    gateway_still_free,
    paid_gateway_allowed,
    require_free,
)
from .questions import NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=25)


class ProviderError(RuntimeError):
    """A provider answered with an error. status carries the code so a leg can be retired."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


# The Gateway free tier answers 429 for tens of seconds, not the ~1s the original
# backoff assumed, so a browser run died two steps in. Tunable for paid accounts.
RETRIES = int(os.environ.get("JEV_RETRIES", "5"))
RETRY_BASE_SECONDS = float(os.environ.get("JEV_RETRY_BASE_SECONDS", "2"))


# Pacing beats backoff on a hard rate limit: spacing every request keeps the run under
# the limit instead of discovering it, then waiting, then discovering it again. The gap
# has to sit here rather than between tasks -- two decisions inside one task are
# milliseconds apart, and that pair is what trips the free tier.
MIN_INTERVAL_SECONDS = float(os.environ.get("JEV_MIN_INTERVAL_SECONDS", "0"))
# The clock has to outlive the process. A rerun a few seconds after the last run is a
# fresh interpreter with no memory of it, so in-process pacing fires immediately and
# walks straight back into the limit -- which is exactly how the board retry died.
PACE_FILE = pathlib.Path(os.environ.get("JEV_PACE_FILE", tempfile.gettempdir() + "/jev-last-request"))


def _last_request_at():
    try:
        return float(PACE_FILE.read_text())
    except (OSError, ValueError):
        return None


def pace():
    if MIN_INTERVAL_SECONDS:
        previous = _last_request_at()
        # time.time(), not monotonic(): monotonic is meaningless across processes.
        wait = 0 if previous is None else previous + MIN_INTERVAL_SECONDS - time.time()
        if wait > 0:
            time.sleep(wait)
    try:
        PACE_FILE.write_text(str(time.time()))
    except OSError:
        pass


def post_json(url, key, body, extra_headers=None, retries=None):
    pace()
    retries = RETRIES if retries is None else retries
    for attempt in range(retries):
        try:
            headers = {"Authorization": f"Bearer {key}", **(extra_headers or {})}
            response = CLIENT.post(url, json=body, headers=headers)
        except httpx.HTTPError:
            raise ProviderError("Model connection failed; no action executed.", None) from None
        if response.status_code in {429, 529, 503} and attempt < retries - 1:
            time.sleep(RETRY_BASE_SECONDS * 2**attempt)
            continue
        if response.is_error:
            # A bare status code hides the one thing that tells you what to do next -- a free-tier
            # rate limit and an expired key both read as "HTTP 429" / "HTTP 401" otherwise.
            try:
                detail = response.json().get("error", {}).get("message", "")
            except ValueError:
                detail = ""
            detail = f" {detail}" if detail else ""
            raise ProviderError(
                f"Model provider returned HTTP {response.status_code};{detail} no action executed.",
                response.status_code,
            )
        return response.json()
    raise RuntimeError("Model unavailable")


GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v4/ai"
GATEWAY_PROTOCOL_VERSION = "0.0.1"


def _gateway_leg(body):
    key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
    if not key:
        return None
    if not gateway_still_free() and not paid_gateway_allowed():
        return None
    model = os.environ.get("GATEWAY_MODEL", "typesafe-ai/jev")
    return {
        "name": "gateway",
        "url": GATEWAY_BASE_URL + "/evaluation-model",
        "key": key,
        # TYPESAFE_MODEL holds a direct-API id such as jev-latest; the Gateway wants its own slug.
        "body": {k: v for k, v in body.items() if k != "model"},
        "headers": {
            "ai-gateway-protocol-version": GATEWAY_PROTOCOL_VERSION,
            "ai-gateway-auth-method": "api-key",
            "ai-evaluation-model-specification-version": "4",
            "ai-model-id": model,
        },
    }


def _direct_leg(body):
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        return None
    return {"name": "direct", "url": "https://api.typesafe.ai/v1/systemone", "key": key,
            "body": body, "headers": None}


def decision_legs(body):
    """Legs to try in order. Measured: the Gateway answers in ~0.57s against the direct
    API's ~0.87s, but only for a burst before the free tier throttles. Free-and-fast
    first, paid-and-reliable behind it, so a run neither stalls nor pays when it need not.
    """
    route = os.environ.get("JEV_ROUTE", "auto").strip()
    gateway, direct = _gateway_leg(body), _direct_leg(body)

    if route == "gateway":
        if not gateway:
            raise Blocked(
                "JEV_ROUTE=gateway, but the Gateway leg is unavailable: either "
                "AI_GATEWAY_API_KEY is empty, or the promotion ended after "
                f"{os.environ.get('JEV_GATEWAY_FREE_THROUGH', GATEWAY_FREE_THROUGH)} and "
                "JEV_ALLOW_PAID_GATEWAY is not set. Refusing to bill the direct API instead."
            )
        return [gateway]
    if route == "direct":
        if not direct:
            raise Blocked("JEV_ROUTE=direct but TYPESAFE_API_KEY is empty.")
        return [direct]

    # JEV_REQUIRE_FREE means free, and the direct API is not. Failing over would bill
    # the key the setting exists to protect.
    legs = [leg for leg in (gateway, None if require_free() else direct) if leg]
    if not legs:
        raise Blocked(
            "No usable route. Set AI_GATEWAY_API_KEY (free through "
            f"{os.environ.get('JEV_GATEWAY_FREE_THROUGH', GATEWAY_FREE_THROUGH)}) "
            "or TYPESAFE_API_KEY, and unset JEV_REQUIRE_FREE to allow the paid fallback."
        )
    return legs


def normalise_usage(usage):
    renamed = {"inputTokens": "input_tokens", "outputTokens": "output_tokens"}
    return {renamed.get(k, k): v for k, v in (usage or {}).items()}


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        answer.setdefault("confidence", max(probabilities.values(), default=0))
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    return answer


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


FAILOVER_STATUSES = {429, 500, 502, 503, 529, None}


def post_decision(body):
    """Walk the legs, retiring one only for a limit or an outage. A 400 is our bug on
    every leg, so failing over would just spend a second provider to learn the same thing.
    """
    legs = decision_legs(body)
    last = None
    for index, leg in enumerate(legs):
        # Backing off on a leg that has another behind it is 30s spent to reach the same
        # failover. Only the last leg, which has nowhere to go, is worth waiting on.
        last_leg = index == len(legs) - 1
        try:
            return post_json(leg["url"], leg["key"], leg["body"], leg["headers"],
                             retries=None if last_leg else 1), leg["name"]
        except ProviderError as error:
            if error.status not in FAILOVER_STATUSES or last_leg:
                raise
            last = error
    raise last


def choose(state, goal, history):
    actions = state["actions"]
    if click_only():
        # Withholding the targets is what keeps this honest: an unoffered head cannot be chosen,
        # so the policy never reaches a TYPE_TEXT it would need a second model to complete.
        actions = [a for a in actions if a["kind"] not in {"fill", "select"}]
    elements, targets, controls = action_space(actions)
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    body = {
        "model": os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    started = time.perf_counter()
    result, leg = post_decision(body)
    if leg == "direct":
        BUDGET.estimate(normalise_usage(result.get("usage")), "decision:direct")
    else:
        BUDGET.record(result, f"decision:{leg}")
    for question, value in result.get("providerMetadata", {}).get("typesafe", {}).get("confidence", {}).items():
        if question in result.get("answers", {}):
            result["answers"][question]["confidence"] = value
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        target_answer = validate_choice(result["answers"].get(operation.lower() + "_target", {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result.get("model") or os.environ.get("GATEWAY_MODEL", "typesafe-ai/jev"),
        "leg": leg,
        "usage": normalise_usage(result.get("usage")),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_text(context):
    if click_only():
        raise Blocked("JEV_CLICK_ONLY is set; refusing to call the text helper model.")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    key = os.environ.get("TEXT_MODEL_API_KEY", "").strip()
    if not key and "ai-gateway.vercel.sh" in base:
        # One Gateway key serves both heads; no reason to paste the same secret twice.
        key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
    if not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor.")
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions",
        key,
        {
            "model": model,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            **reasoning,
            "messages": [
                {"role": "system", "content": TEXT_VALUE},
                {
                    "role": "user",
                    "content": json.dumps(context),
                },
            ],
        },
    )
    BUDGET.record(result, "text")
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }
