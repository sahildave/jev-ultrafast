"""Offline contracts for the action and budget guards. No paid APIs."""

import pytest

from jev_ultrafast import guards, model
from jev_ultrafast.guards import Blocked, Budget, check_action


def act(label, kind="click", **extra):
    return {"kind": kind, "id": "a1", "label": label, **extra}


@pytest.mark.parametrize(
    "label",
    ["Delete board", "Sign in with GitHub", "Continue with Google", "Subscribe", "Checkout", "Revoke access"],
)
def test_irreversible_auth_and_billing_are_refused_even_in_write_mode(label, monkeypatch):
    monkeypatch.setenv("JEV_GUARD", "write")
    with pytest.raises(Blocked, match="irreversible, auth, or billing"):
        check_action(act(label))


@pytest.mark.parametrize("label", ["Post comment", "Upvote", "Create issue", "Submit feedback"])
def test_writes_are_refused_in_the_default_readonly_mode(label, monkeypatch):
    monkeypatch.delenv("JEV_GUARD", raising=False)
    with pytest.raises(Blocked, match="writes in readonly mode"):
        check_action(act(label))


@pytest.mark.parametrize("label", ["Post comment", "Upvote"])
def test_write_mode_allows_writes(label, monkeypatch):
    monkeypatch.setenv("JEV_GUARD", "write")
    check_action(act(label))


@pytest.mark.parametrize("label", ["Roadmap", "Changelog", "Open a board for a repository", "Search"])
def test_navigation_and_search_stay_allowed(label, monkeypatch):
    monkeypatch.delenv("JEV_GUARD", raising=False)
    check_action(act(label))
    check_action(act(label, kind="fill"))


def test_a_guarded_label_hiding_in_the_field_value_is_still_caught(monkeypatch):
    monkeypatch.delenv("JEV_GUARD", raising=False)
    with pytest.raises(Blocked):
        check_action(act("Confirm", current_value="delete this board"))


@pytest.mark.parametrize("kind", ["wait", "scroll_down", "scroll_up"])
def test_jevs_own_controls_are_never_guarded(kind, monkeypatch):
    """'Wait for the page to update' matched the write pattern and blocked a clean run."""
    monkeypatch.delenv("JEV_GUARD", raising=False)
    check_action(act("Wait for the page to update", kind=kind))


def test_off_disables_both_guards(monkeypatch):
    monkeypatch.setenv("JEV_GUARD", "off")
    check_action(act("Delete board"))


def test_budget_sums_the_gateway_cost_and_stops_the_run(monkeypatch):
    monkeypatch.setenv("JEV_MAX_COST_USD", "0.001")
    budget = Budget()
    decision = {"providerMetadata": {"gateway": {"cost": "0.0004"}}}
    assert budget.record(decision, "decision") == 0.0004
    budget.record(decision, "decision")
    with pytest.raises(Blocked, match="over the .0.00 JEV_MAX_COST_USD budget"):
        budget.record(decision, "decision")
    assert budget.spent == pytest.approx(0.0012)


def test_budget_reads_the_text_helper_cost_from_its_own_envelope(monkeypatch):
    monkeypatch.setenv("JEV_MAX_COST_USD", "1")
    budget = Budget()
    text = {"choices": [{"message": {"provider_metadata": {"gateway": {"cost": "0.00002062"}}}}]}
    assert budget.record(text, "text") == pytest.approx(0.00002062)


def test_a_free_model_reporting_zero_never_trips_the_budget(monkeypatch):
    monkeypatch.setenv("JEV_MAX_COST_USD", "0")
    budget = Budget()
    for _ in range(50):
        budget.record({"providerMetadata": {"gateway": {"cost": "0"}}}, "decision")
    assert budget.spent == 0


def test_the_agent_checks_every_action_before_executing_it():
    """The guard must sit before Browser.act, not after; a refusal that runs first is not a guard."""
    source = (guards.__file__.replace("guards.py", "agent.py"))
    with open(source) as handle:
        body = handle.read()
    assert body.index("check_action(action)") < body.index('state["browser"].act(')


def test_click_only_withholds_the_text_and_select_heads(monkeypatch):
    """An unoffered operation cannot be chosen, so the run can never need a second model."""
    monkeypatch.setenv("JEV_CLICK_ONLY", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")  # choose() still resolves a route
    captured = {}

    def fake_post(_url, _key, body, _headers=None, retries=None):
        captured["questions"] = body["questions"]
        return {"answers": {"operation": {"type": "choice", "choice": "DONE",
                                          "probabilities": {"CLICK": 0.0, "DONE": 1.0, "BLOCKED": 0.0},
                                          "confidence": 1.0}}}

    monkeypatch.setattr(model, "post_json", fake_post)
    state = {
        "url": "u", "title": "t", "text": "x",
        "actions": [
            {"kind": "click", "id": "c1", "node": 1, "role": "link", "label": "Roadmap"},
            {"kind": "fill", "id": "f1", "node": 2, "role": "textbox", "label": "Repository"},
            {"kind": "select", "id": "s1", "node": 3, "role": "combobox", "label": "Sort", "value": "new"},
        ],
    }
    model.choose(state, "look around", [])
    operations = captured["questions"]["operation"]["criteria"]
    assert "TYPE_TEXT" not in operations and "SELECT" not in operations
    assert "type_text_target" not in captured["questions"]
    assert "CLICK" in operations


def test_click_only_refuses_the_text_helper_outright(monkeypatch):
    monkeypatch.setenv("JEV_CLICK_ONLY", "1")
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "would-have-worked")
    with pytest.raises(Blocked, match="refusing to call the text helper"):
        model.field_text({"goal": "g"})


def test_pacing_spaces_consecutive_requests(tmp_path, monkeypatch):
    """The gap must sit on every request; two decisions inside one task are milliseconds apart."""
    slept = []
    monkeypatch.setenv("JEV_MIN_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("JEV_PACE_FILE", str(tmp_path / "last"))
    monkeypatch.setattr(model.time, "time", lambda: 1000.0)
    monkeypatch.setattr(model.time, "sleep", slept.append)
    model.pace()
    assert slept == [], "the first request is not delayed"
    model.pace()
    assert slept == [60.0], f"a request 0s after the last must wait the full gap, got {slept}"


def test_pacing_survives_a_fresh_process(tmp_path, monkeypatch):
    """A rerun seconds later is a new interpreter; without this it walks back into the limit."""
    slept = []
    monkeypatch.setenv("JEV_MIN_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("JEV_PACE_FILE", str(tmp_path / "last"))
    monkeypatch.setattr(model.time, "time", lambda: 1000.0)
    monkeypatch.setattr(model.time, "sleep", slept.append)
    model.pace()
    (tmp_path / "last").write_text("995.0")  # a previous process, five seconds ago
    model.pace()
    assert slept == [55.0], f"expected the remainder of the gap, got {slept}"


def test_pacing_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_MIN_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("JEV_PACE_FILE", str(tmp_path / "last"))
    monkeypatch.setattr(model.time, "sleep", lambda _s: pytest.fail("paced with no interval set"))
    model.pace()


@pytest.mark.parametrize("label", ["Log-out", "Sign-in", "Check-out", "force_push", "Clear  all"])
def test_hyphens_and_underscores_do_not_defeat_the_pattern(label, monkeypatch):
    """\\b-anchored words missed "Log-out" entirely."""
    monkeypatch.delenv("JEV_GUARD", raising=False)
    with pytest.raises(Blocked):
        check_action(act(label))


@pytest.mark.parametrize("label", ["Yes", "OK", "Confirm", "Proceed", "Continue", "I understand"])
def test_confirmation_controls_are_refused(label, monkeypatch):
    """The guard only ever sees the control that OPENS a dialog. This is the second half."""
    monkeypatch.delenv("JEV_GUARD", raising=False)
    with pytest.raises(Blocked, match="confirmation control"):
        check_action(act(label))


@pytest.mark.parametrize("label,role", [("button", "button"), ("", "button"), ("link", "link")])
def test_an_unlabelled_control_is_refused(label, role, monkeypatch):
    """snapshot.js falls back to the role name, so an icon-only button arrives as "button"."""
    monkeypatch.delenv("JEV_GUARD", raising=False)
    with pytest.raises(Blocked, match="nothing to check it against"):
        check_action(act(label, role=role))


@pytest.mark.parametrize("label", ["Card details", "Add card", "Watch the demo", "Star Wars"])
def test_ordinary_product_labels_are_not_refused(label, monkeypatch):
    """A board product has cards. Refusing "Card details" makes the guard useless on gitback."""
    monkeypatch.delenv("JEV_GUARD", raising=False)
    check_action(act(label, role="link"))


def test_text_a_user_typed_is_not_matched(monkeypatch):
    """A search box containing "billing" must stay clickable; its value is not a label."""
    monkeypatch.delenv("JEV_GUARD", raising=False)
    check_action(act("Search", kind="click", role="searchbox", current_value="billing questions"))


@pytest.mark.parametrize("label", ["Star", "Watch", "Follow", "Update"])
def test_ambiguous_verbs_are_refused_as_a_whole_label(label, monkeypatch):
    monkeypatch.delenv("JEV_GUARD", raising=False)
    with pytest.raises(Blocked, match="writes in readonly mode"):
        check_action(act(label, role="button"))
