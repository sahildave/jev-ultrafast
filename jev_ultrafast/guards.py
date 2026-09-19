"""Run-level guards: what the agent may not click, and what a run may not spend.

The policy chooses from whatever the page offers, so a live site offers it delete
buttons and checkout flows alongside the navigation the goal actually needs. These
guards refuse the action instead of trusting the choice, and stop the run when the
Gateway's own cost accounting passes a budget.
"""

import os
import re

# Irreversible, authenticating, or billable. Refused on every run.
FORBIDDEN = re.compile(
    r"\b(delete|destroy|remove|revoke|deactivate|close account|unsubscribe|"
    r"sign out|log out|logout|sign in|log in|login|continue with|authorize|authorise|"
    r"install|uninstall|connect|disconnect|subscribe|upgrade|downgrade|"
    r"checkout|check out|buy|purchase|pay|billing|payment|card|"
    r"transfer|archive|merge|close issue|reopen)\b",
    re.I,
)

# Publishes content or changes someone else's data. Refused unless JEV_GUARD=write.
WRITES = re.compile(
    r"\b(comment|reply|post|vote|upvote|downvote|like|star|watch|follow|"
    r"create|new issue|submit|send|publish|save|update|rename|assign|label)\b",
    re.I,
)


def click_only():
    """JEV_CLICK_ONLY keeps the run on Jev alone: no operation that needs the text helper."""
    return os.environ.get("JEV_CLICK_ONLY", "").strip() in {"1", "true", "yes", "on"}


class Blocked(RuntimeError):
    """A guard refused the action. The run stops; nothing was executed."""


def mode():
    return os.environ.get("JEV_GUARD", "readonly").strip().lower()


def check_action(action):
    """Raise before Browser.act() touches the page. Navigation and typing stay allowed."""
    current = mode()
    if current == "off":
        return
    label = " ".join(str(action.get(k, "") or "") for k in ("label", "value", "current_value"))
    if FORBIDDEN.search(label):
        raise Blocked(f"Guard refused {action['kind']} on {action['label']!r}: irreversible, auth, or billing.")
    if current == "readonly" and WRITES.search(label):
        raise Blocked(f"Guard refused {action['kind']} on {action['label']!r}: writes in readonly mode.")


class Budget:
    """Cumulative spend for one run, read from the Gateway's own cost accounting."""

    def __init__(self):
        self.spent = 0.0
        self.calls = []

    @property
    def limit(self):
        return float(os.environ.get("JEV_MAX_COST_USD", "0.25"))

    def record(self, response, label):
        cost = response.get("providerMetadata", {}).get("gateway", {}).get("cost")
        if cost is None:
            message = response.get("choices", [{}])[0].get("message", {})
            cost = message.get("provider_metadata", {}).get("gateway", {}).get("cost")
        amount = float(cost or 0)
        self.spent += amount
        self.calls.append({"label": label, "cost": amount})
        if self.spent > self.limit:
            raise Blocked(f"Run spent ${self.spent:.6f}, over the ${self.limit:.2f} JEV_MAX_COST_USD budget.")
        return amount


BUDGET = Budget()
