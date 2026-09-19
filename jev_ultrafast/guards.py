"""Run-level guards: what the agent may not click, and what a run may not spend.

The policy chooses from whatever the page offers, so a live site offers it delete
buttons and checkout flows alongside the navigation the goal actually needs. These
guards refuse the action instead of trusting the choice, and stop the run when the
Gateway's own cost accounting passes a budget.
"""

import datetime
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


# Jev is free on the Gateway as a promotion, not as a price. Vercel announced it through
# 24 Sept 2026; after that the same call starts costing money on the same key, silently.
GATEWAY_FREE_THROUGH = "2026-09-24"


def gateway_still_free(today=None):
    through = os.environ.get("JEV_GATEWAY_FREE_THROUGH", GATEWAY_FREE_THROUGH).strip()
    if through.lower() in {"", "never", "off"}:
        return True
    today = today or datetime.date.today()
    return today <= datetime.date.fromisoformat(through)


def paid_gateway_allowed():
    return os.environ.get("JEV_ALLOW_PAID_GATEWAY", "").strip() in {"1", "true", "yes", "on"}


class Blocked(RuntimeError):
    """A guard refused the action. The run stops; nothing was executed."""


def mode():
    return os.environ.get("JEV_GUARD", "readonly").strip().lower()


# Only these reach the page. WAIT, SCROLL_UP and SCROLL_DOWN are Jev's own controls and
# carry descriptive labels -- "Wait for the page to update" matched the write pattern on
# the word "update" and blocked a run that had touched nothing.
PAGE_ACTIONS = {"click", "fill", "select"}


def check_action(action):
    """Raise before Browser.act() touches the page. Navigation and typing stay allowed."""
    current = mode()
    if current == "off" or action["kind"] not in PAGE_ACTIONS:
        return
    label = " ".join(str(action.get(k, "") or "") for k in ("label", "value", "current_value"))
    if FORBIDDEN.search(label):
        raise Blocked(f"Guard refused {action['kind']} on {action['label']!r}: irreversible, auth, or billing.")
    if current == "readonly" and WRITES.search(label):
        raise Blocked(f"Guard refused {action['kind']} on {action['label']!r}: writes in readonly mode.")


def require_free():
    return os.environ.get("JEV_REQUIRE_FREE", "").strip() in {"1", "true", "yes", "on"}


class Budget:
    """Cumulative spend for one run, read from the Gateway's own cost accounting."""

    def __init__(self):
        self.spent = 0.0
        self.calls = []

    @property
    def limit(self):
        return float(os.environ.get("JEV_MAX_COST_USD", "0.25"))

    # The Gateway's listed market rate for typesafe-ai/jev. The direct API returns usage
    # but no cost, so a mixed run would otherwise report only half its spend.
    DIRECT_INPUT_USD_PER_TOKEN = 0.000000042

    def estimate(self, usage, label):
        """Charge a direct-API call at the published rate; it reports tokens, not money."""
        tokens = (usage or {}).get("input_tokens") or 0
        return self.charge(tokens * self.DIRECT_INPUT_USD_PER_TOKEN, label)

    def charge(self, amount, label):
        self.spent += amount
        self.calls.append({"label": label, "cost": amount})
        if amount and require_free():
            raise Blocked(
                f"JEV_REQUIRE_FREE is set but the {label} call cost ${amount:.8f}. "
                "The Gateway promotion has ended or this model was never free."
            )
        if self.spent > self.limit:
            raise Blocked(f"Run spent ${self.spent:.6f}, over the ${self.limit:.2f} JEV_MAX_COST_USD budget.")
        return amount

    def record(self, response, label):
        """The date is the announcement; this is the truth. A promo ending early trips here first."""
        cost = response.get("providerMetadata", {}).get("gateway", {}).get("cost")
        if cost is None:
            message = response.get("choices", [{}])[0].get("message", {})
            cost = message.get("provider_metadata", {}).get("gateway", {}).get("cost")
        return self.charge(float(cost or 0), label)


BUDGET = Budget()
