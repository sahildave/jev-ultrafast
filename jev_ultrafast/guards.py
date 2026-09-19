"""Run-level guards: what the agent may not click, and what a run may not spend.

The policy chooses from whatever the page offers, so a live site offers it delete
buttons and checkout flows alongside the navigation the goal actually needs. These
guards refuse the action instead of trusting the choice, and stop the run when the
Gateway's own cost accounting passes a budget.
"""

import datetime
import os
import re

# NOT A SAFETY BOUNDARY. This is an English-language denylist over accessible labels.
# It cannot see intent, it does not speak other languages, and it only sees the control
# that OPENS a confirm dialog -- the "Yes" on the second step is a separate action. Treat
# it as a seatbelt on a run you are supervising, not as authorisation to point the agent
# at something you cannot afford to have clicked.
FORBIDDEN = re.compile(
    r"\b(delete|destroy|remove|erase|trash|discard|wipe|clear all|reset|"
    r"revoke|deactivate|close account|unsubscribe|terminate|ban|block|leave|"
    r"sign out|log out|logout|sign in|log in|login|sign up|register|continue with|"
    r"authorize|authorise|grant|regenerate|rotate|make public|"
    r"install|uninstall|connect|disconnect|subscribe|upgrade|downgrade|"
    r"checkout|check out|buy|purchase|pay|payment|billing|order|donate|bid|withdraw|"
    r"transfer|archive|merge|close issue|reopen|deploy|rollback|force push)\b",
    re.I,
)

# Confirmation controls. Harmless on their own, and the second half of every destructive
# flow -- the dialog's "Yes" carries none of the words above.
CONFIRM = re.compile(r"^\s*(yes|ok|okay|confirm|proceed|continue|accept|agree|i understand|got it)\b", re.I)

# Editable roles hold whatever the user typed; matching their contents means a search box
# containing the word "billing" becomes unclickable.
EDITABLE_ROLES = {"textbox", "searchbox", "combobox", "spinbutton"}

# Publishes content or changes someone else's data. Refused unless JEV_GUARD=write.
WRITES = re.compile(
    r"\b(comment|reply|post|vote|upvote|downvote|"
    r"create|new issue|submit|send|publish|save|rename|assign)\b",
    re.I,
)

# Ordinary English that happens to name a control: "Watch" is a button, "Watch the demo"
# is a link, and "Star Wars" is neither. Matched only as the WHOLE label, because as a
# substring they refuse most of a real product's navigation.
WRITE_EXACT = re.compile(r"^(star|unstar|watch|unwatch|follow|unfollow|like|update|label)$", re.I)


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
    today = today or datetime.datetime.now(datetime.timezone.utc).date()
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


def normalise(text):
    """Hyphens and underscores defeat a \\b-anchored pattern: "Log-out" is not "log out"."""
    return re.sub(r"[\s\-_]+", " ", str(text or "")).strip()


def check_action(action):
    """Raise before Browser.act() touches the page. Navigation and typing stay allowed."""
    current = mode()
    if current == "off" or action["kind"] not in PAGE_ACTIONS:
        return
    role = action.get("role", "")
    fields = ["label"] if role in EDITABLE_ROLES else ["label", "value", "current_value"]
    label = normalise(" ".join(str(action.get(k, "") or "") for k in fields))
    name = normalise(action.get("label"))

    if FORBIDDEN.search(label):
        raise Blocked(f"Guard refused {action['kind']} on {action['label']!r}: irreversible, auth, or billing.")
    if current == "readonly":
        if CONFIRM.match(label):
            raise Blocked(
                f"Guard refused {action['kind']} on {action['label']!r}: a confirmation control. "
                "The guard cannot see what it confirms."
            )
        # snapshot.js falls back to the role name when a control has no accessible name, so
        # an icon-only button arrives labelled "button" and every pattern above misses it.
        if not name or name.lower() == str(role).lower():
            raise Blocked(
                f"Guard refused {action['kind']} on an unlabelled {role or 'control'}: "
                "nothing to check it against."
            )
        if WRITES.search(label) or WRITE_EXACT.match(name):
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

    # Vercel's listed marketCost for typesafe-ai/jev, NOT TypeSafe's own price list --
    # check a direct-route figure against your TypeSafe dashboard before budgeting on it.
    # Input tokens only, so this is a floor: the direct API bills output too and reports
    # neither. The direct API returns usage and no cost, so without this a mixed run
    # reports only the half the Gateway served.
    GATEWAY_LISTED_INPUT_USD_PER_TOKEN = 0.000000042

    def estimate(self, usage, label):
        """Charge a direct-API call at the published rate; it reports tokens, not money."""
        tokens = (usage or {}).get("input_tokens") or 0
        return self.charge(tokens * self.GATEWAY_LISTED_INPUT_USD_PER_TOKEN, label)

    def charge(self, amount, label):
        """Charged after the response arrives, so a run overshoots the cap by at most one
        call -- one paid call when the trip is on a direct or text leg."""
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
        if cost is None and require_free():
            raise Blocked(
                f"JEV_REQUIRE_FREE is set but the {label} response carried no cost field. "
                "Unknown is not free; the provider's schema or pricing may have changed."
            )
        return self.charge(float(cost or 0), label)


    def reset(self):
        """A process that runs several goals -- the inspector does -- must not carry spend
        from the first into the cap of the second."""
        self.spent = 0.0
        self.calls = []
        return self


BUDGET = Budget()
