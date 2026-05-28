from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set


class RulesetLayer(int, Enum):
    BuiltinDefault = 0
    Agent = 1
    User = 2


@dataclass
class Ruleset:
    layer: RulesetLayer
    trusted_prefixes: List[str]
    denied_prefixes: List[str]

    @staticmethod
    def builtin_default() -> "Ruleset":
        return Ruleset(RulesetLayer.BuiltinDefault, [], [])

    @staticmethod
    def agent(trusted: List[str], denied: List[str]) -> "Ruleset":
        return Ruleset(RulesetLayer.Agent, trusted, denied)

    @staticmethod
    def user(trusted: List[str], denied: List[str]) -> "Ruleset":
        return Ruleset(RulesetLayer.User, trusted, denied)


class AskForApproval(str, Enum):
    UnlessTrusted = "unless_trusted"
    OnFailure = "on_failure"
    OnRequest = "on_request"
    Never = "never"
    Reject = "reject"


class NetworkPolicyRuleAction(str, Enum):
    Allow = "allow"
    Deny = "deny"


@dataclass
class NetworkPolicyAmendment:
    host: str
    action: NetworkPolicyRuleAction


@dataclass
class ExecPolicyAmendment:
    prefixes: List[str]


@dataclass
class ExecApprovalRequirement:
    kind: str
    reason: str
    bypass_sandbox: bool = False
    proposed_execpolicy_amendment: Optional[ExecPolicyAmendment] = None
    proposed_network_policy_amendments: List[NetworkPolicyAmendment] = None

    def phase(self) -> str:
        return self.kind


@dataclass
class ExecPolicyDecision:
    allow: bool
    requires_approval: bool
    requirement: ExecApprovalRequirement
    matched_rule: Optional[str]


@dataclass
class ExecPolicyContext:
    command: str
    cwd: str
    ask_for_approval: AskForApproval
    sandbox_mode: Optional[str] = None


class ExecPolicyEngine:
    def __init__(self, trusted_prefixes: Optional[List[str]] = None, denied_prefixes: Optional[List[str]] = None) -> None:
        self._rulesets: List[Ruleset] = []
        self._trusted_prefixes = trusted_prefixes or []
        self._denied_prefixes = denied_prefixes or []
        self._approved_for_session: Set[str] = set()

    def add_ruleset(self, ruleset: Ruleset) -> None:
        self._rulesets.append(ruleset)
        self._rulesets.sort(key=lambda rs: rs.layer)

    def remember_session_approval(self, approval_key: str) -> None:
        self._approved_for_session.add(approval_key)

    def is_session_approved(self, approval_key: str) -> bool:
        return approval_key in self._approved_for_session

    def _resolve_prefixes(self) -> tuple[list[str], list[str]]:
        if not self._rulesets:
            return list(self._trusted_prefixes), list(self._denied_prefixes)
        trusted: List[str] = []
        denied: List[str] = []
        for ruleset in self._rulesets:
            trusted.extend(ruleset.trusted_prefixes)
            denied.extend(ruleset.denied_prefixes)
        trusted.extend(self._trusted_prefixes)
        denied.extend(self._denied_prefixes)
        return trusted, denied

    def check(self, ctx: ExecPolicyContext) -> ExecPolicyDecision:
        normalized = _normalize(ctx.command)
        trusted_prefixes, denied_prefixes = self._resolve_prefixes()

        matched_deny = next(
            (rule for rule in denied_prefixes if normalized.startswith(_normalize(rule))),
            None,
        )
        if matched_deny is not None:
            requirement = ExecApprovalRequirement(
                kind="forbidden",
                reason=f"Command blocked by denied prefix rule '{matched_deny}'",
            )
            return ExecPolicyDecision(
                allow=False,
                requires_approval=False,
                requirement=requirement,
                matched_rule=matched_deny,
            )

        matched_trusted = next(
            (rule for rule in trusted_prefixes if normalized.startswith(_normalize(rule))),
            None,
        )
        is_trusted = matched_trusted is not None

        if ctx.ask_for_approval == AskForApproval.Never:
            requirement = ExecApprovalRequirement(kind="allowed", reason="Execution allowed by policy.")
        elif ctx.ask_for_approval == AskForApproval.UnlessTrusted and is_trusted:
            requirement = ExecApprovalRequirement(kind="allowed", reason="Execution allowed by policy.")
        elif ctx.ask_for_approval == AskForApproval.Reject:
            requirement = ExecApprovalRequirement(kind="forbidden", reason="Policy is configured to reject rule-exceptions.")
        else:
            amendment = None
            if not is_trusted:
                amendment = ExecPolicyAmendment(prefixes=[_first_token(ctx.command)])
            reason = "Approval requested by policy mode." if is_trusted else "Unmatched command prefix requires approval."
            requirement = ExecApprovalRequirement(
                kind="needs_approval",
                reason=reason,
                proposed_execpolicy_amendment=amendment,
                proposed_network_policy_amendments=[
                    NetworkPolicyAmendment(
                        host=ctx.cwd,
                        action=NetworkPolicyRuleAction.Allow,
                    )
                ],
            )

        allow = requirement.kind in ("allowed", "needs_approval")
        requires_approval = requirement.kind == "needs_approval"
        return ExecPolicyDecision(
            allow=allow,
            requires_approval=requires_approval,
            requirement=requirement,
            matched_rule=matched_trusted,
        )


def _normalize(value: str) -> str:
    return value.strip().lower()


def _first_token(command: str) -> str:
    return command.split(maxsplit=1)[0] if command.strip() else ""
