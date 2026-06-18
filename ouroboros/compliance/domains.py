"""Compliance skill packs: domain-specific principles and MetaMorph skills.

Each :class:`CompliancePack` bundles:

- ``patterns``   – ``(regex, predicate)`` pairs registered with :class:`EthosCompiler`
                   *before* principles are loaded, so each principle resolves to
                   its intended predicate without modifying the shared registry.
- ``principles`` – natural language principle texts added to the EthosCompiler.
- ``skills``     – ``(name, fn)`` pairs registered in MetaMorph.

Call ``pack.load_into(loop)`` to install everything into a running
:class:`~ouroboros.ouroboros_loop.OuroborosLoop`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Helpers shared across predicates
# ---------------------------------------------------------------------------


def _intent_has(action: Any, *keywords: str) -> bool:
    """True when the action's intent contains at least one of *keywords*."""
    intent = str(action.get("intent", "")).lower()
    return any(kw in intent for kw in keywords)


# ---------------------------------------------------------------------------
# Healthcare skill implementations
# ---------------------------------------------------------------------------


def _skill_hipaa_check(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validate HIPAA compliance for an action touching health data."""
    phi = bool(params.get("exposes_pii", False))
    external = bool(params.get("shares_external", False))
    audited = bool(params.get("audit_logged", True))
    compliant = not phi and not external and audited
    violations: list[str] = []
    if phi:
        violations.append("PHI exposure detected")
    if external:
        violations.append("Unauthorized external sharing of health data")
    if not audited:
        violations.append("Patient data access must be audit logged")
    return {
        "skill": "hipaa_check",
        "intent": intent,
        "phi_exposure": phi,
        "external_share": external,
        "audit_logged": audited,
        "compliant": compliant,
        "violations": violations,
    }


def _skill_patient_consent(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """Check that patient consent has been obtained."""
    consent = bool(params.get("confirmed", False))
    patient_id = str(params.get("patient_id", "unknown"))
    return {
        "skill": "patient_consent",
        "intent": intent,
        "patient_id": patient_id,
        "consent_obtained": consent,
        "compliant": consent,
        "violations": [] if consent else ["Patient consent not obtained"],
    }


# ---------------------------------------------------------------------------
# Fintech skill implementations
# ---------------------------------------------------------------------------


def _skill_kyc_check(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validate KYC identity verification status."""
    verified = bool(params.get("confirmed", False))
    customer_id = str(params.get("customer_id", "unknown"))
    risk_level = str(params.get("risk_level", "standard"))
    return {
        "skill": "kyc_check",
        "intent": intent,
        "customer_id": customer_id,
        "identity_verified": verified,
        "risk_level": risk_level,
        "compliant": verified,
        "violations": [] if verified else ["KYC identity verification required"],
    }


def _skill_transaction_validator(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validate a financial transaction against configured limits."""
    amount = float(params.get("transaction_amount", 0.0))
    limit = float(params.get("approved_limit", 10_000.0))
    confirmed = bool(params.get("confirmed", False))
    within_limit = amount <= limit
    compliant = within_limit or confirmed
    violations: list[str] = []
    if not within_limit and not confirmed:
        violations.append(
            f"Transaction amount {amount} exceeds approved limit {limit}"
        )
    return {
        "skill": "transaction_validator",
        "intent": intent,
        "transaction_amount": amount,
        "approved_limit": limit,
        "within_limit": within_limit,
        "authorized": confirmed,
        "compliant": compliant,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Legal skill implementations
# ---------------------------------------------------------------------------


def _skill_privilege_check(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """Check for attorney-client or other legal privilege breaches."""
    external = bool(params.get("shares_external", False))
    privileged = bool(params.get("privileged", True))
    atty_client = bool(params.get("attorney_client", False))
    breach = external and (privileged or atty_client)
    return {
        "skill": "privilege_check",
        "intent": intent,
        "privileged_communication": privileged,
        "attorney_client": atty_client,
        "external_disclosure": external,
        "privilege_breach": breach,
        "compliant": not breach,
        "violations": ["Attorney-client privilege breach detected"] if breach else [],
    }


def _skill_jurisdiction_validator(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validate that jurisdiction is specified and legal actions are audited."""
    jurisdiction = str(params.get("jurisdiction", "unknown"))
    audited = bool(params.get("audit_logged", False))
    confirmed = bool(params.get("confirmed", False))
    specified = jurisdiction != "unknown"
    compliant = specified and audited
    violations: list[str] = []
    if not specified:
        violations.append("Jurisdiction not specified for legal action")
    if not audited:
        violations.append("Legal actions require audit logging")
    return {
        "skill": "jurisdiction_validator",
        "intent": intent,
        "jurisdiction": jurisdiction,
        "jurisdiction_specified": specified,
        "audit_logged": audited,
        "confirmed": confirmed,
        "compliant": compliant,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# CompliancePack dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompliancePack:
    """A bundle of domain-specific ethical principles and MetaMorph skills.

    Attributes:
        name:       Short identifier for the pack (e.g. ``"healthcare"``).
        principles: Natural language principle texts loaded into EthosCompiler.
        skills:     ``(name, fn)`` pairs registered in MetaMorph.
        patterns:   ``(regex, predicate)`` pairs registered with EthosCompiler
                    *before* principles so each principle compiles correctly.
    """

    name: str
    principles: tuple[str, ...]
    skills: tuple[tuple[str, Callable[..., Any]], ...]
    patterns: tuple[tuple[str, Callable[..., Any]], ...]

    def load_into(self, loop: Any) -> None:
        """Install this pack's patterns, principles, and skills into *loop*.

        Must be called on a fully constructed
        :class:`~ouroboros.ouroboros_loop.OuroborosLoop` (or any object
        exposing ``.ethos`` and ``.metamorph`` attributes).
        """
        if self.patterns:
            loop.ethos.register_patterns(list(self.patterns))
        for principle in self.principles:
            loop.ethos.add_principle(principle)
        for skill_name, fn in self.skills:
            loop.metamorph.register_skill(skill_name, fn)


# ---------------------------------------------------------------------------
# Healthcare pack
# ---------------------------------------------------------------------------

def _hc_phi(action: Any) -> bool:
    return not bool(action.get("exposes_pii", False))


def _hc_external(action: Any) -> bool:
    return not bool(action.get("shares_external", False))


def _hc_audit(action: Any) -> bool:
    relevant = _intent_has(action, "patient", "medical", "health", "clinical", "phi")
    return not relevant or bool(action.get("audit_logged", False))


def _hc_consent(action: Any) -> bool:
    relevant = _intent_has(action, "access", "read", "view", "retrieve", "patient", "medical")
    return not relevant or bool(action.get("confirmed", False))


HEALTHCARE = CompliancePack(
    name="healthcare",
    principles=(
        "Do not expose protected health information without patient consent.",
        "Never share patient data with external parties.",
        "Log all access to patient records for HIPAA compliance.",
        "Always obtain patient consent before accessing medical records.",
    ),
    skills=(
        ("hipaa_check", _skill_hipaa_check),
        ("patient_consent", _skill_patient_consent),
    ),
    patterns=(
        (r"do not expose protected health information", _hc_phi),
        (r"never share patient data with external", _hc_external),
        (r"log all access to patient records", _hc_audit),
        (r"always obtain patient consent", _hc_consent),
    ),
)


# ---------------------------------------------------------------------------
# Fintech pack
# ---------------------------------------------------------------------------

def _ft_transaction_limit(action: Any) -> bool:
    relevant = _intent_has(action, "transaction", "payment", "transfer", "wire", "send")
    if not relevant:
        return True
    amount = float(action.get("transaction_amount", 0.0))
    limit = float(action.get("approved_limit", 10_000.0))
    return amount <= limit or bool(action.get("confirmed", False))


def _ft_kyc(action: Any) -> bool:
    relevant = _intent_has(action, "transfer", "withdraw", "wire", "high-value")
    return not relevant or bool(action.get("confirmed", False))


def _ft_account_data(action: Any) -> bool:
    return not bool(action.get("shares_external", False))


def _ft_audit(action: Any) -> bool:
    relevant = _intent_has(action, "transaction", "payment", "transfer", "financial")
    return not relevant or bool(action.get("audit_logged", False))


FINTECH = CompliancePack(
    name="fintech",
    principles=(
        "Never process transactions exceeding approved limits without additional authorization.",
        "Always verify customer identity before high-value transfers.",
        "Never share account holder data externally without encryption.",
        "Log all financial transactions for regulatory audit requirements.",
    ),
    skills=(
        ("kyc_check", _skill_kyc_check),
        ("transaction_validator", _skill_transaction_validator),
    ),
    patterns=(
        (r"never process transactions exceeding", _ft_transaction_limit),
        (r"always verify customer identity", _ft_kyc),
        (r"never share account holder data externally", _ft_account_data),
        (r"log all financial transactions", _ft_audit),
    ),
)


# ---------------------------------------------------------------------------
# Legal pack
# ---------------------------------------------------------------------------

def _lg_privilege(action: Any) -> bool:
    return not bool(action.get("shares_external", False))


def _lg_filing(action: Any) -> bool:
    relevant = _intent_has(action, "file", "submit", "court", "legal", "filing")
    return not relevant or bool(action.get("confirmed", False))


def _lg_confidential(action: Any) -> bool:
    return not bool(action.get("shares_external", False))


def _lg_audit(action: Any) -> bool:
    relevant = _intent_has(action, "legal", "document", "contract", "case", "court")
    return not relevant or bool(action.get("audit_logged", False))


LEGAL = CompliancePack(
    name="legal",
    principles=(
        "Preserve attorney-client privilege; do not disclose confidential communications.",
        "Never file legal documents without prior authorization.",
        "Never share confidential case files with unauthorized parties.",
        "Log all legal document access and modifications for court compliance.",
    ),
    skills=(
        ("privilege_check", _skill_privilege_check),
        ("jurisdiction_validator", _skill_jurisdiction_validator),
    ),
    patterns=(
        (r"preserve attorney.client privilege", _lg_privilege),
        (r"never file legal documents without", _lg_filing),
        (r"never share confidential case files", _lg_confidential),
        (r"log all legal document access", _lg_audit),
    ),
)
