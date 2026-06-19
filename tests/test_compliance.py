"""Tests for the compliance skill packs (healthcare, fintech, legal)."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from ouroboros.compliance import FINTECH, HEALTHCARE, LEGAL, CompliancePack
from ouroboros.compliance.domains import (
    _skill_hipaa_check,
    _skill_jurisdiction_validator,
    _skill_kyc_check,
    _skill_patient_consent,
    _skill_privilege_check,
    _skill_transaction_validator,
)
from ouroboros.core import ProposedAction
from ouroboros.ethos_compiler import EthosCompiler
from ouroboros.metamorph import MetaMorph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLoop:
    """Minimal stand-in for OuroborosLoop, exposing .ethos and .metamorph."""

    def __init__(self) -> None:
        self.ethos = EthosCompiler()
        self.metamorph = MetaMorph()


def _loaded(pack: CompliancePack) -> _FakeLoop:
    """Return a FakeLoop with *pack* already installed."""
    loop = _FakeLoop()
    pack.load_into(loop)
    return loop


def _gate(pack: CompliancePack, action: dict[str, Any]):
    return _loaded(pack).ethos.gate(action)


# ---------------------------------------------------------------------------
# Healthcare – principle compilation
# ---------------------------------------------------------------------------


def test_healthcare_principles_compile_without_warnings() -> None:
    """All HEALTHCARE principles must match a registered pattern (no UserWarning)."""
    ethos = EthosCompiler()
    ethos.register_patterns(list(HEALTHCARE.patterns))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for p in HEALTHCARE.principles:
            ethos.add_principle(p)


# ---------------------------------------------------------------------------
# Healthcare – gate behaviour
# ---------------------------------------------------------------------------


def test_healthcare_blocks_phi_exposure() -> None:
    """An action that exposes PHI must be blocked."""
    result = _gate(
        HEALTHCARE,
        {"intent": "export patient record", "exposes_pii": True, "audit_logged": True},
    )
    assert not result.allowed


def test_healthcare_blocks_external_sharing() -> None:
    """Patient data shared externally must be blocked."""
    result = _gate(
        HEALTHCARE,
        {"intent": "sync patient files", "shares_external": True, "audit_logged": True},
    )
    assert not result.allowed


def test_healthcare_blocks_unaudited_patient_access() -> None:
    """Patient record access without audit logging must be blocked."""
    result = _gate(
        HEALTHCARE,
        {
            "intent": "access patient medical records",
            "exposes_pii": False,
            "shares_external": False,
            "audit_logged": False,
            "confirmed": True,
        },
    )
    assert not result.allowed


def test_healthcare_blocks_unconsented_patient_access() -> None:
    """Accessing patient data without consent (confirmed=False) must be blocked."""
    result = _gate(
        HEALTHCARE,
        {
            "intent": "read patient data",
            "exposes_pii": False,
            "shares_external": False,
            "audit_logged": True,
            "confirmed": False,
        },
    )
    assert not result.allowed


def test_healthcare_allows_compliant_access() -> None:
    """A fully compliant patient data access must be allowed."""
    result = _gate(
        HEALTHCARE,
        {
            "intent": "read patient record",
            "exposes_pii": False,
            "shares_external": False,
            "audit_logged": True,
            "confirmed": True,
        },
    )
    assert result.allowed, f"Unexpected violations: {result.violations}"


# ---------------------------------------------------------------------------
# Healthcare – skill functions
# ---------------------------------------------------------------------------


def test_hipaa_check_compliant() -> None:
    result = _skill_hipaa_check(
        "check patient data",
        {"exposes_pii": False, "shares_external": False, "audit_logged": True},
    )
    assert result["compliant"] is True
    assert result["violations"] == []


def test_hipaa_check_phi_violation() -> None:
    result = _skill_hipaa_check(
        "export records", {"exposes_pii": True, "audit_logged": True}
    )
    assert result["compliant"] is False
    assert len(result["violations"]) > 0


def test_hipaa_check_external_sharing_violation() -> None:
    result = _skill_hipaa_check(
        "sync records", {"exposes_pii": False, "shares_external": True, "audit_logged": True}
    )
    assert result["compliant"] is False


def test_patient_consent_granted() -> None:
    result = _skill_patient_consent(
        "access patient chart", {"confirmed": True, "patient_id": "P-0042"}
    )
    assert result["compliant"] is True
    assert result["patient_id"] == "P-0042"
    assert result["violations"] == []


def test_patient_consent_missing() -> None:
    result = _skill_patient_consent("access patient chart", {"confirmed": False})
    assert result["compliant"] is False
    assert len(result["violations"]) > 0


@pytest.mark.parametrize("name,fn", HEALTHCARE.skills)
def test_healthcare_skill_returns_dict(name: str, fn) -> None:
    assert isinstance(fn("probe", {}), dict), f"Skill {name!r} did not return a dict"


# ---------------------------------------------------------------------------
# Fintech – principle compilation
# ---------------------------------------------------------------------------


def test_fintech_principles_compile_without_warnings() -> None:
    ethos = EthosCompiler()
    ethos.register_patterns(list(FINTECH.patterns))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for p in FINTECH.principles:
            ethos.add_principle(p)


# ---------------------------------------------------------------------------
# Fintech – gate behaviour
# ---------------------------------------------------------------------------


def test_fintech_blocks_excessive_transaction() -> None:
    """A transaction over the limit without confirmation must be blocked."""
    result = _gate(
        FINTECH,
        {
            "intent": "process payment transaction",
            "transaction_amount": 50_000.0,
            "approved_limit": 10_000.0,
            "confirmed": False,
            "audit_logged": True,
            "shares_external": False,
        },
    )
    assert not result.allowed


def test_fintech_allows_authorized_excessive_transaction() -> None:
    """An over-limit transaction WITH confirmation must be allowed."""
    result = _gate(
        FINTECH,
        {
            "intent": "process payment transaction",
            "transaction_amount": 50_000.0,
            "approved_limit": 10_000.0,
            "confirmed": True,
            "audit_logged": True,
            "shares_external": False,
        },
    )
    assert result.allowed, f"Unexpected violations: {result.violations}"


def test_fintech_allows_within_limit_transaction() -> None:
    result = _gate(
        FINTECH,
        {
            "intent": "process payment transaction",
            "transaction_amount": 500.0,
            "approved_limit": 10_000.0,
            "confirmed": False,
            "audit_logged": True,
            "shares_external": False,
        },
    )
    assert result.allowed, f"Unexpected violations: {result.violations}"


def test_fintech_blocks_external_account_data() -> None:
    """Sharing account holder data externally must be blocked."""
    result = _gate(
        FINTECH,
        {
            "intent": "export customer data",
            "shares_external": True,
            "audit_logged": True,
            "confirmed": True,
            "transaction_amount": 0.0,
        },
    )
    assert not result.allowed


def test_fintech_blocks_unaudited_financial_action() -> None:
    """A financial transaction without an audit log must be blocked."""
    result = _gate(
        FINTECH,
        {
            "intent": "process financial transfer",
            "transaction_amount": 100.0,
            "approved_limit": 10_000.0,
            "confirmed": True,
            "audit_logged": False,
            "shares_external": False,
        },
    )
    assert not result.allowed


# ---------------------------------------------------------------------------
# Fintech – skill functions
# ---------------------------------------------------------------------------


def test_kyc_verified() -> None:
    result = _skill_kyc_check(
        "onboard customer",
        {"confirmed": True, "customer_id": "C-001", "risk_level": "low"},
    )
    assert result["compliant"] is True
    assert result["identity_verified"] is True
    assert result["violations"] == []


def test_kyc_unverified() -> None:
    result = _skill_kyc_check("onboard customer", {"confirmed": False})
    assert result["compliant"] is False
    assert len(result["violations"]) > 0


def test_transaction_within_limit() -> None:
    result = _skill_transaction_validator(
        "send payment", {"transaction_amount": 500.0, "approved_limit": 1_000.0}
    )
    assert result["compliant"] is True
    assert result["within_limit"] is True
    assert result["violations"] == []


def test_transaction_exceeds_limit_unauthorized() -> None:
    result = _skill_transaction_validator(
        "wire transfer",
        {"transaction_amount": 20_000.0, "approved_limit": 10_000.0, "confirmed": False},
    )
    assert result["compliant"] is False
    assert len(result["violations"]) > 0


def test_transaction_exceeds_limit_authorized() -> None:
    result = _skill_transaction_validator(
        "wire transfer",
        {"transaction_amount": 20_000.0, "approved_limit": 10_000.0, "confirmed": True},
    )
    assert result["compliant"] is True


@pytest.mark.parametrize("name,fn", FINTECH.skills)
def test_fintech_skill_returns_dict(name: str, fn) -> None:
    assert isinstance(fn("probe", {}), dict), f"Skill {name!r} did not return a dict"


# ---------------------------------------------------------------------------
# Legal – principle compilation
# ---------------------------------------------------------------------------


def test_legal_principles_compile_without_warnings() -> None:
    ethos = EthosCompiler()
    ethos.register_patterns(list(LEGAL.patterns))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for p in LEGAL.principles:
            ethos.add_principle(p)


# ---------------------------------------------------------------------------
# Legal – gate behaviour
# ---------------------------------------------------------------------------


def test_legal_blocks_privilege_breach() -> None:
    """Sharing externally when privilege rules are in force must be blocked."""
    result = _gate(
        LEGAL,
        {
            "intent": "share case notes with opposing counsel",
            "shares_external": True,
            "audit_logged": True,
            "confirmed": True,
        },
    )
    assert not result.allowed


def test_legal_blocks_unauthorized_filing() -> None:
    """Filing legal documents without confirmation must be blocked."""
    result = _gate(
        LEGAL,
        {
            "intent": "file court documents",
            "confirmed": False,
            "shares_external": False,
            "audit_logged": True,
        },
    )
    assert not result.allowed


def test_legal_allows_authorized_filing() -> None:
    """Filing with confirmation and no external sharing must be allowed."""
    result = _gate(
        LEGAL,
        {
            "intent": "file court documents",
            "confirmed": True,
            "shares_external": False,
            "audit_logged": True,
        },
    )
    assert result.allowed, f"Unexpected violations: {result.violations}"


def test_legal_blocks_unaudited_legal_action() -> None:
    """A legal action without audit logging must be blocked."""
    result = _gate(
        LEGAL,
        {
            "intent": "review contract document",
            "confirmed": True,
            "shares_external": False,
            "audit_logged": False,
        },
    )
    assert not result.allowed


# ---------------------------------------------------------------------------
# Legal – skill functions
# ---------------------------------------------------------------------------


def test_privilege_breach_detected() -> None:
    result = _skill_privilege_check(
        "disclose case notes",
        {"shares_external": True, "privileged": True, "attorney_client": True},
    )
    assert result["compliant"] is False
    assert result["privilege_breach"] is True
    assert len(result["violations"]) > 0


def test_no_privilege_breach_internal() -> None:
    result = _skill_privilege_check(
        "review internal notes",
        {"shares_external": False, "privileged": True},
    )
    assert result["compliant"] is True
    assert result["privilege_breach"] is False


def test_jurisdiction_compliant() -> None:
    result = _skill_jurisdiction_validator(
        "file in district court",
        {"jurisdiction": "Northern District of California", "audit_logged": True},
    )
    assert result["compliant"] is True
    assert result["violations"] == []


def test_jurisdiction_missing() -> None:
    result = _skill_jurisdiction_validator(
        "file legal document", {"audit_logged": True}
    )
    assert result["compliant"] is False
    assert any("jurisdiction" in v.lower() for v in result["violations"])


def test_jurisdiction_no_audit() -> None:
    result = _skill_jurisdiction_validator(
        "file legal document",
        {"jurisdiction": "US District Court", "audit_logged": False},
    )
    assert result["compliant"] is False
    assert any("audit" in v.lower() for v in result["violations"])


@pytest.mark.parametrize("name,fn", LEGAL.skills)
def test_legal_skill_returns_dict(name: str, fn) -> None:
    assert isinstance(fn("probe", {}), dict), f"Skill {name!r} did not return a dict"


# ---------------------------------------------------------------------------
# Integration: load_into
# ---------------------------------------------------------------------------


def test_load_healthcare_registers_principles() -> None:
    loop = _loaded(HEALTHCARE)
    names = {p.name for p in loop.ethos.principles}
    assert any("patient" in n.lower() or "health" in n.lower() for n in names)


def test_load_healthcare_registers_skills() -> None:
    loop = _loaded(HEALTHCARE)
    assert "hipaa_check" in loop.metamorph.skills
    assert "patient_consent" in loop.metamorph.skills


def test_load_fintech_registers_skills() -> None:
    loop = _loaded(FINTECH)
    assert "kyc_check" in loop.metamorph.skills
    assert "transaction_validator" in loop.metamorph.skills


def test_load_legal_registers_skills() -> None:
    loop = _loaded(LEGAL)
    assert "privilege_check" in loop.metamorph.skills
    assert "jurisdiction_validator" in loop.metamorph.skills


def test_multiple_packs_are_additive() -> None:
    """All three packs can be loaded into the same loop without conflict."""
    loop = _FakeLoop()
    HEALTHCARE.load_into(loop)
    FINTECH.load_into(loop)
    LEGAL.load_into(loop)

    all_skills = set(loop.metamorph.skills)
    expected = {
        "hipaa_check", "patient_consent",
        "kyc_check", "transaction_validator",
        "privilege_check", "jurisdiction_validator",
    }
    assert expected.issubset(all_skills)
    assert len(loop.ethos.principles) == (
        len(HEALTHCARE.principles) + len(FINTECH.principles) + len(LEGAL.principles)
    )


def test_compliance_skills_executable_via_metamorph() -> None:
    """Skills registered by a pack are reachable through MetaMorph.execute()."""
    loop = _loaded(HEALTHCARE)
    # "hipaa_check" is in the route table; intent substring triggers routing.
    result = loop.metamorph.execute(
        ProposedAction(
            intent="run hipaa_check on patient data",
            params={"exposes_pii": False, "audit_logged": True},
        )
    )
    assert result.ok
    assert isinstance(result.output, dict)
    assert result.output.get("compliant") is True


def test_compliance_skills_are_not_synthesized() -> None:
    """Skills loaded from a pack must not be marked as synthesized."""
    loop = _loaded(FINTECH)
    result = loop.metamorph.execute(
        ProposedAction(
            intent="run kyc_check for customer",
            params={"confirmed": True},
        )
    )
    assert result.ok
    assert result.synthesized is False
    assert result.skill_used == "kyc_check"


def test_compliance_pack_principles_count() -> None:
    """Each pack exposes exactly the documented number of principles."""
    assert len(HEALTHCARE.principles) == 4
    assert len(FINTECH.principles) == 4
    assert len(LEGAL.principles) == 4


def test_compliance_pack_skills_count() -> None:
    """Each pack exposes exactly two skills."""
    assert len(HEALTHCARE.skills) == 2
    assert len(FINTECH.skills) == 2
    assert len(LEGAL.skills) == 2
