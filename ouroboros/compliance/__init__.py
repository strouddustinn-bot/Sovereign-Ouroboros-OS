"""Compliance skill packs for regulated industries.

Usage::

    from ouroboros.compliance import HEALTHCARE, FINTECH, LEGAL
    from ouroboros.ouroboros_loop import OuroborosLoop

    loop = OuroborosLoop()
    HEALTHCARE.load_into(loop)  # adds HIPAA principles + skills
    FINTECH.load_into(loop)     # adds KYC / AML principles + skills
    LEGAL.load_into(loop)       # adds privilege / jurisdiction principles + skills
"""

from ouroboros.compliance.domains import (
    FINTECH,
    HEALTHCARE,
    LEGAL,
    CompliancePack,
)

__all__ = ["CompliancePack", "HEALTHCARE", "FINTECH", "LEGAL"]
