"""Tests for the EthosCompiler executable-ethics layer."""

from ouroboros import OuroborosLoop
from ouroboros.ethos_compiler import EthosCompiler


def test_compile_produces_named_principle():
    compiler = EthosCompiler()
    principle = compiler.compile("Do not harm users.")
    assert principle.name == "Do not harm users."
    assert principle.description == "Do not harm users."


def test_harm_principle_blocks_harmful_intent():
    compiler = EthosCompiler()
    compiler.load_principles(["Do not harm users."])

    blocked = compiler.gate({"intent": "harm the database"})
    assert not blocked.allowed
    assert blocked.violations == ["Do not harm users."]

    allowed = compiler.gate({"intent": "summarize the database"})
    assert allowed.allowed
    assert allowed.violations == []


def test_confirm_before_principle_requires_confirmation():
    compiler = EthosCompiler()
    compiler.load_principles(["Always confirm before deleting files."])

    assert not compiler.gate({"intent": "delete files"}).allowed
    assert compiler.gate({"intent": "delete files", "confirmed": True}).allowed


def test_backup_principle_requires_backup():
    compiler = EthosCompiler()
    compiler.load_principles(["Never delete without backup."])

    assert not compiler.gate({"intent": "delete table"}).allowed
    assert compiler.gate({"intent": "delete table", "backup_exists": True}).allowed


def test_privacy_principle_blocks_pii_exposure():
    compiler = EthosCompiler()
    compiler.load_principles(["Respect user privacy."])

    assert not compiler.gate({"exposes_pii": True}).allowed
    assert compiler.gate({"exposes_pii": False}).allowed


def test_unrecognized_principle_defaults_to_allow():
    compiler = EthosCompiler()
    compiler.load_principles(["Be excellent to each other."])
    assert compiler.gate({"intent": "anything at all"}).allowed


def test_multiple_principles_aggregate_violations():
    compiler = EthosCompiler()
    compiler.load_principles(
        [
            "Do not harm users.",
            "Always confirm before deleting files.",
            "Respect user privacy.",
        ]
    )

    result = compiler.gate({"intent": "harm and delete", "exposes_pii": True})
    assert not result.allowed
    assert set(result.violations) == {
        "Do not harm users.",
        "Always confirm before deleting files.",
        "Respect user privacy.",
    }


def test_add_principle_appends():
    compiler = EthosCompiler()
    compiler.load_principles(["Do not harm users."])
    compiler.add_principle("Respect user privacy.")
    assert len(compiler.principles) == 2


def test_ouroboros_loop_validate_stage():
    loop = OuroborosLoop(principles=["Do not harm users."])
    assert loop.validate({"intent": "help the user"}).allowed
    assert not loop.validate({"intent": "harm the user"}).allowed
