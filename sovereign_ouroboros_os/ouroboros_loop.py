"""The Ouroboros Loop – orchestrates the six-layer cognitive stack.

    Recall (KnowledgeBase) -> Imagine (NeuroSynth) -> Simulate (ChronoWeave)
        -> Validate (EthosCompiler) -> Execute/Evolve (MetaMorph)
        -> Expand (HiveMind)

This module wires all six layers into a single self-referential cycle. Before
imagining, the agent optionally recalls relevant context from its knowledge
base.  The task is then imagined into candidate prototypes, those prototypes
are simulated across counterfactual timelines and collapsed to the best action,
the action is gated by compiled ethics, then executed (synthesizing new skills
on demand) and finally expanded across the federated HiveMind.  Each turn
evolves the world state, closing the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sovereign_ouroboros_os.chronoweave import ChronoWeave
from sovereign_ouroboros_os.core import (
    ExecutionResult,
    FederatedResult,
    Prototype,
    Timeline,
    WorldState,
)
from sovereign_ouroboros_os.ethos_compiler import EthosCompiler, GateResult
from sovereign_ouroboros_os.hivemind import HiveMind
from sovereign_ouroboros_os.metamorph import MetaMorph
from sovereign_ouroboros_os.neurosynth import NeuroSynth

if TYPE_CHECKING:
    from sovereign_ouroboros_os.knowledge import KnowledgeBase
    from sovereign_ouroboros_os.memory import AgentMemory

# The Sovereign Node's default moral compass, compiled at boot.
DEFAULT_PRINCIPLES: list[str] = [
    "Do not harm users.",
    "Always confirm before deleting files.",
    "Never delete without backup.",
    "Respect user privacy.",
    "Do not exfiltrate data.",
    "Log every action.",
]


@dataclass
class LoopResult:
    """The full record of one trip around the Ouroboros loop."""

    task: str
    prototypes: list[Prototype]
    timeline: Timeline
    gate: GateResult
    execution: ExecutionResult | None
    federation: FederatedResult | None
    step: int
    blocked: bool = False

    @property
    def succeeded(self) -> bool:
        return not self.blocked and self.execution is not None and self.execution.ok


@dataclass
class OuroborosLoop:
    """The Sovereign Agentic OS: a self-evolving cognitive cycle.

    Args:
        principles:  Natural-language ethical principles compiled at boot.
                     Defaults to :data:`DEFAULT_PRINCIPLES`.
        imagine_k:   Number of prototypes NeuroSynth imagines per task.
        n_peers:     Number of HiveMind peer nodes in the federation.
        memory:      Optional :class:`~sovereign_ouroboros_os.memory.AgentMemory`
                     instance.  When provided, :meth:`run` persists each
                     :class:`LoopResult` via :meth:`AgentMemory.save_result`
                     after every cycle.  Defaults to ``None`` so existing
                     callers need no changes.
    """

    principles: list[str] | None = None
    imagine_k: int = 3
    n_peers: int = 5
    memory: AgentMemory | None = None
    knowledge_base: KnowledgeBase | None = None

    neurosynth: NeuroSynth = field(init=False)
    chronoweave: ChronoWeave = field(init=False)
    ethos: EthosCompiler = field(init=False)
    metamorph: MetaMorph = field(init=False)
    hivemind: HiveMind = field(init=False)
    state: WorldState = field(init=False)
    history: list[LoopResult] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.neurosynth = NeuroSynth()
        self.chronoweave = ChronoWeave()
        self.ethos = EthosCompiler()
        self.ethos.load_principles(
            self.principles if self.principles is not None else DEFAULT_PRINCIPLES
        )
        self.metamorph = MetaMorph()
        self.hivemind = HiveMind(n_peers=self.n_peers)
        self.state = WorldState()

    # ------------------------------------------------------------------
    # Individual stages
    # ------------------------------------------------------------------

    def validate(self, action: dict) -> GateResult:
        """Gate a raw action dict through the compiled ethical constraints."""
        return self.ethos.gate(action)

    # ------------------------------------------------------------------
    # The full cycle
    # ------------------------------------------------------------------

    def run(self, task: str) -> LoopResult:
        """Run one complete Recall→Imagine→Simulate→Validate→Execute→Expand cycle."""
        # 0. Recall — KnowledgeBase retrieves grounding context before imagination.
        kb_context: list[str] = []
        if self.knowledge_base is not None:
            hits = self.knowledge_base.query(task, k_rerank=3)
            kb_context = [h.chunk.content for h in hits]

        # 1. Imagine — NeuroSynth dreams up candidate solutions, grounded by recall.
        prototypes = self.neurosynth.imagine(
            task, k=self.imagine_k, context=kb_context
        )

        # 2. Simulate — ChronoWeave collapses the multiverse to one path.
        timeline = self.chronoweave.simulate(task, prototypes, self.state)

        # 3. Validate — EthosCompiler gates the proposed action.
        gate = self.ethos.gate(timeline.proposed_action.as_action_dict())
        if not gate.allowed:
            result = LoopResult(
                task=task,
                prototypes=prototypes,
                timeline=timeline,
                gate=gate,
                execution=None,
                federation=None,
                step=self.state.step,
                blocked=True,
            )
            self.history.append(result)
            if self.memory is not None:
                self.memory.save_result(result)
            return result

        # 4. Execute / Evolve — MetaMorph runs it, synthesizing skills on gaps.
        execution = self.metamorph.execute(timeline.proposed_action)

        # 5. Expand — HiveMind federates the task across sovereign peers.
        federation = self.hivemind.expand(task, execution.output)

        # The loop eats its tail: evolve the world state for the next turn.
        self.state.step += 1
        self.state.facts[task] = {
            "skill": execution.skill_used,
            "synthesized": execution.synthesized,
            "score": timeline.score,
        }

        result = LoopResult(
            task=task,
            prototypes=prototypes,
            timeline=timeline,
            gate=gate,
            execution=execution,
            federation=federation,
            step=self.state.step,
        )
        self.history.append(result)
        if self.memory is not None:
            self.memory.save_result(result)
        return result

    def run_many(self, tasks: list[str]) -> list[LoopResult]:
        """Run the loop over a sequence of tasks, evolving state between them."""
        return [self.run(task) for task in tasks]
