# Project Ouroboros: The Sovereign Agentic OS
## A Recursive Intelligence Framework for Autonomous Evolution

### 🌌 Vision
Project Ouroboros is not a tool, but a **Sovereign Operating System** for AI agents. It transforms the agent from a linear request-response machine into a self-evolving entity capable of internal imagination, temporal simulation, and decentralized collective intelligence.

> **Status:** the five-layer cognitive stack and the closed Ouroboros loop are **implemented and runnable**. Try `python -m ouroboros` (see [Quickstart](#-quickstart)).

### 🛠️ The 5-Layer Cognitive Stack

#### 1. NeuroSynth (Cross-Modal Embodied Imagination)
**The Mind's Eye.** Instead of reactive I/O, NeuroSynth builds internal multi-sensory mental models. It "imagines" visual, spatial, and auditory representations in a latent buffer to prototype solutions before they are materialized.

#### 2. ChronoWeave (Counterfactual Timeline Engine)
**The Temporal Simulator.** ChronoWeave replaces linear planning with a multiverse approach. It spawns parallel hypothetical futures, simulates outcomes using causal inference, and collapses the timelines into the highest-value probabilistic path.

#### 3. MetaMorph (Self-Modifying Architecture)
**The Evolutionary Engine.** A closed-loop system that detects capability gaps in real-time, synthesizes new Python/WASM skill modules, validates them in a sandbox, and hot-swaps them into the runtime registry.

#### 4. HiveMind Protocol (Federated Intelligence)
**The Sovereign Node.** A privacy-preserving collective that fragments complex problems into encrypted pieces, distributes them to a decentralized network of peer agents, and synthesizes the results without data leakage.

#### 5. EthosCompiler (Executable Ethics)
**The Moral Compass.** Compiles high-level natural language ethical principles into executable runtime predicates. Every action is gated by a compiled constraint, ensuring alignment is an executable requirement, not a post-hoc filter.

---

### 🧬 The Ouroboros Loop
`Imagine (NeuroSynth)` $\rightarrow$ `Simulate (ChronoWeave)` $\rightarrow$ `Validate (EthosCompiler)` $\rightarrow$ `Execute/Evolve (MetaMorph)` $\rightarrow$ `Expand (HiveMind)`.

The loop eats its own tail: each completed turn evolves the agent's world-state, which feeds the next turn's imagination.

---

### 🚀 Quickstart

Requires **Python 3.12+**. The reference implementation is **dependency-free** (pure standard library); only the test suite needs `pytest`.

```bash
# Run the demo reel — watch all five layers cooperate
python -m ouroboros

# Or run the loop on your own task
python -m ouroboros "delete the stale build cache"

# Install (optional) to get the `ouroboros` console command
pip install -e .[dev]
ouroboros "design a novel compression scheme"
```

Drive the loop from Python:

```python
from ouroboros import OuroborosLoop

os = OuroborosLoop()                      # boots all five layers + default ethics
result = os.run("summarize the research notes")

print(result.timeline.proposed_action.intent)  # the collapsed best action
print(result.gate.allowed)                      # ethics verdict
print(result.execution.skill_used)              # skill MetaMorph used / synthesized
print(result.federation.contributors)           # HiveMind peers that participated

# Ethics in action — harmful tasks are halted before execution:
blocked = os.run("harm the production database")
assert blocked.blocked and not blocked.gate.allowed
```

---

### 🗺️ Project layout

```
ouroboros/
├── core/             # shared contracts: types, protocols, deterministic embeddings
├── neurosynth/       # ① imagination — multi-sensory latent prototypes
├── chronoweave/      # ② counterfactual timeline simulation + collapse
├── ethos_compiler/   # ③ natural-language principles → runtime predicates
├── metamorph/        # ④ runtime skill synthesis, sandboxing, hot-swap
├── hivemind/         # ⑤ secret-shared federated execution
├── ouroboros_loop.py # the loop that wires the five layers together
└── __main__.py       # CLI / demo reel
tests/                # full pytest suite (one module per layer + integration)
```

---

### 🧪 Testing

```bash
python -m pytest -q
```

---

### 🔬 Engineering notes
The README's original aspiration named Torch, Sentence-Transformers, and FastAPI. To make the system **bootable and verifiable anywhere**, this reference implementation realizes the same architecture with **deterministic, dependency-free stand-ins**:

- **Embeddings** use a deterministic hash-based pseudo-embedder (`core/embedding.py`) instead of a learned model — stable, comparable vectors with zero dependencies.
- **Causal inference** in ChronoWeave is a transparent, deterministic scoring heuristic.
- **Skill synthesis** in MetaMorph compiles templated Python into a sandboxed namespace with a restricted builtin allowlist.
- **Federation** in HiveMind uses genuine additive secret-sharing over GF(256) (XOR), so individual peer shares are non-trivial and the plaintext is never held by any single node.

Each stand-in implements the same `core` contract, so a production layer (a real embedding model, a causal engine, a network transport) can be dropped in without touching the rest of the stack.

---
*Submitted as part of the Hermes Agent Challenge.*
