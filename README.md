# Verifying Decisions Under Uncertainty

**A value-aware evaluation framework for sequential language-model agents**

Imen Al Samarai · [ORCID 0009-0001-0060-9225](https://orcid.org/0009-0001-0060-9225) · [Paper (PDF)](paper/paper.pdf)

This repository contains the benchmark environment, the evaluation logs and the analysis code for the paper. Every analysis of the benchmark and every figure can be reproduced from the released logs, without calls to any model.

## Overview

When decisions are made under uncertainty, realised outcomes mix decision quality with chance, and action accuracy ignores how much each error costs. The framework evaluates each decision of an agent against an exact reference: the action values of the Bellman-optimal policy of a stochastic control problem. Each decision is scored at the state the agent actually reached, decisions are classified as *greedy-consistent* (the immediate payoff settles them) or *continuation-sensitive* (future value changes the optimal action), and the agent's expected regret is decomposed along its own trajectory through the performance-difference identity.

The benchmark is a finite-horizon optimal-switching problem from quantitative finance. At each step the agent decides whether a trading strategy is active, given a mean-reverting (Ornstein–Uhlenbeck) signal and a fixed switching cost. The repository includes evaluations of Claude Opus 4, Sonnet 4 and Haiku 4.5 on 500 episodes each, with and without a planning instruction.

## Repository structure

```
optimal_switching/             benchmark environment
  problems/regime_switching/     generator.py   episode generator and released Bellman solver
                                 verifier.py    decision parser and scoring
                                 prompts.py     system and step prompts (Appendix B)
  agents/                        greedy and optimal reference agents
analysis/
  solver.py                      independent implementation of the Bellman solver (Appendix C.1)
  heuristics.py                  calibration of the horizon-aware heuristics (Appendix C.2)
  reproduce.py                   all other analyses and figures (see the table below)
data/
  opus4_default.json             Claude Opus 4, default prompt, 500 episodes
  sonnet4_default.json           Claude Sonnet 4, default prompt
  haiku45_default.json           Claude Haiku 4.5, default prompt
  opus4_instruction.json         Claude Opus 4 with the planning instruction (Section 5.3)
  per_step_decisions.csv         the four runs, one row per decision
scripts/eval_parallel.py       evaluation harness used to collect the logs (requires an API key)
paper/                         paper source, PDF and figures
tests/                         environment tests
```

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/ImenAlSamarai/decision-level-evaluation.git
cd decision-level-evaluation
pip install -e ".[analysis,dev]"
pytest -q
```

## Reproducing the results

```bash
python analysis/heuristics.py    # Appendix C.2 (seconds)
python analysis/reproduce.py     # all other results and figures (about 10 minutes)
```

`reproduce.py` runs every section in order; a single section can be run with `python analysis/reproduce.py <section>`. Results are written to `analysis/results/`, figures to `paper/`.

| Section | Paper content |
|---|---|
| `reference` | §4 numerical stability; App. C.1 agreement between the two solvers; §4 value at stake (Figure 3, left) |
| `identify` | §4 and App. C.3: estimation of the mean-reversion speed within one episode |
| `numerics` | §4: high-cost regime with a wider grid and linear extrapolation |
| `regimes` | §4 difficulty characterisation (Figure 2) |
| `infogap` | §4 and App. C.3: planners without parameter knowledge; greedy rule in Table 1 |
| `wideprior` | §4 and App. C.3: Bayesian learner with a wide prior |
| `completeness` | §5 and App. C.5: responses without a stated decision |
| `scored` | Tables 1 and 2; §5.2 and App. C.4 horizon regressions |
| `bystep` | Table 3; §5 and §5.3 robustness checks; App. B parser check; App. C.3 white-noise test |
| `checks` | Table 1 optimal expected utility; App. C.2 heuristic transfer; App. C.5 stricter scoring rule |
| `figures` | Figures 2, 3 and 4 |
| `episode` | Figure 1 |

The examples in Appendix A are responses from `data/opus4_default.json` (seed 31, step 2; seed 223, step 4). The fitted mean-reversion speeds in the first two rows of Table 4 come from AR(1) regressions on public daily market data and are not part of this release.

## Model evaluations

The logs were collected through the Anthropic API with `scripts/eval_parallel.py`:

| Run | Model |
|---|---|
| Opus 4 | `claude-opus-4-20250514` |
| Sonnet 4 | `claude-sonnet-4-20250514` |
| Haiku 4.5 | `claude-haiku-4-5-20251001` |

Each episode is one conversation using the prompts in `optimal_switching/problems/regime_switching/prompts.py`, with a maximum of 300 output tokens, no tools, no extended thinking and the API's default sampling temperature. The decision is read as the last expression of the form `s_t = 0/1` in a response; responses that do not state a decision within the token limit are excluded from scoring (Appendix C.5). For the three default runs, seeds 0–29 come from an initial run with identical prompts and settings and seeds 30–499 from `eval_parallel.py`.

Re-running the evaluations requires `ANTHROPIC_API_KEY` in the environment and incurs API costs; some of these model versions may no longer be available. New logs are written to `runs/`:

```bash
pip install -e ".[eval]"
python scripts/eval_parallel.py --model opus --start 0 --end 499           # default prompt
python scripts/eval_parallel.py --model opus --start 0 --end 499 --hint    # planning instruction
```

## Data format

Each log in `data/*.json` contains a list `instances`, one per episode, with the episode parameters (`seed`, `kappa`, `alpha`, `lam`, `sigma_z`, `T`), the parsed `decisions` and the full model responses `raw_texts`.

`data/per_step_decisions.csv` has one row per decision: run key (`file`: `opus_n500`, `sonnet_n500`, `haiku_n500`, `opus_hint_n500`), `seed`, step `t`, steps remaining `rem`, the episode parameters, signal `z`, incoming position `state`, the model's action `model`, the reference action `opt` and the greedy action `greedy`.

## Citation

```bibtex
@misc{alsamarai2026verifying,
  title  = {Verifying Decisions Under Uncertainty: A Value-Aware Evaluation Framework
            for Sequential Language-Model Agents},
  author = {Al Samarai, Imen},
  year   = {2026},
  note   = {Code and data: https://github.com/ImenAlSamarai/decision-level-evaluation, release v1.0.0}
}
```

## License

Code: MIT (see `LICENSE`). The evaluation logs in `data/` are released under the same terms.

## Contact

Imen Al Samarai — imen.alsamarai@gmail.com
