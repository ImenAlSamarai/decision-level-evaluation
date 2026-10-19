"""Horizon-aware heuristics H1 and H2 (Appendix C.2): held-out calibration, and comparison with the Bellman reference
at the states the models visited.

Input:   data/per_step_decisions.csv
Outputs: analysis/results/heuristics.json, heuristics_table.md, heuristics_fig.png
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
from solver import solve_bellman, optimal_action, greedy_action, simulate_signal
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(HERE, "..", "data", "per_step_decisions.csv"))
rng = np.random.default_rng(0)

# ---------- 1. attach exact Q-gaps to every visited state ----------
cache = {}
def Qtab(row):
    key = (row.seed, round(row.kappa, 9))
    if key not in cache:
        cache[key] = solve_bellman(row.kappa, row.alpha, row.lam, int(row.T), row.sigma_z)
    return cache[key]

gaps, opt_chk = [], []
for r in df.itertuples():
    zg, pi, V, Q = Qtab(r)
    a, g = optimal_action(r.z, int(r.state), int(r.t), zg, Q)
    gaps.append(g); opt_chk.append(a)
df["gapQ"] = gaps                       # Q(1)-Q(0) at the visited state
df["opt_chk"] = opt_chk
assert (df.opt_chk == df.opt).mean() > 0.995, (df.opt_chk == df.opt).mean()
df["stake"] = df.gapQ.abs()
df["regret_model"] = np.where(df.model == df.opt, 0.0, df.stake)   # Δ_t = Q*(a*) - Q*(a_LLM)
df["hard"] = df.opt != df.greedy

# ---------- 2. held-out calibration of horizon-aware heuristics ----------
# Heuristic H1: switch iff alpha*z*m(rem) beats lam, m(rem) = (1-rho^rem)/(1-rho)  (geometric amortisation, does not know kappa)
# Heuristic H2: same but m(rem) = min(rem, K)  (flat amortisation window)
# Calibrated on 300 fresh training episodes from the same generator distribution (different seeds), by maximising mean utility.
def sample_params(rng):
    return dict(kappa=rng.uniform(0.1, 0.25), r=rng.uniform(0.3, 0.5), alpha=rng.uniform(0.1, 0.5), sigma_z=rng.uniform(0.1, 0.3), sigma_x=rng.uniform(0.1, 0.3), T=int(rng.integers(5, 21)), s0=int(rng.integers(0, 2)))

def heuristic_action(kind, p, z, s, rem, alpha, lam):
    if kind == "H1": m = (1 - p ** rem) / (1 - p)
    else: m = min(rem, p)
    v = alpha * z * m
    return (1 if v > lam else 0) if s == 0 else (0 if v < -lam else 1)

def run_episode(policy, ep):
    z, x, lam, alpha, T = ep["z"], ep["x"], ep["lam"], ep["alpha"], ep["T"]
    s, J = ep["s0"], 0.0
    for t in range(T):
        a = policy(z[t], s, T - t, t)
        J += a * x[t] - lam * (a != s); s = a
    return J

train = []
for i in range(300):
    p = sample_params(rng); lam = p["r"] * p["alpha"]
    z = simulate_signal(p["kappa"], p["sigma_z"], p["T"], rng, z0=rng.normal(0, p["sigma_z"] / np.sqrt(2 * p["kappa"])))
    x = p["alpha"] * z + p["sigma_x"] * rng.normal(size=p["T"])
    train.append(dict(z=z, x=x, lam=lam, alpha=p["alpha"], T=p["T"], s0=p["s0"]))

def mean_util(kind, p):
    return np.mean([run_episode(lambda z, s, rem, t: heuristic_action(kind, p, z, s, rem, ep["alpha"], ep["lam"]), ep) for ep in train])
rho_grid = np.round(np.arange(0.0, 0.96, 0.05), 2); K_grid = list(range(1, 8))
rho_best = max(rho_grid, key=lambda p: mean_util("H1", p)); K_best = max(K_grid, key=lambda p: mean_util("H2", p))
calib = dict(rho=float(rho_best), K=int(K_best), util_H1=float(mean_util("H1", rho_best)), util_H2=float(mean_util("H2", K_best)), util_greedy=float(mean_util("H1", 0.0)))

# ---------- 3. heuristic labels at the LLM-visited states (held-out: eval episodes, different seeds) ----------
df["H1"] = [heuristic_action("H1", rho_best, r.z, int(r.state), int(r.rem), r.alpha, r.lam) for r in df.itertuples()]
df["H2"] = [heuristic_action("H2", K_best, r.z, int(r.state), int(r.rem), r.alpha, r.lam) for r in df.itertuples()]
for h in ["H1", "H2"]:
    df[f"regret_{h}"] = np.where(df[h] == df.opt, 0.0, df.stake)     # regret of the heuristic's own choice at that state
    df[f"hard_{h}"] = df[h] != df.greedy                               # 'hard' as the heuristic would define it

# ---------- 4. the comparison table ----------
labels = {"opus_n500": "Opus 4", "sonnet_n500": "Sonnet 4", "haiku_n500": "Haiku 4.5", "opus_hint_n500": "Opus 4 + instruction"}
res = {"calibration": calib, "models": {}}
for f, name in labels.items():
    d = df[df.file == f]; easy, hard = d[~d.hard], d[d.hard]
    m = dict(
        n=len(d), n_hard=int(d.hard.sum()),
        # LLM judged against the ORACLE
        easy_acc=float((easy.model == easy.opt).mean()), hard_acc=float((hard.model == hard.opt).mean()),
        regret_total=float(d.regret_model.sum() / d.seed.nunique()), regret_easy=float(easy.regret_model.sum() / d.seed.nunique()), regret_hard=float(hard.regret_model.sum() / d.seed.nunique()),
        stake_median_hard=float((hard.stake / hard.lam).median()),
        # reference policies' own regret at the same visited states
        regret_greedy_at_states=float(np.where(d.greedy == d.opt, 0, d.stake).sum() / d.seed.nunique()),
        regret_H1_at_states=float(d.regret_H1.sum() / d.seed.nunique()), regret_H2_at_states=float(d.regret_H2.sum() / d.seed.nunique()),
        # how well the heuristics reproduce the oracle's labels
        H1_agree_oracle=float((d.H1 == d.opt).mean()), H1_agree_oracle_hard=float((hard.H1 == hard.opt).mean()),
        H2_agree_oracle=float((d.H2 == d.opt).mean()), H2_agree_oracle_hard=float((hard.H2 == hard.opt).mean()),
        # the LLM judged against the HEURISTIC as if it were the reference
        H1_hard_n=int(d.hard_H1.sum()), H1_easy_acc=float((d[~d.hard_H1].model == d[~d.hard_H1].H1).mean()), H1_hard_acc=float((d[d.hard_H1].model == d[d.hard_H1].H1).mean()) if d.hard_H1.any() else np.nan,
        H2_hard_n=int(d.hard_H2.sum()), H2_easy_acc=float((d[~d.hard_H2].model == d[~d.hard_H2].H2).mean()), H2_hard_acc=float((d[d.hard_H2].model == d[d.hard_H2].H2).mean()) if d.hard_H2.any() else np.nan,
        # overlap of hard sets
        hard_overlap_H1=float((d.hard & d.hard_H1).sum() / max(1, (d.hard | d.hard_H1).sum())),
        hard_overlap_H2=float((d.hard & d.hard_H2).sum() / max(1, (d.hard | d.hard_H2).sum())),
        # where oracle and heuristic labels disagree, who does the LLM follow, and what does it cost?
        n_label_disagree_H1=int((d.H1 != d.opt).sum()), llm_follows_H1_when_disagree=float((d[d.H1 != d.opt].model == d[d.H1 != d.opt].H1).mean()),
        regret_in_disagree_H1=float(d[d.H1 != d.opt].regret_model.sum() / d.seed.nunique()),
    )
    res["models"][name] = m

json.dump(res, open(os.path.join(OUT, "heuristics.json"), "w"), indent=1)

# ---------- 5. markdown table + figure ----------
L = ["## Phase 1 — oracle vs held-out heuristics vs regret (eval regime, N=500 episodes per model)", "",
     f"Heuristics calibrated on 300 fresh training episodes (different seeds), maximising utility: H1 geometric amortisation ρ={calib['rho']}, H2 flat window K={calib['K']}. "
     f"Training-set mean utility: greedy {calib['util_greedy']:.3f}, H1 {calib['util_H1']:.3f}, H2 {calib['util_H2']:.3f}.", "",
     "### A. Expected decision regret per episode (Σ_t Q*(a*) − Q*(a_taken), at the states the model actually visited; optimal continuation)", "",
     "| model | regret total | on easy | on hard | greedy's regret at same states | H1's regret | H2's regret |", "|---|---|---|---|---|---|---|"]
for n, m in res["models"].items():
    L.append(f"| {n} | {m['regret_total']:.4f} | {m['regret_easy']:.4f} | {m['regret_hard']:.4f} | {m['regret_greedy_at_states']:.4f} | {m['regret_H1_at_states']:.4f} | {m['regret_H2_at_states']:.4f} |")
L += ["", "### B. Would a heuristic reference reach the same diagnosis? (LLM scored against each reference's own easy/hard split)", "",
      "| model | ref | #hard | easy acc | hard acc | ref agrees with oracle (all / on oracle-hard) | hard-set overlap (Jaccard) |", "|---|---|---|---|---|---|---|"]
for n, m in res["models"].items():
    L.append(f"| {n} | oracle | {m['n_hard']} | {m['easy_acc']*100:.1f}% | {m['hard_acc']*100:.1f}% | — | — |")
    L.append(f"| {n} | H1 | {m['H1_hard_n']} | {m['H1_easy_acc']*100:.1f}% | {m['H1_hard_acc']*100:.1f}% | {m['H1_agree_oracle']*100:.1f}% / {m['H1_agree_oracle_hard']*100:.1f}% | {m['hard_overlap_H1']:.2f} |")
    L.append(f"| {n} | H2 | {m['H2_hard_n']} | {m['H2_easy_acc']*100:.1f}% | {m['H2_hard_acc']*100:.1f}% | {m['H2_agree_oracle']*100:.1f}% / {m['H2_agree_oracle_hard']*100:.1f}% | {m['hard_overlap_H2']:.2f} |")
L += ["", "### C. Where oracle and H1 labels disagree", "", "| model | #states | LLM follows H1 | LLM regret concentrated there (per episode) |", "|---|---|---|---|"]
for n, m in res["models"].items():
    L.append(f"| {n} | {m['n_label_disagree_H1']} | {m['llm_follows_H1_when_disagree']*100:.1f}% | {m['regret_in_disagree_H1']:.4f} of {m['regret_total']:.4f} |")
open(os.path.join(OUT, "heuristics_table.md"), "w").write("\n".join(L))
print("\n".join(L))

# figure: regret decomposition per model vs references
BLUE, ORANGE, AQUA, VIOLET, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#898781"
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
names = list(res["models"]); x = np.arange(len(names))
ax[0].bar(x - 0.2, [res["models"][n]["regret_easy"] for n in names], 0.4, color=BLUE, label="greedy-consistent decisions")
ax[0].bar(x + 0.2, [res["models"][n]["regret_hard"] for n in names], 0.4, color=ORANGE, label="continuation-sensitive decisions")
ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=8); ax[0].set_ylabel("expected regret per episode"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("Where the models lose value", fontsize=9)
ax[1].bar(x - 0.3, [res["models"][n]["regret_greedy_at_states"] for n in names], 0.2, color=MUTED, label="greedy")
ax[1].bar(x - 0.1, [res["models"][n]["regret_H1_at_states"] for n in names], 0.2, color=AQUA, label=r"H1 ($\rho$=" + f"{calib['rho']})")
ax[1].bar(x + 0.1, [res["models"][n]["regret_H2_at_states"] for n in names], 0.2, color=VIOLET, label=r"H2 ($K$=" + f"{calib['K']})")
ax[1].bar(x + 0.3, [res["models"][n]["regret_total"] for n in names], 0.2, color=ORANGE, label="language model")
ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=8); ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("Action-value loss at common visited states", fontsize=9)
for a in ax: a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color="#e1e0d9", lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "heuristics_fig.png"), dpi=160)
