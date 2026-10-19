"""Every analysis of the paper that uses the Bellman reference (no model calls). Run from the repository root,
all sections in order or one at a time:
    python analysis/reproduce.py reference     # S4 and App. C.1: solver agreement, grid and quadrature refinements, value at stake
    python analysis/reproduce.py identify      # S4 and App. C.3: spread of the least-squares estimate of kappa within one episode (table)
    python analysis/reproduce.py numerics      # S4: high-cost regime with a wider grid and linear extrapolation
    python analysis/reproduce.py regimes       # S4 and Figure 2: greedy expected regret by regime and the difficulty surface
    python analysis/reproduce.py infogap       # S4 and App. C.3: fixed (kappa_hat, sigma_hat) planners and the range-prior Bayesian learner
    python analysis/reproduce.py wideprior     # S4 and App. C.3: Bayesian learner with a prior from near-random walk to white noise
    python analysis/reproduce.py completeness  # App. C.5: classification of responses that do not state a decision within the output limit
    python analysis/reproduce.py scored        # S5, Tables 1-2, App. C.4: model results on scored decisions, horizon regressions
    python analysis/reproduce.py bystep        # S5, S5.2, S5.3, App. B, C.3-C.5: agreement by step (Table 3), excluded decisions, parser and white-noise checks
    python analysis/reproduce.py checks        # Table 1 caption, App. C.2 and C.5: optimal expected utility, heuristic in the high-cost regime, stricter scoring rule
    python analysis/reproduce.py figures       # Figures 2, 3 and 4 (needs the outputs of infogap, wideprior, regimes and scored)
    python analysis/reproduce.py episode       # Figure 1
Results are written to analysis/results/<section>.json and figures to paper/.
"""
import os, sys, json, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from solver import solve_bellman, optimal_action, greedy_action, GH_EPS, GH_PW
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(os.path.join(DATA, "per_step_decisions.csv"))
LABELS = {"opus_n500": "Opus 4", "sonnet_n500": "Sonnet 4", "haiku_n500": "Haiku 4.5", "opus_hint_n500": "Opus 4 + instr."}
save = lambda name, obj: json.dump(obj, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1, default=float)
load = lambda name: json.load(open(os.path.join(OUT, f"{name}.json")))
def qpair(z, t, s, sol):
    zg, Q = sol[0], sol[3]; return np.interp(z, zg, Q[t, :, s, 0]), np.interp(z, zg, Q[t, :, s, 1])
def episodes():
    """The 500 evaluation episodes: parameters, signal path (identical across runs) and initial position."""
    out = {}
    for seed, ep in df[df.file == "opus_n500"].groupby("seed"):
        ep = ep.sort_values("t"); r = ep.iloc[0]
        out[seed] = dict(kappa=r.kappa, alpha=r.alpha, lam=r.lam, sigma_z=r.sigma_z, T=int(r["T"]), z=ep.z.values, s0=int(r.state))
    return out

# ---------------------------------------------------------------- information gap
def infogap():
    FIX = [(k, s) for k in (0.10, 0.175, 0.25) for s in (0.10, 0.20, 0.30)]
    GRID = [(k, s) for k in np.linspace(0.1, 0.25, 7) for s in np.linspace(0.1, 0.3, 7)]
    def loglik(z, k, s):  # stationary z0 as in the generator, Gaussian AR(1) transitions
        v0 = s**2/(2*k); ll = -0.5*np.log(2*np.pi*v0) - z[0]**2/(2*v0)
        if len(z) > 1: e = z[1:] - (1-k)*z[:-1]; ll += np.sum(-0.5*np.log(2*np.pi*s**2) - e**2/(2*s**2))
        return ll
    E = episodes(); ON = {str(p): [] for p in FIX + ["bayes", "greedy"]}; ROWS = {name: [] for name in LABELS.values()}
    byseed = {f: df[df.file == f].groupby("seed") for f in LABELS}
    for seed, e in E.items():   # one episode at a time to keep memory small
        a, lam, T = e["alpha"], e["lam"], e["T"]
        true = solve_bellman(e["kappa"], a, lam, T, e["sigma_z"])
        fixed = {p: solve_bellman(p[0], a, lam, T, p[1]) for p in FIX}; grid = [solve_bellman(p[0], a, lam, T, p[1]) for p in GRID]
        W = []
        for t in range(T):
            ll = np.array([loglik(e["z"][:t+1], *p) for p in GRID]); w = np.exp(ll - ll.max()); W.append(w/w.sum())
        def actions(t, z, s):
            q0, q1 = qpair(z, t, s, true); o = dict(opt=int(q1 > q0), gap=abs(q1 - q0), greedy=greedy_action(z, s, a, lam))
            for p, sol in fixed.items(): f0, f1 = qpair(z, t, s, sol); o[p] = int(f1 > f0)
            bw = W[t] @ np.array([qpair(z, t, s, sol) for sol in grid]); o["bayes"] = int(bw[1] > bw[0]); return o
        for pol in FIX + ["bayes", "greedy"]:
            s = e["s0"]; tot = 0.0
            for t in range(T):
                o = actions(t, e["z"][t], s); b = o[pol]; tot += o["gap"] if b != o["opt"] else 0.0; s = b
            ON[str(pol)].append(tot)
        for f, name in LABELS.items():
            for x in byseed[f].get_group(seed).itertuples():
                o = actions(int(x.t), x.z, int(x.state)); o["model"] = int(x.model); ROWS[name].append(o)
    res = {"onpolicy": {k: dict(mean=float(np.mean(v)), se=float(np.std(v)/np.sqrt(len(v)))) for k, v in ON.items()}, "at_model_states": {}}
    for name, rows in ROWS.items():
        L = pd.DataFrame(rows); cs = L.opt != L.greedy; err = L.model != L.opt
        res["at_model_states"][name] = dict(n_cs=int(cs.sum()), bayes_agree_cs=float((L.bayes[cs] == L.opt[cs]).mean()),
            fixed_agree_cs={str(p): float((L[p][cs] == L.opt[cs]).mean()) for p in FIX},
            share_model_regret_where_bayes_correct=float(L.gap[err & (L.bayes == L.opt)].sum()/L.gap[err].sum()))
    save("infogap", res); print(json.dumps(res, indent=1))

# ---------------------------------------------------------------- Bayesian learner with a wide prior
def wideprior():
    """Same learner as infogap, but the prior does not use the evaluation ranges: kappa log-uniform on [0.01, 1]
    (from a near-random walk to white noise) and sigma_z log-uniform on [0.03, 1]."""
    GRID = [(k, s) for k in np.geomspace(0.01, 1.0, 12) for s in np.geomspace(0.03, 1.0, 9)]
    def loglik(z, k, s):
        v0 = s**2/(2*k - k*k); ll = -0.5*np.log(2*np.pi*v0) - z[0]**2/(2*v0)
        if len(z) > 1: e = z[1:] - (1-k)*z[:-1]; ll += np.sum(-0.5*np.log(2*np.pi*s**2) - e**2/(2*s**2))
        return ll
    E = episodes(); ON = []; ROWS = []; byseed = df[df.file == "opus_n500"].groupby("seed"); POST = []
    for seed, e in E.items():
        a, lam, T = e["alpha"], e["lam"], e["T"]
        true = solve_bellman(e["kappa"], a, lam, T, e["sigma_z"]); grid = [solve_bellman(p[0], a, lam, T, p[1]) for p in GRID]
        W = []
        for t in range(T):
            ll = np.array([loglik(e["z"][:t+1], *p) for p in GRID]); w = np.exp(ll - ll.max()); W.append(w/w.sum())
        POST.append(float(sum(w for w, p in zip(W[-1], GRID) if 0.1 <= p[0] <= 0.25)))
        def act(t, z, s):
            q0, q1 = qpair(z, t, s, true); bw = W[t] @ np.array([qpair(z, t, s, sol) for sol in grid])
            return int(q1 > q0), abs(q1 - q0), int(bw[1] > bw[0]), greedy_action(z, s, a, lam)
        s = e["s0"]; tot = 0.0
        for t in range(T):
            o, g, b, _ = act(t, e["z"][t], s); tot += g if b != o else 0.0; s = b
        ON.append(tot)
        for x in byseed.get_group(seed).itertuples():
            o, g, b, gr = act(int(x.t), x.z, int(x.state)); ROWS.append(dict(opt=o, gap=g, bayes=b, greedy=gr, model=int(x.model)))
    L = pd.DataFrame(ROWS); cs = L.opt != L.greedy; err = L.model != L.opt
    out = dict(onpolicy=dict(mean=float(np.mean(ON)), se=float(np.std(ON)/np.sqrt(len(ON)))), bayes_agree_cs_opus=float((L.bayes[cs] == L.opt[cs]).mean()),
               share_opus_regret_where_bayes_correct=float(L.gap[err & (L.bayes == L.opt)].sum()/L.gap[err].sum()),
               final_posterior_mass_in_eval_kappa_range=float(np.mean(POST)))
    save("wideprior", out); print(json.dumps(out, indent=1))

# ---------------------------------------------------------------- responses cut off by the output limit
PAT = r"(?<![a-zA-Z])s(?:_\{?[\w\-]*\}?)?\s*=\s*([01])\b"   # the harness parser (verifier._parse_decision): last match, 0 if none
def classify(txt):
    """complete: the response ends with its decision (at most 40 characters after the last s_t = 0/1), or states it after the
    word 'decision' and ends with terminal punctuation; cut off: anything else, including responses with no decision."""
    ms = list(re.finditer(PAT, txt))
    if not ms: return "none"
    if len(txt[ms[-1].end():].strip()) <= 40: return "final"
    if "decision" in txt[max(0, ms[-1].start() - 40):ms[-1].start()].lower() and txt.rstrip()[-1:] in ".!)*": return "final_marked"
    return "cut_off"
RUNS = [("opus4_default.json", "opus_n500"), ("sonnet4_default.json", "sonnet_n500"), ("haiku45_default.json", "haiku_n500"), ("opus4_instruction.json", "opus_hint_n500")]   # (log file, run key in per_step_decisions.csv)
def decision_table():
    """One row per logged decision of the four N=500 runs, with the response class, the action-value gap and the scoring flags.
    A decision is scored if its response states a decision (classify: final or final_marked)."""
    rows = []
    for f, key in RUNS:
        for e in json.load(open(os.path.join(DATA, f)))["instances"]:
            for t, txt in enumerate(e["raw_texts"]): rows.append(dict(file=key, seed=int(e["seed"]), t=t, cls=classify(txt), length=len(txt), endpunct=txt.rstrip()[-1:] in ".!?)*"))
    A = pd.DataFrame(rows).merge(df, on=["file", "seed", "t"]); cache = {}; gap = []
    for r in A.itertuples():
        if r.seed not in cache: cache[r.seed] = solve_bellman(r.kappa, r.alpha, r.lam, int(r.T), r.sigma_z)
        gap.append(abs(optimal_action(r.z, int(r.state), int(r.t), cache[r.seed][0], cache[r.seed][3])[1]))
    A["gap"] = gap; A["gc"] = A.opt == A.greedy; A["err"] = A.model != A.opt; A["reg"] = np.where(A.err, A.gap, 0.0)
    A["complete"] = A.cls.isin(["final", "final_marked"]); A["sw"] = A.model != A.state; A["need"] = A.opt != A.state
    A["clear"] = (A.alpha*A.z/A.lam).abs() > 1
    A["dir"] = np.where(A.state == 0, 1, -1)*A.alpha*A.z/A.lam; A["ra"] = A.lam/A.alpha
    return A
def completeness():
    import statsmodels.api as sm
    A = decision_table()
    out = {}
    for key, name in LABELS.items():
        d = A[A.file == key]; n = d.seed.nunique(); c = d[d.complete]; u = d[~d.complete]
        agree = lambda x: [int((~x.err).sum()), int(len(x))]
        out[name] = dict(responses=int(len(d)), cut_off=int((~d.complete).sum()), no_decision=int((d.cls == "none").sum()),
            cut_off_share_gc=float((~d[d.gc].complete).mean()), cut_off_share_cs=float((~d[~d.gc].complete).mean()),
            median_len_complete=float(c.length.median()), median_len_cut=float(u.length.median()), endpunct_cut=float(u.endpunct.mean()),
            regret=float(d.reg.sum()/n), regret_complete_gc=float(c[c.gc].reg.sum()/n), regret_complete_cs=float(c[~c.gc].reg.sum()/n),
            regret_cut_gc=float(u[u.gc].reg.sum()/n), regret_cut_cs=float(u[~u.gc].reg.sum()/n),
            agree_complete_gc=agree(c[c.gc]), agree_complete_cs=agree(c[~c.gc]), agree_cut_gc=agree(u[u.gc]), agree_cut_cs=agree(u[~u.gc]),
            unnecessary_switches=[int((c.sw & ~c.need).sum()), int((u.sw & ~u.need).sum())], missed_switches=[int((~c.sw & c.need).sum()), int((~u.sw & u.need).sum())],
            follows_greedy_clear=[float((c[c.clear].model == c[c.clear].greedy).mean()), float((u[u.clear].model == u[u.clear].greedy).mean())])
    # horizon regression on complete responses (specification of the horizon section)
    COV = ["dir", "rem", "kappa", "ra", "T", "sigma_z", "state"]
    for key, name in LABELS.items():
        b = A[(A.file == key) & A.complete & (A.dir > 0.25) & (A.dir <= 1.0) & (A.rem >= 2)]
        r = sm.Logit(b.sw.astype(int).values, sm.add_constant(b[COV].astype(float).values)).fit(disp=0, maxiter=500, cov_type="cluster", cov_kwds={"groups": b.seed.values})
        out[name]["horizon_complete"] = dict(coef=float(r.params[2]), se=float(r.bse[2]), p=float(r.pvalues[2]), n=int(len(b)), switch_rate=float(b.sw.mean()), opt_switch_rate=float(b.need.mean()))
    # matched states where both runs' responses are complete
    D = A[A.file == "opus_n500"].set_index(["seed", "t"]); H = A[A.file == "opus_hint_n500"].set_index(["seed", "t"])
    J = D.join(H[["state", "model", "complete"]], rsuffix="_h", how="inner"); M = J[(J.state == J.state_h) & J.complete & J.complete_h]
    out["matched_both_complete"] = dict(n=int(len(M)), gc=[float((M[M.gc].model == M[M.gc].opt).mean()), float((M[M.gc].model_h == M[M.gc].opt).mean())],
                                         cs=[float((M[~M.gc].model == M[~M.gc].opt).mean()), float((M[~M.gc].model_h == M[~M.gc].opt).mean())], n_cs=int((~M.gc).sum()))
    save("completeness", out); print(json.dumps(out, indent=1))

# ---------------------------------------------------------------- results on scored decisions (Tables 1-2, Figure 4)
def scored():
    """All model results of the paper, computed on scored decisions: responses that state a decision within the output limit.
    Unscored responses (no decision stated within the limit) are excluded from agreement and regret."""
    import statsmodels.api as sm
    A = decision_table(); A[["file", "seed", "t", "complete"]].to_csv(os.path.join(OUT, "scored_flags.csv"), index=False)
    rng = np.random.default_rng(0); out = {"table1": {}, "horizon": {}}
    agree = lambda x: [int((~x.err).sum()), int(len(x))]
    for key, name in LABELS.items():
        d = A[A.file == key]; c = d[d.complete]; per = c.groupby("seed").agg(gc=("reg", lambda v: v[c.loc[v.index, "gc"]].sum()), cs=("reg", lambda v: v[~c.loc[v.index, "gc"]].sum()))
        per = per.reindex(range(500), fill_value=0.0); tot = per.gc + per.cs
        bs = [tot.values[rng.integers(0, 500, 500)].mean() for _ in range(2000)]
        g = c.groupby("seed").apply(lambda x: x.gap[x.greedy != x.opt].sum()).reindex(range(500), fill_value=0.0)
        out["table1"][name] = dict(unscored=int((~d.complete).sum()), unscored_share=float((~d.complete).mean()), unscored_share_cs=float((~d[~d.gc].complete).mean()),
            unscored_share_gc=float((~d[d.gc].complete).mean()), agree_gc=agree(c[c.gc]), agree_cs=agree(c[~c.gc]), agree_all=agree(c),
            regret=float(tot.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], regret_gc=float(per.gc.mean()), regret_cs=float(per.cs.mean()),
            greedy_action_loss_same_states=float(g.mean()))
    # case study: matched states where both runs state a decision
    D = A[A.file == "opus_n500"].set_index(["seed", "t"]); H = A[A.file == "opus_hint_n500"].set_index(["seed", "t"])
    J = D.join(H[["state", "model", "complete", "reg", "err", "sw"]], rsuffix="_h", how="inner"); M = J[(J.state == J.state_h) & J.complete & J.complete_h].reset_index()
    diffs = {}
    for lab, x in [("all", M), ("gc", M[M.gc]), ("cs", M[~M.gc])]:   # episode-level bootstrap of the per-decision difference
        P = x.groupby("seed").agg(d=("reg", "sum"), h=("reg_h", "sum"), n=("reg", "size")); idx = rng.integers(0, len(P), (4000, len(P)))
        diffs[lab] = (P.h.values[idx].sum(1) - P.d.values[idx].sum(1))/P.n.values[idx].sum(1)
    cs_ = {}
    for lab, x in [("all", M), ("gc", M[M.gc]), ("cs", M[~M.gc])]:
        v = np.array(diffs[lab]); cs_[lab] = dict(n=int(len(x)), loss_default=float(x.reg.mean()), loss_instr=float(x.reg_h.mean()), diff=float(x.reg_h.mean() - x.reg.mean()),
            ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], p=float(2*min((v > 0).mean(), (v < 0).mean())),
            agree_default=float((x.model == x.opt).mean()), agree_instr=float((x.model_h == x.opt).mean()))
    cs_["pairs_total"] = int(len(J)); cs_["pairs_same_state"] = int((J.state == J.state_h).sum()); out["case_study"] = cs_
    # horizon, scored band states, final step excluded
    COV = ["dir", "rem", "kappa", "ra", "T", "sigma_z", "state"]
    def logit(b, y, cols, coef):
        r = sm.Logit(y, sm.add_constant(b[cols].astype(float).values)).fit(disp=0, maxiter=500, cov_type="cluster", cov_kwds={"groups": b.seed.values})
        i = cols.index(coef) + 1; return dict(coef=float(r.params[i]), se=float(r.bse[i]), p=float(r.pvalues[i]), n=int(len(b)))
    band = lambda d: d[(d.dir > 0.25) & (d.dir <= 1.0) & (d.rem >= 2)]
    bins = [(2, 4), (5, 8), (9, 12), (13, 20)]
    for key, name in LABELS.items():
        b = band(A[(A.file == key) & A.complete]); c2 = [x for x in COV if x != "state"]
        out["horizon"][name] = dict(model=logit(b, b.sw.astype(int).values, COV, "rem"), opt=logit(b, b.need.astype(int).values, COV, "rem"),
            split={lab: dict(model=logit(bb, bb.sw.astype(int).values, c2, "rem"), opt=logit(bb, bb.need.astype(int).values, c2, "rem")) for lab, bb in [("off", b[b.state == 0]), ("on", b[b.state == 1])]},
            rates=[dict(bin=f"{lo}-{hi}", n=int(len(x)), model=float(x.sw.mean()), opt=float(x.need.mean())) for lo, hi in bins for x in [b[(b.rem >= lo) & (b.rem <= hi)]]],
            switch_rate=float(b.sw.mean()), opt_rate=float(b.need.mean()))
    Mb = band(M); Sd = Mb.copy(); Sh = Mb.copy(); Sd["instr"] = 0; Sh["instr"] = 1; Sh["sw"] = Sh.sw_h; S = pd.concat([Sd, Sh]); S["ixr"] = S.instr*S.rem
    out["horizon"]["matched"] = dict(n=int(len(Mb)), instr=logit(Mb, Mb.sw_h.astype(int).values, COV, "rem"), default=logit(Mb, Mb.sw.astype(int).values, COV, "rem"),
        interaction=logit(S, S.sw.astype(int).values, COV + ["instr", "ixr"], "ixr"), main=logit(S, S.sw.astype(int).values, COV + ["instr", "ixr"], "instr"))
    save("scored", out); print(json.dumps(out, indent=1))

# ---------------------------------------------------------------- agreement by stage of the episode
def bystep():
    """Continuation-sensitive agreement on scored decisions, by step t of the episode."""
    A = decision_table(); A = A[A.complete]; bins = [(0, 1), (2, 4), (5, 9), (10, 19)]; out = {}
    for key, name in LABELS.items():
        cs = A[(A.file == key) & ~A.gc]
        out[name] = [dict(steps=f"{lo}-{hi}", agree=int((~c.err).sum()), n=int(len(c))) for lo, hi in bins for c in [cs[(cs.t >= lo) & (cs.t <= hi)]]]
        out[name + " share_regret_t>=5"] = float(cs[cs.t >= 5].reg.sum()/cs.reg.sum())
    # value at stake of scored vs excluded continuation-sensitive decisions, and position changes by excluded responses (App. C.5, S5.3)
    B = decision_table(); B["stake"] = B.gap/B.lam
    for key, name in LABELS.items():
        cs = B[(B.file == key) & ~B.gc]; u = B[(B.file == key) & ~B.complete]
        out[name + " stake_median_scored_excluded"] = [float(cs[cs.complete].stake.median()), float(cs[~cs.complete].stake.median())]
        out[name + " excluded_changed_position"] = int((u.model != u.state).sum())
    # regret if each excluded decision lost the mean of the scored decisions in its class; new errors under the instruction
    for key, name in LABELS.items():
        g = B[B.file == key]; sc, ex = g[g.complete], g[~g.complete]
        imp = sum(sc[sc.gc == c].reg.mean()*(ex.gc == c).sum() for c in (True, False))
        out[name + " imputed_regret_increase"] = float(imp/sc.reg.sum())
    e = B[(B.file == "opus_hint_n500") & B.complete & B.gc & B.err]
    out["instr GC errors: n, switches, cost share from ON"] = [int(len(e)), int((e.model != e.state).sum()), float(e[e.state == 1].reg.sum()/e.reg.sum())]
    # parser check: in every scored response the last match is the current decision
    bad = 0
    for f, key in RUNS:
        for ep in json.load(open(os.path.join(DATA, f)))["instances"]:
            for t, txt in enumerate(ep["raw_texts"]):
                if classify(txt) in ("final", "final_marked"):
                    tok = re.match(r"s(?:_\{?[\w\-]*\}?)?", list(re.finditer(PAT, txt))[-1].group(0)).group(0).replace("{", "").replace("}", "")
                    bad += tok not in ("s", "s_t", f"s_{t}")
    out["scored responses whose last match is not the current decision"] = bad
    # power of a one-sided 5% test of white noise (phi = 0) from n observations, App. C.3
    rng = np.random.default_rng(7)
    def phis(k, n, N=200000):
        z = np.zeros((N, n)); z[:, 0] = rng.normal(0, 0.1/np.sqrt(2*k - k*k), N)
        for t in range(1, n): z[:, t] = (1 - k)*z[:, t-1] + 0.1*rng.normal(size=N)
        return (z[:, :-1]*z[:, 1:]).sum(1)/(z[:, :-1]**2).sum(1)
    for n in (8, 12):
        c = np.quantile(phis(1.0, n), 0.95); out[f"white-noise test power, n={n}, kappa=0.1/0.175/0.25"] = [float((phis(k, n) > c).mean()) for k in (0.1, 0.175, 0.25)]
    # S5.3: share of the instructed run's new greedy-consistent errors out of ON that follow a fall in the signal
    B = B.sort_values(["file", "seed", "t"]); B["dz"] = B.groupby(["file", "seed"]).z.diff()
    e = B[(B.file == "opus_hint_n500") & B.complete & B.gc & B.err & (B.state == 1)]
    out["instr GC errors out of ON after a fall in z"] = float((e.dz < 0).mean())
    # App. B: responses without a decision, by the position they were made from
    nm = []
    for ep in json.load(open(os.path.join(DATA, "opus4_instruction.json")))["instances"]:
        for t, txt in enumerate(ep["raw_texts"]):
            if classify(txt) == "none": nm.append((int(ep["seed"]), t))
    S = B[B.file == "opus_hint_n500"].set_index(["seed", "t"]).state
    out["no-decision responses: n, made from ON"] = [len(nm), int(sum(S.loc[k] == 1 for k in nm))]
    save("bystep", out); print(json.dumps(out, indent=1))

# ---------------------------------------------------------------- numerics outside the evaluation regime
def _interp_lin(x, xp, fp):
    y = np.interp(x, xp, fp)
    y = np.where(x < xp[0], fp[0] + (x - xp[0])*(fp[1] - fp[0])/(xp[1] - xp[0]), y)
    return np.where(x > xp[-1], fp[-1] + (x - xp[-1])*(fp[-1] - fp[-2])/(xp[-1] - xp[-2]), y)
def _solve(k, a, lam, T, sz, zg, extrap):
    n = len(zg); V = np.zeros((T+1, n, 2)); Q = np.zeros((T, n, 2, 2)); f = _interp_lin if extrap else np.interp
    for t in range(T-1, -1, -1):
        zp = ((1-k)*zg)[:, None] + sz*GH_EPS[None, :]
        cont = np.stack([f(zp.ravel(), zg, V[t+1, :, b]).reshape(zp.shape) @ GH_PW for b in (0, 1)], 1)
        for s in (0, 1):
            for b in (0, 1): Q[t, :, s, b] = b*a*zg - lam*(b != s) + cont[:, b]
            V[t, :, s] = Q[t, :, s].max(1)
    return zg, None, V, Q
def released(k, a, lam, T, sz):
    sd = sz/np.sqrt(2*k); return _solve(k, a, lam, T, sz, np.linspace(-4*sd, 4*sd, 200), False)
def accurate(k, a, lam, T, sz, r):
    sd = sz/np.sqrt(2*k - k*k); W = max(12*sd, 3*r); return _solve(k, a, lam, T, sz, np.linspace(-W, W, 1600), True)
def act_nearest(z, s, t, sol):
    zg, Q = sol[0], sol[3]; i = int(np.argmin(np.abs(zg - z))); return int(Q[t, i, s, 1] > Q[t, i, s, 0])
def numerics():
    out = {}; rng = np.random.default_rng(7); dis = n = outside = 0; worst = 0.0
    for _ in range(400):
        k = rng.uniform(0.01, 0.10); r = rng.uniform(2, 5); a = rng.uniform(0.1, 0.5); sz = rng.uniform(0.1, 0.3); rng.uniform(0.1, 0.3)
        T = int(rng.integers(5, 21)); lam = a*r; s0 = int(rng.integers(0, 2))
        z = np.empty(T); z[0] = rng.normal(0, sz/np.sqrt(2*k))
        for t in range(1, T): z[t] = (1-k)*z[t-1] + sz*rng.normal()
        rng.normal(size=T)
        Rs, As = released(k, a, lam, T, sz), accurate(k, a, lam, T, sz, r); outside += int((np.abs(z) > 4*sz/np.sqrt(2*k)).sum())
        s = s0
        for t in range(T):
            ar = act_nearest(z[t], s, t, Rs); q0, q1 = qpair(z[t], t, s, As); aa = int(q1 > q0); n += 1
            if ar != aa: dis += 1; worst = max(worst, abs(q1 - q0)/lam)
            s = greedy_action(z[t], s, a, lam)
    out["high_cost"] = dict(states=n, action_disagree=dis/n, max_gap_over_lambda=worst, states_outside_grid=outside/n)
    save("numerics", out); print(json.dumps(out, indent=1))

# ---------------------------------------------------------------- regimes and difficulty surface
def _greedy_episode(rng, k, r):
    a = rng.uniform(0.1, 0.5); sz = rng.uniform(0.1, 0.3); T = int(rng.integers(5, 21)); lam = a*r; s0 = int(rng.integers(0, 2))
    z = np.empty(T); z[0] = rng.normal(0, sz/np.sqrt(2*k))
    for t in range(1, T): z[t] = (1-k)*z[t-1] + sz*rng.normal()
    sol = solve_bellman(k, a, lam, T, sz); v0 = np.interp(z[0], sol[0], sol[2][0, :, s0]); s = s0; cs = 0; reg = 0.0
    for t in range(T):
        ao, gap = optimal_action(z[t], s, t, sol[0], sol[3]); g = greedy_action(z[t], s, a, lam); cs += ao != g
        if g != ao: reg += abs(gap)
        s = g
    return v0, reg, cs, T
def regimes():
    kap = [0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]; rs = [0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    H = np.zeros((len(kap), len(rs))); G = np.zeros_like(H); rng = np.random.default_rng(3)
    for i, k in enumerate(kap):
        for j, r in enumerate(rs):
            res = [_greedy_episode(rng, k, r) for _ in range(30)]
            H[i, j] = sum(x[2] for x in res)/sum(x[3] for x in res); G[i, j] = np.mean([x[1] for x in res])
    out = dict(kap=kap, rs=rs, H=H.tolist(), G=G.tolist())
    for name, kr, rr in [("evaluation", (0.1, 0.25), (0.3, 0.5)), ("high_cost", (0.01, 0.10), (2, 5))]:
        rng = np.random.default_rng(17); R = [_greedy_episode(rng, rng.uniform(*kr), rng.uniform(*rr)) for _ in range(2000)]
        v0 = np.array([x[0] for x in R]); reg = np.array([x[1] for x in R])
        out[name] = dict(V0=float(v0.mean()), greedy_regret=float(reg.mean()), se=float(reg.std()/np.sqrt(len(reg))), cs_frac=float(sum(x[2] for x in R)/sum(x[3] for x in R)))
    save("regimes", out); print(json.dumps({k: out[k] for k in ("evaluation", "high_cost")}, indent=1))


# ---------------------------------------------------------------- validation of the reference (S4, App. C.1)
def reference():
    import solver as S
    def setq(n):
        X, W = np.polynomial.hermite.hermgauss(n); S.GH_EPS = X*np.sqrt(2); S.GH_PW = W/np.sqrt(np.pi)
    def sol(r, n_grid=200, width=5, nq=20):
        setq(nq); sd = r.sigma_z/np.sqrt(1 - (1 - r.kappa)**2); g = np.linspace(-width*sd, width*sd, n_grid)
        out = S.solve_bellman(r.kappa, r.alpha, r.lam, int(r["T"]), r.sigma_z, zgrid=g); setq(20); return out
    V = {"base": {}, "grid400": dict(n_grid=400), "grid800": dict(n_grid=800), "quad40": dict(nq=40), "grid800_quad40": dict(n_grid=800, nq=40), "wider": dict(width=10, n_grid=400)}
    ref = {k: dict(actions=0, classes=0, max_gap_change_over_lam=0.0, max_tie_distance_of_changed=0.0, changed=[]) for k in V if k != "base"}
    d = df[df.file == "opus_n500"]; same = 0
    for seed, ep in d.groupby("seed"):
        r0 = ep.iloc[0]; sols = {k: sol(r0, **kw) for k, kw in V.items()}
        for x in ep.itertuples():
            t, s, z = int(x.t), int(x.state), x.z; b = optimal_action(z, s, t, sols["base"][0], sols["base"][3]); gb = greedy_action(z, s, x.alpha, x.lam)
            same += b[0] == x.opt
            for k, v in ref.items():
                a = optimal_action(z, s, t, sols[k][0], sols[k][3]); v["max_gap_change_over_lam"] = max(v["max_gap_change_over_lam"], abs(abs(a[1]) - abs(b[1]))/x.lam)
                if a[0] != b[0]: v["actions"] += 1; v["changed"].append([int(seed), t]); v["max_tie_distance_of_changed"] = max(v["max_tie_distance_of_changed"], abs(b[1])/x.lam)
                v["classes"] += (a[0] == gb) != (b[0] == gb)
    out = dict(states=int(len(d)), released_vs_reimplementation_same_action=same/len(d), refinements=ref, stakes={})
    for key, name in list(LABELS.items())[:3]:   # value at stake at the continuation-sensitive states each default model visited
        c = df[(df.file == key) & (df.opt != df.greedy)]; cache = {}; g = []
        for r in c.itertuples():
            if r.seed not in cache: cache[r.seed] = solve_bellman(r.kappa, r.alpha, r.lam, int(r.T), r.sigma_z)
            g.append(abs(optimal_action(r.z, int(r.state), int(r.t), cache[r.seed][0], cache[r.seed][3])[1])/r.lam)
        g = np.array(g); out["stakes"][name] = dict(n=int(len(g)), median=float(np.median(g)), iqr=[float(np.percentile(g, 25)), float(np.percentile(g, 75))], within_005=float((g < 0.05).mean()), within_010=float((g < 0.10).mean()))
    save("reference", out); print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- identification of kappa within one episode (S4, App. C.3 table)
def identify():
    rng = np.random.default_rng(5); N = 200000; out = {}
    def khat(k, n):
        z = np.zeros((N, n)); z[:, 0] = rng.normal(0, 0.1/np.sqrt(2*k - k*k), N)
        for t in range(1, n): z[:, t] = (1 - k)*z[:, t-1] + 0.1*rng.normal(size=N)
        return 1 - (z[:, :-1]*z[:, 1:]).sum(1)/(z[:, :-1]**2).sum(1)   # least-squares kappa_hat; invariant to sigma_z
    for k in (0.10, 0.175, 0.25):
        out[f"sd_kappa_{k}"] = {n: float(khat(k, n).std()) for n in (5, 8, 12, 16, 25)}
    out["p_within_0.075_kappa_0.175"] = {n: float((np.abs(khat(0.175, n) - 0.175) < 0.075).mean()) for n in (5, 8, 12, 16, 25)}
    save("identify", out); print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- further checks (Table 1, App. C.2, App. C.5)
def checks():
    out = {}
    # optimal expected utility on the 500 evaluation episodes (Table 1 caption)
    d = df[df.file == "opus_n500"].sort_values(["seed", "t"]); V0 = []
    for seed, ep in d.groupby("seed"):
        r0 = ep.iloc[0]; zg, _, V, _ = solve_bellman(r0.kappa, r0.alpha, r0.lam, int(r0["T"]), r0.sigma_z); V0.append(np.interp(r0.z, zg, V[0, :, int(r0.state)]))
    out["E_V0_evaluation_episodes"] = float(np.mean(V0))
    # stricter scoring rule: score only responses that end with their decision (App. C.5)
    A = decision_table(); dev = dict(agreement_points=0.0, regret_per_episode=0.0)
    for key in list(LABELS)[:3]:   # the default runs of Table 1
        x = A[A.file == key]
        res = []
        for m in (x.complete, x.cls == "final"):
            c = x[m]; res.append([100*(~c[c.gc].err).mean(), 100*(~c[~c.gc].err).mean(), c.reg.sum()/500])
        dev["agreement_points"] = max(dev["agreement_points"], abs(res[0][0] - res[1][0]), abs(res[0][1] - res[1][1]))
        dev["regret_per_episode"] = max(dev["regret_per_episode"], abs(res[0][2] - res[1][2]))
    out["strict_rule_max_change"] = dev
    # heuristic H1 (rho = 0.55, calibrated in heuristics.py) carried to the high-cost regime of Section 4
    rng = np.random.default_rng(17); agree = n = 0; rho = 0.55
    for _ in range(2000):
        k = rng.uniform(0.01, 0.10); r = rng.uniform(2, 5); a = rng.uniform(0.1, 0.5); sz = rng.uniform(0.1, 0.3); T = int(rng.integers(5, 21)); lam = a*r; s = int(rng.integers(0, 2))
        z = np.empty(T); z[0] = rng.normal(0, sz/np.sqrt(2*k))
        for t in range(1, T): z[t] = (1-k)*z[t-1] + sz*rng.normal()
        sol = solve_bellman(k, a, lam, T, sz)
        for t in range(T):
            ao = optimal_action(z[t], s, t, sol[0], sol[3])[0]; g = greedy_action(z[t], s, a, lam); m = (1 - rho**(T - t))/(1 - rho)
            h = (1 if a*z[t]*m > lam else 0) if s == 0 else (0 if a*z[t]*m < -lam else 1)
            if ao != g: n += 1; agree += h == ao
            s = g
    out["H1_high_cost_regime"] = dict(cs_decisions=n, agreement=agree/n)
    save("checks", out); print(json.dumps(out, indent=1))

# ---------------------------------------------------------------- Figure 1: an example episode
def episode():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from solver import simulate_signal, run_policy
    BLUE, ORANGE, AQUA, MUTED, GRID, INK = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#e1e0d9", "#0b0b0b"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "font.family": "serif"})
    ALPHA, SX, kappa, r, T, SZ = 0.37, 0.1, 0.15, 0.4, 20, 0.1; lam = ALPHA*r; seed = 161   # seed chosen from 0-399 for a large optimum-greedy gap
    zgrid, _, _, Q = solve_bellman(kappa, ALPHA, lam, T, SZ)
    rng = np.random.default_rng(seed); z = simulate_signal(kappa, SZ, T, rng, z0=0.0); x = ALPHA*z + SX*rng.normal(size=T)
    Jo, ao = run_policy(z, x, lambda t, zz, ss: optimal_action(zz, ss, t, zgrid, Q)[0], lam)
    Jg, ag = run_policy(z, x, lambda t, zz, ss: greedy_action(zz, ss, ALPHA, lam), lam)
    hard = []; s = 0
    for t in range(T): hard.append(optimal_action(z[t], s, t, zgrid, Q)[0] != ag[t]); s = ag[t]
    cum = lambda a: np.cumsum([a[t]*x[t] - lam*(a[t] != (a[t-1] if t else 0)) for t in range(T)])
    fig, axs = plt.subplots(3, 1, figsize=(3.4, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.2, 2], hspace=0.15)); tt = np.arange(T)
    axs[0].axhline(0, color="#c3c2b7", lw=0.6); axs[0].axhline(r, color=MUTED, lw=0.6, ls="--"); axs[0].plot(tt, z, color=BLUE, lw=1.4); axs[0].set_ylabel("signal $z_t$")
    axs[0].text(T - 0.5, r, r"greedy threshold $\lambda/\alpha$", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    for row, (acts, col) in enumerate([(ao, AQUA), (ag, ORANGE)]):
        for t in range(T):
            axs[1].add_patch(plt.Rectangle((t - 0.5, 1 - row - 0.4), 1, 0.8, facecolor=col if acts[t] else "#eeeeea", edgecolor="#d03b3b" if (hard[t] and row == 1) else "white", lw=1.0 if hard[t] and row == 1 else 0.4))
    axs[1].set_ylim(-0.6, 1.6); axs[1].set_yticks([1, 0]); axs[1].set_yticklabels(["optimal", "greedy"]); axs[1].set_xlim(-0.5, T - 0.5)
    axs[2].plot(tt, cum(ao), color=AQUA, lw=1.4, label=f"optimal  J={Jo:.2f}"); axs[2].plot(tt, cum(ag), color=ORANGE, lw=1.4, label=f"greedy   J={Jg:.2f}")
    axs[2].axhline(0, color="#c3c2b7", lw=0.6); axs[2].set_ylabel("cumulative $J$"); axs[2].set_xlabel("step $t$"); axs[2].legend(loc="upper left", fontsize=6.5)
    for a in axs: a.grid(axis="y")
    fig.savefig(os.path.join(ROOT, "paper", "fig1_episode.pdf"), bbox_inches="tight")
    save("episode", dict(seed=seed, J_opt=float(Jo), J_greedy=float(Jg), first_optimal_on=int(np.argmax(ao)), first_greedy_on=int(np.argmax(ag)))); print(f"Figure 1: J_opt={Jo:.2f}, J_greedy={Jg:.2f}")

# ---------------------------------------------------------------- figures
def figures():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, MUTED, GRID, INK, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#e1e0d9", "#0b0b0b", "#4a3aa7"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "font.family": "serif"})
    FIG = os.path.join(ROOT, "paper")
    # Figure 3
    d = df[(df.file == "opus_n500") & (df.opt != df.greedy)]; cache = {}; gaps = []
    for r in d.itertuples():
        if r.seed not in cache: cache[r.seed] = solve_bellman(r.kappa, r.alpha, r.lam, int(r.T), r.sigma_z)
        zg, Q = cache[r.seed][0], cache[r.seed][3]; gaps.append(abs(optimal_action(r.z, int(r.state), int(r.t), zg, Q)[1])/r.lam)
    gaps = np.array(gaps); ig = load("infogap")["onpolicy"]; fx = {k: v for k, v in ig.items() if k.startswith("(")}
    best = min(fx.values(), key=lambda v: v["mean"]); worst = max(fx.values(), key=lambda v: v["mean"])
    wp = load("wideprior")["onpolicy"]
    rows = [("Bayesian learner, range prior", ig["bayes"]["mean"], 1.96*ig["bayes"]["se"], AQUA), ("Bayesian learner, wide prior", wp["mean"], 1.96*wp["se"], AQUA), ("best fixed planner", best["mean"], 1.96*best["se"], AQUA),
            ("worst fixed planner", worst["mean"], 1.96*worst["se"], AQUA), ("greedy rule", ig["greedy"]["mean"], 1.96*ig["greedy"]["se"], ORANGE)]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.0), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.8))
    axs[0].hist(gaps, bins=np.linspace(0, 1.2, 31), color=BLUE, edgecolor="white", lw=0.4); axs[0].axvline(np.median(gaps), color=ORANGE, lw=1.2)
    axs[0].text(np.median(gaps) + 0.03, axs[0].get_ylim()[1]*0.88, f"median {np.median(gaps):.2f}", color=ORANGE, fontsize=7)
    axs[0].set_xlabel(r"$|Q_{\rm ON}-Q_{\rm OFF}|\,/\,\lambda$, continuation-sensitive decisions"); axs[0].set_ylabel("count"); axs[0].grid(axis="y")
    yp = np.arange(len(rows))[::-1]
    for y, (lab, v, e, c) in zip(yp, rows):
        axs[1].barh(y, v, color=c, height=0.6); axs[1].errorbar(v, y, xerr=e, color=INK, lw=0.7, capsize=2)
        axs[1].text(v*1.45 + e, y, f"{v:.4f}" if v < 0.01 else f"{v:.3f}", va="center", fontsize=6.5, color=INK)
    axs[1].set_xscale("log"); axs[1].set_xlim(2e-4, 0.6); axs[1].set_yticks(yp); axs[1].set_yticklabels([r[0] for r in rows], fontsize=6.8)
    axs[1].set_xlabel("expected regret per episode (log scale)"); axs[1].grid(axis="x"); fig.savefig(os.path.join(FIG, "fig3_kappa.pdf"), bbox_inches="tight")
    # Figure 2 (difficulty surface)
    S = load("regimes"); kap = np.array(S["kap"]); rs = np.array(S["rs"]); H = np.array(S["H"]); G = np.array(S["G"])
    def edges(v):
        lv = np.log(v); m = (lv[1:] + lv[:-1])/2; return np.exp(np.concatenate([[2*lv[0] - m[0]], m, [2*lv[-1] - m[-1]]]))
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw=dict(wspace=0.35))
    for ax, M, title in [(axs[0], H*100, "continuation-sensitive decisions (%), greedy trajectory"), (axs[1], G, r"expected utility gap $\mathbb{E}[V^*_0]-\mathbb{E}[V^{\rm greedy}_0]$")]:
        im = ax.pcolormesh(edges(rs), edges(kap), M, cmap="Blues", vmin=0, shading="flat"); ax.set_xscale("log"); ax.set_yscale("log")
        xt = [0.2, 0.3, 0.5, 1, 2, 3, 5]; yt = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
        ax.set_xticks(xt); ax.set_xticklabels([f"{r:g}" for r in xt], fontsize=6.5); ax.set_yticks(yt); ax.set_yticklabels([f"{k:g}" for k in yt], fontsize=6.5)
        ax.minorticks_off(); ax.set_xlabel(r"cost ratio $r=\lambda/\alpha$"); ax.set_title(title, fontsize=7.5)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
        for (k0, k1, r0, r1, col, lab) in [(0.10, 0.25, 0.3, 0.5, ORANGE, "evaluation"), (0.01, 0.10, 2.0, 5.0, AQUA, "high-cost")]:
            ax.add_patch(plt.Rectangle((r0, k0), r1 - r0, k1 - k0, fill=False, edgecolor=col, lw=1.4))
            ax.text(r0, k0/1.07 if lab == "evaluation" else k1*1.06, lab, color=col, fontsize=6.5, va="top" if lab == "evaluation" else "bottom", ha="left", fontweight="bold")
    axs[0].set_ylabel(r"mean-reversion speed $\kappa$"); fig.savefig(os.path.join(FIG, "fig2_surface.pdf"), bbox_inches="tight")
    # Figure 4 (horizon response)
    flags = pd.read_csv(os.path.join(OUT, "scored_flags.csv"))   # written by the scored section
    d5 = df.merge(flags, on=["file", "seed", "t"]); d5 = d5[d5.complete].copy(); d5["zr"] = d5.alpha*d5.z/d5.lam
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.0), sharey=True, gridspec_kw=dict(wspace=0.12))
    rb = [(2, 4), (5, 8), (9, 12), (13, 20)]; xs = np.arange(4)
    series = [("Bellman optimum", "opt", "opus_n500", AQUA), ("Opus 4, default", "model", "opus_n500", ORANGE), ("Opus 4, planning instruction", "model", "opus_hint_n500", VIOLET), ("Sonnet 4", "model", "sonnet_n500", BLUE), ("Haiku 4.5", "model", "haiku_n500", MUTED)]
    for ax, (state, lo, hi, title) in zip(axs, [(0, 0.25, 1.0, r"OFF, $0.25<\alpha z_t/\lambda\leq 1$: P(switch ON)"), (1, -1.0, -0.25, r"ON, $-1\leq\alpha z_t/\lambda<-0.25$: P(switch OFF)")]):
        for name, who, f, col in series:
            dd = d5[(d5.file == f) & (d5.state == state) & (d5.zr > lo) & (d5.zr <= hi)] if state == 0 else d5[(d5.file == f) & (d5.state == state) & (d5.zr >= lo) & (d5.zr < hi)]
            ys = [(lambda e: e[who].mean() if state == 0 else 1 - e[who].mean())(dd[(dd.rem >= a_) & (dd.rem <= b_)]) for a_, b_ in rb]
            ax.plot(xs, ys, marker="o", ms=4, lw=2.2 if name.startswith("Bell") else 1.6, color=col, label=name, ls="--" if name.startswith("Bell") else "-")
        ax.set_xticks(xs); ax.set_xticklabels(["2–4", "5–8", "9–12", "13–20"]); ax.set_xlabel(r"steps remaining $n=T-t$"); ax.set_title(title, fontsize=8); ax.grid(axis="y"); ax.set_ylim(0, 0.7)
    axs[0].set_ylabel("switch probability"); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, ncol=5, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(os.path.join(FIG, "fig4_horizon.pdf"), bbox_inches="tight"); print("figures written to paper/")

if __name__ == "__main__":
    SECTIONS = dict(reference=reference, identify=identify, numerics=numerics, regimes=regimes, infogap=infogap, wideprior=wideprior, completeness=completeness, scored=scored, bystep=bystep, checks=checks, figures=figures, episode=episode)
    for s in (sys.argv[1:] or list(SECTIONS)): SECTIONS[s]()
