"""Re-implementation of the regime-switching benchmark from the paper's equations.

J(s) = sum_t [ s_t * x_t - lam * 1{s_t != s_{t-1}} ],  s_{-1} = 0
z_{t+1} = z_t + kappa (theta - z_t) + sigma_z eps_t,   theta = 0
x_t = alpha z_t + sigma_x eta_t
Bellman: V_t(z,s) = max_a [ a alpha z - lam 1{a!=s} + E V_{t+1}(z', a) ],  V_T = 0
Expectation via Gauss-Hermite (20 nodes) on a z-grid (200 points), linear interpolation.
Greedy: V_{t+1} = 0.
"""
import numpy as np

GH_X, GH_W = np.polynomial.hermite.hermgauss(20)  # for exp(-x^2) weight
# E f(mu + sigma eps), eps~N(0,1): sum w_i f(mu + sigma sqrt2 x_i)/sqrt(pi)
GH_EPS = GH_X * np.sqrt(2.0)
GH_PW = GH_W / np.sqrt(np.pi)


def solve_bellman(kappa, alpha, lam, T, sigma_z, theta=0.0, zgrid=None, n_grid=200):
    """Return policy table pi[t, i, s] in {0,1} on zgrid, and values V[t, i, s]."""
    if zgrid is None:
        sd_stat = sigma_z / np.sqrt(1 - (1 - kappa) ** 2)
        zgrid = np.linspace(-5 * sd_stat, 5 * sd_stat, n_grid)
    n = len(zgrid)
    V = np.zeros((T + 1, n, 2))
    pi = np.zeros((T, n, 2), dtype=int)
    Q = np.zeros((T, n, 2, 2))  # Q[t,i,s,a]
    for t in range(T - 1, -1, -1):
        # continuation: E V_{t+1}(z', a) for each z on grid and a
        mu = zgrid + kappa * (theta - zgrid)
        zp = mu[:, None] + sigma_z * GH_EPS[None, :]  # (n, 20)
        cont = np.zeros((n, 2))
        for a in (0, 1):
            vals = np.interp(zp, zgrid, V[t + 1, :, a])  # linear interp, clamp at edges
            cont[:, a] = vals @ GH_PW
        for s in (0, 1):
            for a in (0, 1):
                Q[t, :, s, a] = a * alpha * zgrid - lam * (a != s) + cont[:, a]
            pi[t, :, s] = np.argmax(Q[t, :, s, :], axis=1)
            V[t, :, s] = np.max(Q[t, :, s, :], axis=1)
    return zgrid, pi, V, Q


def greedy_action(z, s, alpha, lam):
    if s == 0:
        return 1 if alpha * z > lam else 0
    return 0 if alpha * z < -lam else 1


def optimal_action(z, s, t, zgrid, Q):
    """Evaluate Q at z by interpolation and pick argmax (avoids grid snapping)."""
    q0 = np.interp(z, zgrid, Q[t, :, s, 0])
    q1 = np.interp(z, zgrid, Q[t, :, s, 1])
    return (1 if q1 > q0 else 0), (q1 - q0)


def simulate_signal(kappa, sigma_z, T, rng, z0=None, theta=0.0):
    sd_stat = sigma_z / np.sqrt(1 - (1 - kappa) ** 2)
    z = np.empty(T)
    z[0] = rng.normal(0, sd_stat) if z0 is None else z0
    for t in range(1, T):
        z[t] = z[t - 1] + kappa * (theta - z[t - 1]) + sigma_z * rng.normal()
    return z


def run_policy(z, x, policy, lam):
    """policy(t, z_t, s_prev) -> a. Returns J and action list."""
    s = 0
    J = 0.0
    acts = []
    for t in range(len(z)):
        a = policy(t, z[t], s)
        J += a * x[t] - lam * (a != s)
        acts.append(a)
        s = a
    return J, acts
