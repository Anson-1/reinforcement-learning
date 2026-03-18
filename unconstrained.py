"""
Backward Induction (Dynamic Programming) for Unconstrained Multi-Asset Portfolio

Solves the Bellman equation via backward induction on a discretized state space.
Allows short selling and borrowing with wealth-based leverage limits.
No neural networks — pure dynamic programming.

Bellman equation:
  V_T(W) = U(W) = (1 - exp(-A*W)) / A               (terminal)
  V_t(W, p) = max_a  E[ V_{t+1}(W', p') ]           (recursive)

State: (t, W, p_1, ..., p_n)  — time, wealth, risky asset proportions
       p_0 = 1 - sum(p_k) is the cash proportion (derived)
Action: (dp_1, ..., dp_n) — changes to risky proportions

Constraints:
  - Turnover: 0.5*(|dp_0| + sum|dp_k|) <= turnover_limit
  - Leverage: gross_exposure(p_new) <= 1 + leverage_factor * max(W, 0)
"""

import numpy as np
import time
from itertools import product
from scipy.interpolate import RegularGridInterpolator


class PortfolioDP:

    def __init__(
        self,
        n=3,                        # number of risky assets
        T=9,                        # time horizon (number of steps)
        r=0.07,                     # risk-free rate
        A=0.5,                      # risk aversion coefficient
        a_k=None,                   # mean returns (n,)
        s_k=None,                   # return variances (n,)
        p_init=None,                # initial portfolio [cash, a1, ..., an]
        wealth_points=15,           # number of wealth grid points
        wealth_min=0.05,
        wealth_max=3.5,
        prop_min=-0.5,              # min proportion per risky asset
        prop_max=1.5,               # max proportion per risky asset
        prop_step=0.25,             # proportion grid step size
        action_step=0.05,           # action grid step size
        action_max=0.10,            # max change per asset per step
        turnover_limit=0.10,        # max portfolio turnover per step
        leverage_factor=1.0,        # leverage scales with wealth
        integration='quadrature',   # 'quadrature' or 'monte_carlo'
        n_quad=5,                   # Gauss-Hermite nodes per dimension
        n_mc=200,                   # Monte Carlo samples (if MC mode)
    ):
        self.n = n
        self.T = T
        self.r = r
        self.A = A
        self.a_k = np.array(a_k if a_k is not None else [0.08, 0.06, 0.10][:n])
        self.s_k = np.array(s_k if s_k is not None else [0.02, 0.015, 0.04][:n])
        self.std_k = np.sqrt(self.s_k)
        self.leverage_factor = leverage_factor
        self.turnover_limit = turnover_limit
        self.integration = integration
        self.n_mc = n_mc

        # Initial portfolio
        if p_init is not None:
            self.p_init = np.array(p_init, dtype=float)
        else:
            w = 0.60 / n
            self.p_init = np.array([1.0 - n * w] + [w] * n)

        # Wealth grid
        self.wealth_grid = np.linspace(wealth_min, wealth_max, wealth_points)

        # Proportion grid per risky asset
        self.prop_grid = np.arange(prop_min, prop_max + prop_step * 0.5, prop_step)
        self.prop_grid = np.round(self.prop_grid, 6)
        n_pg = len(self.prop_grid)

        # Action space: all combos filtered by turnover
        avals = np.arange(-action_max, action_max + action_step * 0.5, action_step)
        avals = np.round(avals, 6)
        all_act = np.array(list(product(avals, repeat=n)))
        d_cash = -all_act.sum(axis=1)
        turnover = 0.5 * (np.abs(d_cash) + np.abs(all_act).sum(axis=1))
        self.actions = all_act[turnover <= turnover_limit + 1e-10]

        # Quadrature nodes and weights
        if integration == 'quadrature':
            self._setup_quadrature(n_quad)

        # All proportion index tuples and corresponding values
        self.prop_indices = list(product(range(n_pg), repeat=n))
        self.prop_values = np.array(
            [[self.prop_grid[i] for i in idx] for idx in self.prop_indices]
        )

        # Storage for value function and policy at each time step
        self.grid_shape = (len(self.wealth_grid),) + (n_pg,) * n
        self.V = {}
        self.policy = {}

        # Terminal condition: V_T(W, p) = U(W) for all p
        self.V[T] = np.zeros(self.grid_shape)
        for wi, W in enumerate(self.wealth_grid):
            self.V[T][wi] = self._utility(W)

        print("PortfolioDP initialized:")
        print(f"  {n} assets, T={T}, r={r}, A={A}")
        print(f"  Mean returns:   {list(self.a_k)}")
        print(f"  Variances:      {list(self.s_k)}")
        print(f"  Excess returns: {list(np.round(self.a_k - r, 4))}")
        print(f"  Wealth grid:    {len(self.wealth_grid)} pts in [{wealth_min}, {wealth_max}]")
        print(f"  Prop grid:      {n_pg} pts in [{self.prop_grid[0]}, {self.prop_grid[-1]}]"
              f", step={prop_step}")
        print(f"  State combos:   {len(self.wealth_grid) * len(self.prop_indices):,}"
              f" per time step")
        print(f"  Actions:        {len(self.actions)} (turnover <= {turnover_limit})")
        if integration == 'quadrature':
            print(f"  Quadrature:     {n_quad}^{n} = {n_quad**n} nodes")
        else:
            print(f"  Monte Carlo:    {n_mc} samples")

    def _utility(self, W):
        """Exponential utility: U(W) = (1 - exp(-A*W)) / A"""
        return (1.0 - np.exp(-self.A * np.clip(W, -20, 100))) / self.A

    def _setup_quadrature(self, n_quad):
        """Gauss-Hermite quadrature for n-dim return integration.

        Transform: R_k = a_k + sqrt(2*s_k)*x, weight = w/sqrt(pi)
        """
        x, w = np.polynomial.hermite.hermgauss(n_quad)
        pts_1d = [self.a_k[k] + np.sqrt(2.0 * self.s_k[k]) * x
                  for k in range(self.n)]
        wts_1d = [w / np.sqrt(np.pi)] * self.n

        # n-dimensional tensor product
        grids = np.meshgrid(*pts_1d, indexing='ij')
        self.quad_returns = np.stack([g.ravel() for g in grids], axis=1)  # (Q, n)

        wgrids = np.meshgrid(*wts_1d, indexing='ij')
        self.quad_weights = np.prod(
            np.stack([g.ravel() for g in wgrids], axis=1), axis=1
        )  # (Q,)

    def _build_interpolator(self, V_arr):
        """Build RegularGridInterpolator for a value function array."""
        axes = (self.wealth_grid,) + (self.prop_grid,) * self.n
        return RegularGridInterpolator(
            axes, V_arr, method='linear',
            bounds_error=False, fill_value=None,
        )

    def _compute_expected_value(self, W, new_p_batch, interp):
        """Compute E[V_{t+1}(W', p')] for a batch of candidate allocations.

        Args:
            W: current wealth (scalar)
            new_p_batch: (N, n) proposed risky-asset proportions
            interp: interpolator for V_{t+1}

        Returns:
            (N,) expected values
        """
        N = new_p_batch.shape[0]
        new_cash = 1.0 - new_p_batch.sum(axis=1)  # (N,)

        if self.integration == 'quadrature':
            R = self.quad_returns     # (Q, n)
            wts = self.quad_weights   # (Q,)
        else:
            R = np.random.normal(self.a_k, self.std_k, (self.n_mc, self.n))
            wts = np.full(self.n_mc, 1.0 / self.n_mc)

        Q = R.shape[0]

        # Portfolio return for each (allocation, scenario): (N, Q)
        port_ret = new_cash[:, None] * self.r + new_p_batch @ R.T

        # Next-period wealth: (N, Q)
        W_next = W * (1.0 + port_ret)

        # Next-period proportions: (N, Q, n)
        denom = 1.0 + port_ret
        p_next = (new_p_batch[:, None, :] * (1.0 + R[None, :, :])
                  ) / denom[:, :, None]

        # Clip to interpolation grid bounds
        W_clip = np.clip(W_next, self.wealth_grid[0], self.wealth_grid[-1])
        p_clip = np.clip(p_next, self.prop_grid[0], self.prop_grid[-1])

        # Assemble interpolation points: (N*Q, 1+n)
        NQ = N * Q
        pts = np.empty((NQ, 1 + self.n))
        pts[:, 0] = W_clip.ravel()
        for k in range(self.n):
            pts[:, 1 + k] = p_clip[:, :, k].ravel()

        # Interpolate V_{t+1}
        vals = interp(pts)  # (N*Q,)

        # Bankrupt states get penalty utility
        bankrupt = W_next.ravel() <= 0
        if np.any(bankrupt):
            vals[bankrupt] = self._utility(0.001)

        # Weighted average over scenarios: (N,)
        return vals.reshape(N, Q) @ wts

    def solve(self):
        """Backward induction: compute V_t and optimal policy for t=T-1,...,0."""
        print(f"\n{'='*60}")
        print("Backward Induction")
        print(f"{'='*60}")
        t_total = time.time()

        for t in range(self.T - 1, -1, -1):
            ts = time.time()
            interp = self._build_interpolator(self.V[t + 1])

            V_t = np.full(self.grid_shape, -1e20)
            pol_t = np.zeros(self.grid_shape + (self.n,))
            n_computed = 0

            for wi, W in enumerate(self.wealth_grid):
                max_lev = 1.0 + self.leverage_factor * max(W, 0.0)

                for pi, pidx in enumerate(self.prop_indices):
                    p_risky = self.prop_values[pi]  # (n,)

                    # Skip states with wildly excessive gross exposure
                    p_cash = 1.0 - p_risky.sum()
                    if np.abs(p_cash) + np.abs(p_risky).sum() > max_lev + 1.0:
                        V_t[(wi,) + pidx] = self._utility(0.01)
                        continue

                    # Candidate allocations: current + each action
                    new_p = p_risky + self.actions  # (N_act, n)
                    new_cash = 1.0 - new_p.sum(axis=1)
                    gross = np.abs(new_cash) + np.abs(new_p).sum(axis=1)
                    feasible = gross <= max_lev + 1e-10
                    fidx = np.where(feasible)[0]

                    if len(fidx) == 0:
                        V_t[(wi,) + pidx] = self._utility(0.01)
                        continue

                    # Expected values for all feasible actions (vectorized)
                    evs = self._compute_expected_value(W, new_p[fidx], interp)

                    best = np.argmax(evs)
                    V_t[(wi,) + pidx] = evs[best]
                    pol_t[(wi,) + pidx] = self.actions[fidx[best]]
                    n_computed += 1

            self.V[t] = V_t
            self.policy[t] = pol_t
            print(f"  t={t}: {n_computed:,} states optimized, {time.time() - ts:.1f}s")

        print(f"\nTotal solve time: {time.time() - t_total:.1f}s")

    def _lookup_action(self, t, W, p_risky):
        """Find optimal action via nearest-neighbor grid lookup."""
        wi = int(np.argmin(np.abs(self.wealth_grid - W)))
        pidx = tuple(
            int(np.argmin(np.abs(self.prop_grid - p_risky[k])))
            for k in range(self.n)
        )
        return self.policy[t][(wi,) + pidx].copy()

    def simulate(self, num_episodes=2000, verbose_episode=False):
        """Forward-simulate the optimal policy and report results."""
        print(f"\n{'='*60}")
        print(f"Simulating {num_episodes} episodes")
        print(f"{'='*60}")

        np.random.seed(42)
        utilities = []
        terminal_wealths = []
        short_steps = np.zeros(self.n, dtype=int)

        for ep in range(num_episodes):
            W = 1.0
            p = self.p_init.copy()  # [cash, a1, ..., an]
            traj = []

            for t in range(self.T):
                p_risky = p[1:].copy()
                action = self._lookup_action(t, W, p_risky)

                # Apply action
                new_pr = p_risky + action
                new_cash = 1.0 - new_pr.sum()

                # Enforce leverage constraint
                max_lev = 1.0 + self.leverage_factor * max(W, 0.0)
                gross = np.abs(new_cash) + np.abs(new_pr).sum()
                if gross > max_lev + 1e-10:
                    scale = max_lev / gross
                    new_pr = p_risky + action * scale
                    new_cash = 1.0 - new_pr.sum()

                # Track short selling
                for k in range(self.n):
                    if new_pr[k] < -1e-10:
                        short_steps[k] += 1

                if verbose_episode and ep == 0:
                    traj.append((t, W, p.copy(), action.copy(),
                                 np.concatenate([[new_cash], new_pr])))

                # Market step
                R = np.random.normal(self.a_k, self.std_k)
                port_ret = new_cash * self.r + (new_pr * R).sum()
                W_new = W * (1.0 + port_ret)

                if W_new <= 0.001:
                    W = 0.001
                    p = np.concatenate([[1.0], np.zeros(self.n)])
                else:
                    denom = 1.0 + port_ret
                    p = np.concatenate([
                        [new_cash * (1.0 + self.r) / denom],
                        new_pr * (1.0 + R) / denom,
                    ])
                    W = W_new

            utilities.append(self._utility(W))
            terminal_wealths.append(W)

            if verbose_episode and ep == 0:
                print("\n--- Sample Episode ---")
                for ts, Ws, pb, act, pa in traj:
                    names = [f"a{k+1}" for k in range(self.n)]
                    before = (f"cash={pb[0]:.3f}, "
                              + ", ".join(f"{names[k]}={pb[k+1]:.3f}"
                                          for k in range(self.n)))
                    deltas = ", ".join(f"d{names[k]}={act[k]:+.3f}"
                                       for k in range(self.n))
                    after = (f"cash={pa[0]:.3f}, "
                             + ", ".join(f"{names[k]}={pa[k+1]:.3f}"
                                         for k in range(self.n)))
                    flag = " [SHORT]" if any(pa[1:] < -1e-10) else ""
                    print(f"  t={ts}: W={Ws:.4f}")
                    print(f"    before: [{before}]")
                    print(f"    action: [{deltas}]")
                    print(f"    after:  [{after}]{flag}")
                print(f"  Terminal: W={terminal_wealths[0]:.4f}, "
                      f"U(W)={utilities[0]:.4f}")

        avg_u = np.mean(utilities)
        std_err = np.std(utilities) / np.sqrt(num_episodes)
        avg_w = np.mean(terminal_wealths)
        med_w = np.median(terminal_wealths)
        total_steps = num_episodes * self.T

        print(f"\nResults ({num_episodes} episodes):")
        print(f"  Average utility:         {avg_u:.6f} +/- {std_err:.6f}")
        print(f"  Average terminal wealth: {avg_w:.4f}")
        print(f"  Median terminal wealth:  {med_w:.4f}")
        print(f"\nShort-selling frequency:")
        for k in range(self.n):
            pct = 100.0 * short_steps[k] / total_steps
            excess = self.a_k[k] - self.r
            print(f"  Asset {k+1} (mu={self.a_k[k]:.2f}, excess={excess:+.4f}): "
                  f"{short_steps[k]:,}/{total_steps:,} steps ({pct:.1f}%)")

        return avg_u


if __name__ == '__main__':
    print("=" * 60)
    print("Unconstrained Portfolio - Backward Induction (DP)")
    print("=" * 60)

    dp = PortfolioDP(
        n=3, T=9, r=0.07, A=0.5,
        a_k=[0.08, 0.06, 0.10],
        s_k=[0.02, 0.015, 0.04],
        p_init=[0.40, 0.20, 0.20, 0.20],
    )

    dp.solve()
    avg_utility = dp.simulate(num_episodes=2000, verbose_episode=True)

    print(f"\n{'='*60}")
    print("Compare with constrained SB3 version to verify improvement.")
    print("Unconstrained (shorts + leverage) should yield >= constrained utility.")
    print(f"{'='*60}")
