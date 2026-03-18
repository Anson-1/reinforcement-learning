# Test Cases for Backward Induction DP

## Verification Strategy

Since there's no closed-form solution for the constrained multi-asset problem,
we verify correctness by checking **financial properties that must hold**.
If the DP violates any of these, there's a bug.

---

## A. Value Function Tests (n=1)

### Case A1: V(W) properties at each time step

**Parameters:** n=1, T=5, r=0.07, A=0.5, μ=0.08, σ²=0.02

**Plot:** V_t(W) for t=0,1,2,...,T on the same axes (fix proportion at mid-grid)

**What must hold:**
- **Increasing in W**: more wealth → more utility, at every t
- **Concave in W**: diminishing marginal value (CARA utility is concave, DP preserves this)
- **V_0 ≥ V_1 ≥ ... ≥ V_T**: more time remaining → more opportunity to invest → higher value
- **V_T(W) = U(W)**: terminal condition is just the utility function

**Why this matters:** If any curve crosses another or bends upward, the backward induction has a bug.

---

### Case A2: Risk aversion comparison

**Parameters:** n=1, T=5, r=0.07, μ=0.08, σ²=0.02, **A=0.5 vs A=2.0 vs A=5.0**

**Plot:**
1. Terminal wealth histograms (overlaid, 3 colors)
2. Average risky proportion over time for each A

**What must hold:**
- Higher A → **tighter** terminal wealth distribution (less variance)
- Higher A → **lower** average risky allocation at every time step
- Higher A → average terminal wealth closer to (1+r)^T (approaching all-cash)
- All three should have avg utility > cash utility (DP should always beat cash)

**Why this matters:** Risk aversion is the core parameter. If A=5.0 is more aggressive than A=0.5, something is fundamentally wrong.

---

## B. Multi-Asset Tests

### Case B1: Good asset vs bad asset (n=2)

**Parameters:** n=2, T=3, r=0.07, A=0.5
- Asset 1: μ=0.10, σ²=0.02 (excess = +0.03, good)
- Asset 2: μ=0.04, σ²=0.02 (excess = -0.03, bad)

**Plot:** Average allocation to each asset over time (bar chart per time step)

**What must hold:**
- Asset 1 (good): **positive** average allocation at all times
- Asset 2 (bad): **negative** average allocation (shorted) at all times
- |allocation to asset 1| ≈ |allocation to asset 2| (symmetric excess returns, same variance)
- Cash allocation adjusts to keep leverage reasonable

**Why this matters:** An optimizer that can't distinguish good from bad assets is broken.

---

### Case B2: Three assets ranked by excess return (n=3)

**Parameters:** n=3, T=3, r=0.07, A=0.5
- Asset 1: μ=0.08, σ²=0.02 (excess = +0.01)
- Asset 2: μ=0.06, σ²=0.015 (excess = -0.01)
- Asset 3: μ=0.10, σ²=0.04 (excess = +0.03, but high variance)

**Plot:** Average final allocation (bar chart: cash, a1, a2, a3)

**What must hold:**
- Asset 2 (negative excess): shorted
- Asset 3 (highest excess): largest positive allocation
- Asset 1 (small positive excess): moderate positive allocation
- Ordering by allocation size: a3 > a1 > 0 > a2

**Why this matters:** Tests that the DP correctly weighs return vs risk across multiple assets.

---

### Case B3: All assets below risk-free rate (n=2)

**Parameters:** n=2, T=3, r=0.07, A=0.5
- Asset 1: μ=0.05, σ²=0.02 (excess = -0.02)
- Asset 2: μ=0.03, σ²=0.015 (excess = -0.04)

**Plot:** Average allocation over time + terminal wealth histogram

**What must hold:**
- Both risky assets get **negative** average allocation (shorted)
- Cash proportion > 1 (using short proceeds to hold more cash)
- Terminal wealth distribution should be **tighter** than Case B1 (less risk-taking)
- Average utility should still beat pure cash (shorting bad assets is profitable)

**Why this matters:** Edge case — verifies the DP correctly exploits shorting when all risky assets are inferior.

---

## C. Scaling T Tests (n=1)

### Case C1: T = 1, 3, 5, 9

**Parameters:** n=1, r=0.07, A=0.5, μ=0.08, σ²=0.02

**Plot:**
1. Average utility vs T (line plot with error bars)
2. Terminal wealth distributions (overlaid histograms for T=1,3,5,9)
3. Average risky proportion at t=0 for each T

**What must hold:**
- Average utility **increases** with T (more periods → more compounding of good strategy)
- Terminal wealth distribution **widens** with T (more periods → more variance)
- Average terminal wealth increases with T
- Risky proportion at t=0 may **decrease** with T (the (1+r)^(T-t-1) factor makes early allocation more conservative)
- DP should beat cash at every T

**Why this matters:** Demonstrates the code works across the full range T=1 to T=9 as required by the problem.

---

### Case C2: T = 9 full run (n=1 and n=3)

**Parameters:**
- Run 1: n=1, T=9, r=0.07, A=0.5, μ=0.08, σ²=0.02
- Run 2: n=3, T=9, default params (coarse grid for speed)

**Plot:** Wealth path fan chart (mean ± 1σ, 2σ bands across time)

**What must hold:**
- Wealth paths trend upward on average
- Fan widens over time (increasing uncertainty)
- n=3 should have higher average utility than n=1 (more assets → better diversification)

**Why this matters:** The "showcase" case — proves the code handles the maximum required complexity.

---

## Summary: Minimum Tests for Correctness

| Test | What it proves | Expected runtime |
|------|---------------|-----------------|
| A1: Value function curves | DP backward induction is internally consistent | ~2s |
| A2: Risk aversion comparison | Core parameter works correctly | ~5s |
| B1: Good vs bad asset | Optimizer distinguishes return quality | ~10s |
| B2: Three asset ranking | Multi-asset allocation ordering correct | ~30s |
| B3: All below risk-free | Shorting behavior correct | ~10s |
| C1: Scaling T=1,3,5,9 | Works across time horizons | ~15s |
| C2: T=9 showcase | Full complexity demonstration | ~60s |

Total: ~2 minutes. If all plots look correct, we have strong evidence the DP works.
