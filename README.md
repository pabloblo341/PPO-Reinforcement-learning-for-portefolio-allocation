# 📈 PPO-Based Dynamic Portfolio Allocation

A Reinforcement Learning approach to portfolio management using **Proximal Policy Optimization (PPO)**. Instead of the classical estimate-then-optimize pipeline (Markowitz), the agent directly learns an allocation policy end-to-end — including transaction costs.

---

## Motivation

The traditional two-step approach to portfolio construction (estimate returns/covariances → optimize) suffers from three key issues:

- Estimation errors get magnified during optimization
- The objective (best statistical model) is decoupled from the real goal (best future decision)
- High-dimensional covariance estimation is statistically fragile

Reinforcement Learning bypasses these issues: the agent directly learns to allocate assets by interacting with the market environment, optimizing for long-term portfolio growth.

---

## Problem Formulation

Daily rebalancing across **3 assets** (Equities, Bonds, Gold) from January 1999 to September 2025.

**State**: rolling window of past returns + current portfolio weights

$$s_t = [r_{t-L:t-1},\ w_{t-1}]$$

**Action**: portfolio weights on the simplex

$$w_t \in \Delta^3 \quad \text{with} \quad \sum_{i=1}^{3} w_i = 1, \quad w_i \ge 0$$

**Portfolio value dynamics**:

$$V_{t+1} = V_t \cdot \left(1 + w_t^\top r_t - \text{TC}_t\right)$$

**Transaction costs** (one-way):
- Equities: 5 bps
- Bonds: 2 bps  
- Gold: 10 bps

**Reward function** (regularized):

$$r_t = \log\left(\frac{V_{t+1}}{V_t}\right) - \lambda_{\text{var}} \cdot (w_t^\top r_t)^2 - \lambda_{\text{herf}} \cdot \sum_{i=1}^{3} w_{t,i}^2$$

The variance and Herfindahl penalties prevent degenerate concentration strategies.

---

## Why PPO

1. **Stable policy updates** — clipped objective prevents destructive weight jumps, critical for portfolio allocation
2. **Continuous action space** — portfolio weights live on a simplex; PPO natively handles continuous policies via softmax output
3. **Robustness to noisy rewards** — financial returns are non-stationary; PPO's GAE advantage estimation handles this well

---

## Architecture

**Actor-Critic with shared backbone:**
- Input: rolling window of returns + current weights
- Shared MLP layers
- Actor head: outputs logits → softmax → portfolio weights
- Critic head: scalar value estimate

**Training objective:**

$$L(\theta) = \mathbb{E}_t\left[L^{\text{CLIP}}_t(\theta) - c_1 L^{\text{VF}}_t(\theta) + c_2 S[\pi_\theta](s_t)\right]$$

**Key hyperparameters:**
| Parameter | Value |
|---|---|
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| ε (clip) | 0.2 |
| Value loss coeff c1 | 0.5 |

---

## Benchmark

PPO is compared against **Markowitz mean-variance optimization** (efficient frontier) as a classical baseline, with transaction costs applied to both strategies.

---

## Stack

| Component | Technology |
|---|---|
| RL framework | PyTorch (custom PPO implementation) |
| Optimization | CVXPY (Markowitz baseline) |
| Data | Excel (daily returns 1999–2025) |
| Analysis | pandas, numpy, matplotlib |

---

## Data

Daily returns for 3 assets (Jan 1999 – Sep 2025):
- **Equities**: MSCI World Index (MXWD)
- **Gold**: XAU Curncy
- **Fixed Income**: Bloomberg Global Aggregate TR Index
