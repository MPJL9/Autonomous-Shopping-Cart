# Policy Evaluation

How we compare BC policies on the cart in a fair, reproducible way.

## Why a metric

Validation MSE on a held-out fold tells us how well the model fits the
demonstration distribution, not how well it *follows the operator on
hardware*. Two models with similar val MSE can still feel completely
different to drive — one commits, the other twitches; one corrects
heading, the other ignores it. We need a number computed from
**closed-loop deployment** that captures the closed-loop properties we
actually care about.

## What we measure

Three things, weighted into one scalar:

1. **Distance error outside the desired follow band.**
   The cart is supposed to maintain operator distance inside
   $[1.26, 1.56]\,\mathrm{m}$ (the empirical inter-quartile range of
   natural follow distance). Inside the band: zero penalty. Outside:
   quadratic in how far outside.

2. **Heading error.** Quadratic in $\theta$, the operator bearing in
   radians. We want the cart facing the operator.

3. **Action jerk.** The L2 norm of consecutive action differences,
   $\|a_t - a_{t-1}\|_2$. Smooth control is preferred over twitchy
   control even if the average position is the same.

Combined as the per-step cost
$$
c_t \;=\; w_d \,\bigl(\Delta d_t^{\text{out}}\bigr)^{2}
       \;+\; w_\theta \, \theta_t^{2}
       \;+\; w_j \, \|a_t - a_{t-1}\|_2^{\,2}
$$
with
$$
\Delta d_t^{\text{out}}
   \;=\;
   \max\!\bigl(\,0,\; d_{\text{lo}} - d_t,\; d_t - d_{\text{hi}}\,\bigr).
$$
The metric reported per session is the time-average,
$$
J \;=\; \frac{1}{N} \sum_{t=1}^{N} c_t
$$
so different recording lengths (15 s vs 1 min) are directly comparable.
Lower $J$ is better; $J = 0$ is the unattainable ideal of
perfectly-in-band, perfectly-on-axis, perfectly-smooth behavior.

## Default weights

| Symbol | Meaning | Default | Rationale |
|---|---|---|---|
| $w_d$ | distance penalty weight | $1.0\;\mathrm{m}^{-2}$ | reference unit |
| $w_\theta$ | heading penalty weight | $0.5\;\mathrm{rad}^{-2}$ | makes a $0.7\,\mathrm{m}$ out-of-band distance error and a $30^\circ$ heading error contribute roughly equally |
| $w_j$ | jerk penalty weight | $0.1$ | small enough that smooth wins ties but doesn't dominate the cost |
| $[d_{\text{lo}}, d_{\text{hi}}]$ | follow band | $[1.01, 1.32]\,\mathrm{m}$ | combined training Q25--Q75 (4/29 + 5/1); see "Band derivation" below |

If we want to argue different priorities (e.g. "heading matters more"),
re-run with different weights — the metric is parametric, not opinionated.

### Band derivation

The follow band $[d_{\text{lo}}, d_{\text{hi}}]$ should match the
operator's natural follow distance during the demonstrations the policy
was trained on. We derive it as the inter-quartile range (Q25--Q75) of
ArUco-distance values in the combined training set:

| dataset                           | N      | Q25  | Q50  | Q75  |
|-----------------------------------|--------|------|------|------|
| 4/29 only                         | 6{,}325 | 1.06 | 1.17 | 1.28 |
| 5/1 only                          | 2{,}557 | 0.94 | 1.05 | 1.55 |
| **combined 4/29 + 5/1 (current)** | 8{,}882 | **1.01** | **1.16** | **1.32** |

The 5/1 distribution is bimodal (operator did deliberate forward/back
sweeps) which widens its IQR; combining with 4/29 reins this in. We use
the combined band $[1.01, 1.32]\,\mathrm{m}$ as the canonical evaluation
band, which also matches the implicit setpoint the BC policy was
supervised toward (median $\approx 1.16\,\mathrm{m}$).

When the band is changed, the **runtime deadband should match it** so
the agent and the metric are consistent --- both use the same env vars:
\texttt{CART\_BC\_BAND\_MIN\_M} and \texttt{CART\_BC\_BAND\_MAX\_M}. If
they're inconsistent the cart parks where the runtime deadband says is
fine while the metric penalises it as out-of-band.

## Reporting

Don't report a single number. Always report the breakdown so a reader can
see *which* dimension a policy is winning or losing on:

| Policy | $J$ | time-in-band % | RMS distance err (m) | RMS heading err (deg) | RMS jerk |
|---|---|---|---|---|---|
| bc_2d_mirror | … | … | … | … | … |
| ... | | | | | |

Auxiliary numbers we'll also log:

- **time-in-band %**: fraction of steps where $d \in [1.26, 1.56]\,\mathrm{m}$.
- **track-loss %**: fraction of steps where the tracker reports `locked=False`. High track-loss invalidates a comparison; bring it up before reading $J$.
- **action-saturation %**: fraction of steps where $|L|$ or $|R|$ is at $1.0$ (the $\tanh$ ceiling). Hits at the policy committing maximally.

## Eval scenario (15-second standardized routine)

Identical script every time, so all recordings are directly comparable:

| t (s) | what the operator does |
|---|---|
| 0–5 | Stand still at $\sim 1.4\,\mathrm{m}$, centered on the cart's camera. Tests deadband stability. |
| 5–10 | Walk straight back to $\sim 2.0\,\mathrm{m}$ (cart should drive forward), then forward to $\sim 1.0\,\mathrm{m}$ (cart should reverse). Tests forward/reverse tracking. |
| 10–15 | Side-step $\sim 30^\circ$ to the cart's left, hold for 1 s, side-step to the right, hold. Tests heading correction. |

The 1-minute version is the same routine repeated 4×.

## How to run an eval session

1. **Start a recording** in the dashboard with `policy = bc_2d_mirror`
   (or whichever variant), `step_hz = 5`, `max_steps = 75` for 15 s
   ($75 \times 0.2\,\mathrm{s}$) or `300` for 1 min.
2. **Run the routine** above. Click `Stop` if you finish early; the
   metric auto-normalizes by step count.
3. **Download the session bundle** and drop it under
   `RL data/<date>/eval/<policy_name>/`.
4. **Process** the bundle so the npz has offline-truth ArUco obs:
   ```bash
   python3 training/process_session.py --keep-lost \
       "RL data/<date>/eval/<policy_name>"
   ```
5. **Score** the npz:
   ```bash
   python3 training/evaluate_policy.py \
       training/data/processed/<date>/<eval_session>.npz
   ```
   Output is a one-row metric table; multiple npzs give one row each.

## What "better" means

A new policy is reported as an improvement only when:

- $J$ decreases by **at least 10%**, AND
- track-loss does not significantly increase (no winning by hiding from the tracker), AND
- the change in components matches a story we expected to improve
  (e.g., if the new training added side-step demonstrations, heading-RMS
  should drop; if jerk drops while distance-err is unchanged, that's the
  block-cap-on-active-runs paying off).

A policy that has lower $J$ but higher track-loss is suspicious. So is
one that has very low jerk but is sitting at zero output (it's not
moving, it's just steady-state at the wrong distance). Sanity-check the
breakdown.

## Honest caveats

1. **Operator behavior is not perfectly reproducible.** Two recordings
   of the same routine will have minor differences in walking speed and
   sidestep angle. To smooth this out, run **3 recordings per policy**
   and report mean ± std for each component. The 15 s × 3 = 45 s of
   total recording per policy keeps the eval cycle short.
2. **Deadband stability is in the metric.** The runtime deadband already
   zeros the forward command inside the band, so a policy can score well
   on the in-band segment without having to do anything except output
   small commands. The jerk and heading terms keep this honest.
3. **Heading metric ignores lost-lock frames.** When the tracker drops
   the marker (or YOLO doesn't see the operator), the metric does not
   evaluate that step. Track-loss % is reported separately so a
   high-loss policy can't escape evaluation.
4. **The metric is intentionally imitation-loss-agnostic.** It does not
   look at the policy's MSE against demonstrations. The point is to
   measure deployment behavior, not training fit.
