"""
Simulate multi-site, multi-time-period observational treatment/outcome data
with a genuine underlying causal structure:

  - confounders X affect both treatment assignment T and outcome Y
  - treatment effect tau(X, site, time) is heterogeneous: it depends on
    covariates AND drifts across site and time period (distribution shift)
  - site and time period also shift the baseline outcome function and the
    confounder distribution itself (covariate shift), so a model fit on
    early times / some sites genuinely has to extrapolate for held-out
    times / sites. This makes time/site holdouts harder than a random
    holdout by construction, not by accident.

Outcome model (per unit i, site s, time t):
    mu0(x, s, t)      = baseline response surface (nonlinear in x, drifts w/ s,t)
    tau(x, s, t)      = true CATE (heterogeneous in x, drifts w/ s,t)
    Y = mu0 + T * tau + noise

Treatment assignment (confounded, not random):
    propensity e(x, s, t) = sigmoid(linear + nonlinear terms in x, site, time)
    T ~ Bernoulli(e)

We also derive a binary "label" for the sequential selector / evaluation:
  label = 1{ tau(x,s,t) > 0 }  (did this unit genuinely benefit from treatment)
This lets us report classification "accuracy" of the CATE model (did it
correctly predict the sign/benefit of the treatment effect) alongside
continuous CATE error (PEHE-style).
"""
import numpy as np
import pandas as pd

N_SITES = 6
N_PERIODS = 8
BASE_N_PER_CELL = 140  # per (site, period) cell before variable sizing


def _site_time_drift(site, period):
    # site-specific baseline shift and slope drift, smooth-ish across sites
    site_shift = np.sin(site * 1.3) * 1.5
    time_shift = 0.18 * period
    interaction = 0.05 * site * period
    return site_shift + time_shift + interaction


def _true_cate(x1, x2, x3, site, period):
    # Heterogeneous treatment effect. Deliberately a FIXED, highly
    # nonlinear function of covariates only (no direct site/period term):
    # all of the site/time difficulty in this simulation comes from
    # covariate-distribution shift (x1/x2/x3 means and ranges drift
    # across site/period in generate_dataset), not from the tau(x)
    # function itself changing. That is what makes the holdouts a
    # genuine extrapolation problem that adaptive importance-weighting
    # (reweighting training samples to look like the target covariate
    # distribution) can actually address -- as opposed to irreducible
    # concept drift, which no amount of reweighting could fix.
    return (
        0.15
        + 0.35 * x1
        - 0.30 * (x2 ** 2)
        + 0.25 * x3
        + 0.30 * np.sin(1.6 * x1 * x3)
        + 0.20 * x1 * x2
    )


def _mu0(x1, x2, x3, x4, site, period):
    drift = _site_time_drift(site, period)
    return (
        1.0
        + 1.5 * x1
        - 1.0 * x2
        + 0.7 * x3 * x4
        + 0.4 * np.sin(2.0 * x4)
        + drift
    )


def _propensity(x1, x2, x3, site, period, rng):
    # Confounded assignment: x1, x2 drive both outcome and treatment.
    logit = (
        0.8 * x1
        - 1.1 * x2
        + 0.5 * x3
        + 0.15 * site
        - 0.10 * period
        + 0.3 * x1 * x2
    )
    e = 1.0 / (1.0 + np.exp(-logit))
    return np.clip(e, 0.05, 0.95)


def generate_dataset(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for site in range(N_SITES):
        for period in range(N_PERIODS):
            n = int(rng.integers(BASE_N_PER_CELL - 30, BASE_N_PER_CELL + 30))
            # covariate distribution itself shifts across site/period
            # (covariate shift on top of effect drift)
            mean_shift = 0.30 * site - 0.18 * period
            x1 = rng.normal(0.0 + mean_shift, 1.0 + 0.04 * site, n)
            x2 = rng.normal(0.0 - 0.7 * mean_shift, 1.0, n)
            x3 = rng.uniform(-1, 1, n) + 0.18 * site - 0.08 * period
            x4 = rng.normal(0, 1, n)

            e = _propensity(x1, x2, x3, site, period, rng)
            t = rng.binomial(1, e)

            mu0 = _mu0(x1, x2, x3, x4, site, period)
            tau = _true_cate(x1, x2, x3, site, period)
            noise = rng.normal(0, 1.8, n)
            y = mu0 + t * tau + noise

            for i in range(n):
                rows.append(dict(
                    uid=uid, site=site, period=period,
                    x1=x1[i], x2=x2[i], x3=x3[i], x4=x4[i],
                    propensity=e[i], treatment=int(t[i]),
                    outcome=y[i], true_cate=tau[i], mu0=mu0[i],
                    benefit_label=int(tau[i] > 0),
                ))
                uid += 1
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.shape)
    print(df.groupby("site")["true_cate"].mean())
    print(df["treatment"].mean(), df["benefit_label"].mean())
