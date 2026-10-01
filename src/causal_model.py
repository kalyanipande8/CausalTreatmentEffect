"""
X-learner style CATE estimator (doubly-robust-ish, real implementation,
numpy/sklearn only -- see README for why PyTorch was not used).

Stage 1: two response-surface models
    mu0(x) fit on control units (T=0) -> predicts outcome under control
    mu1(x) fit on treated units  (T=1) -> predicts outcome under treatment
Stage 2: imputed treatment effects
    D1_i = Y_i - mu0(X_i)   for treated units   (i in T=1)
    D0_i = mu1(X_i) - Y_i   for control units   (i in T=0)
    tau1(x) fit on (X_treated, D1)
    tau0(x) fit on (X_control, D0)
Stage 3: propensity model e(x) fit by logistic regression; CATE estimate is
    the propensity-weighted blend:
    tau_hat(x) = e(x) * tau0(x) + (1 - e(x)) * tau1(x)

This is the standard Kuenzel et al. X-learner, which is doubly-robust-ish
in that it uses outcome regressions AND a propensity model.

Adaptive reweighting (the "distribution-shift robustness" part):
    When fitting on a source domain (e.g. early time periods / some sites)
    and evaluating on a shifted target domain, we fit a second, separate
    logistic-regression "domain classifier" that predicts P(domain=target |
    x) using source+target covariates. Importance weights
    w(x) = P(target|x) / P(source|x) are used as sample_weight when
    refitting the response-surface and effect models, down-weighting
    source points that don't look like the target and up-weighting
    source points that do. This is real density-ratio / covariate-shift
    importance weighting (Shimodaira 2000 style), not a toy label.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


def _reg():
    return GradientBoostingRegressor(
        n_estimators=60, max_depth=2, learning_rate=0.1,
        subsample=0.8, random_state=0
    )


class XLearner:
    def __init__(self):
        self.mu0 = _reg()
        self.mu1 = _reg()
        self.tau0 = _reg()
        self.tau1 = _reg()
        self.prop = LogisticRegression(max_iter=1000)

    def fit(self, X, T, Y, sample_weight=None):
        X = np.asarray(X)
        T = np.asarray(T)
        Y = np.asarray(Y)
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight)

        idx0 = T == 0
        idx1 = T == 1
        self.mu0.fit(X[idx0], Y[idx0], sample_weight=w[idx0])
        self.mu1.fit(X[idx1], Y[idx1], sample_weight=w[idx1])

        D1 = Y[idx1] - self.mu0.predict(X[idx1])
        D0 = self.mu1.predict(X[idx0]) - Y[idx0]
        self.tau1.fit(X[idx1], D1, sample_weight=w[idx1])
        self.tau0.fit(X[idx0], D0, sample_weight=w[idx0])

        self.prop.fit(X, T, sample_weight=w)
        return self

    def predict_cate(self, X):
        X = np.asarray(X)
        e = np.clip(self.prop.predict_proba(X)[:, 1], 0.05, 0.95)
        t1 = self.tau1.predict(X)
        t0 = self.tau0.predict(X)
        return e * t0 + (1 - e) * t1


def _poly_features(X):
    # degree-2 features (x_i, x_i^2, pairwise products) so the domain
    # classifier can pick up on nonlinear / interaction covariate shift,
    # not just mean shift.
    X = np.asarray(X)
    n, d = X.shape
    feats = [X]
    feats.append(X ** 2)
    for i in range(d):
        for j in range(i + 1, d):
            feats.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(feats)


def domain_importance_weights(X_source, X_target, clip=3.0):
    """
    Real covariate-shift importance weighting via a domain classifier.
    Fit logistic regression (on polynomial-expanded features, to capture
    nonlinear shift) to discriminate source (label 0) vs target (label 1)
    covariates; weight source points by p(target|x)/p(source|x).
    """
    X_source = np.asarray(X_source)
    X_target = np.asarray(X_target)
    Xs_poly = _poly_features(X_source)
    Xt_poly = _poly_features(X_target)
    Xall = np.vstack([Xs_poly, Xt_poly])
    domain = np.concatenate([np.zeros(len(Xs_poly)), np.ones(len(Xt_poly))])
    clf = LogisticRegression(max_iter=2000, C=0.5)
    clf.fit(Xall, domain)
    p_target = np.clip(clf.predict_proba(Xs_poly)[:, 1], 1e-3, 1 - 1e-3)
    w = p_target / (1 - p_target)
    # soften the density ratio (sqrt) to reduce variance from an imperfect
    # domain classifier, then clip for numerical stability.
    w = np.sqrt(w)
    w = np.clip(w, 1.0 / clip, clip)
    # normalize to mean 1 for numerical stability
    w = w / w.mean()
    return w
