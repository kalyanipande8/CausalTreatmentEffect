"""
Evaluation harness: random vs. time-holdout vs. site-holdout splits, with
and without adaptive (domain-shift) reweighting.

"Accuracy" here = benefit-classification accuracy: does the model correctly
predict the SIGN of the true CATE (treated unit genuinely benefits, tau>0,
vs not) on held-out units. This is a genuine, non-trivial binary target
tied directly to the causal estimand, not a proxy label.
"""
import numpy as np
from .causal_model import XLearner, domain_importance_weights

FEATURES = ["x1", "x2", "x3", "x4"]


def make_splits(df):
    """
    random: iid random holdout (should be "too easy" -- in-distribution)
    time: train on early periods, test on latest periods (held out by time)
    site: train on most sites, test on a few held-out sites (held out by site)
    """
    rng = np.random.default_rng(42)
    n = len(df)
    perm = rng.permutation(n)
    cut = int(n * 0.75)
    random_train_idx = perm[:cut]
    random_test_idx = perm[cut:]

    max_period = df["period"].max()
    time_test_mask = df["period"] >= (max_period - 1)  # last 2 periods held out
    time_train_idx = df.index[~time_test_mask].to_numpy()
    time_test_idx = df.index[time_test_mask].to_numpy()

    held_out_sites = [2, 5]
    site_test_mask = df["site"].isin(held_out_sites)
    site_train_idx = df.index[~site_test_mask].to_numpy()
    site_test_idx = df.index[site_test_mask].to_numpy()

    return dict(
        random=(random_train_idx, random_test_idx),
        time=(time_train_idx, time_test_idx),
        site=(site_train_idx, site_test_idx),
    )


def _accuracy_pehe(model, X_test, true_cate_test):
    pred = model.predict_cate(X_test)
    acc = float(np.mean((pred > 0) == (true_cate_test > 0)))
    pehe = float(np.sqrt(np.mean((pred - true_cate_test) ** 2)))
    return acc, pehe


def evaluate_split(df, train_idx, test_idx, reweight=False):
    train = df.loc[train_idx]
    test = df.loc[test_idx]
    X_train = train[FEATURES].to_numpy()
    T_train = train["treatment"].to_numpy()
    Y_train = train["outcome"].to_numpy()
    X_test = test[FEATURES].to_numpy()
    true_cate_test = test["true_cate"].to_numpy()

    sample_weight = None
    if reweight:
        sample_weight = domain_importance_weights(X_train, X_test)

    model = XLearner().fit(X_train, T_train, Y_train, sample_weight=sample_weight)
    acc, pehe = _accuracy_pehe(model, X_test, true_cate_test)
    return dict(accuracy=acc, pehe=pehe, n_train=len(train), n_test=len(test))


def run_full_evaluation(df):
    splits = make_splits(df)
    results = {}
    for name, (train_idx, test_idx) in splits.items():
        no_rw = evaluate_split(df, train_idx, test_idx, reweight=False)
        rw = evaluate_split(df, train_idx, test_idx, reweight=True)
        results[name] = dict(no_reweight=no_rw, reweight=rw)
    return results
