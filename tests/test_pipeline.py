"""
Pipeline sanity tests. These are real checks run against the actual
generator / model / evaluation code (no mocks), covering the claims made
in the README:
  - time and site holdouts are genuinely harder than a random holdout
  - adaptive reweighting improves (or at least does not meaningfully hurt)
    generalization relative to no reweighting, on average across the
    distribution-shift splits
  - overall accuracy lands in a reasonable band (not trivially 50%, not
    suspiciously ~100%)
  - the active-learning loop actually reduces CATE error (PEHE) as more
    data is acquired
  - an injected sanity check: a model trained on pure noise (shuffled
    treatment) should NOT beat chance, confirming the pipeline doesn't
    silently leak labels
"""
import numpy as np
import pytest

from src.data_gen import generate_dataset
from src.evaluation import run_full_evaluation, FEATURES, make_splits, evaluate_split
from src.causal_model import XLearner
from src.active_learning import run_active_learning


@pytest.fixture(scope="module")
def df():
    return generate_dataset(seed=0)


@pytest.fixture(scope="module")
def results(df):
    return run_full_evaluation(df)


def test_dataset_shape_and_structure(df):
    assert len(df) > 1000
    assert df["site"].nunique() >= 4
    assert df["period"].nunique() >= 4
    # treatment assignment must be confounded, not random: it should
    # correlate with the covariates that also drive the outcome
    corr = np.corrcoef(df["treatment"], df["x1"])[0, 1]
    assert abs(corr) > 0.05, "treatment assignment should be confounded on x1"


def test_time_and_site_holdouts_are_harder_than_random(results):
    random_acc = results["random"]["no_reweight"]["accuracy"]
    time_acc = results["time"]["no_reweight"]["accuracy"]
    site_acc = results["site"]["no_reweight"]["accuracy"]
    assert time_acc < random_acc, (
        f"time holdout ({time_acc:.3f}) should be harder than random ({random_acc:.3f})"
    )
    assert site_acc < random_acc, (
        f"site holdout ({site_acc:.3f}) should be harder than random ({random_acc:.3f})"
    )


def test_reweighting_improves_generalization_on_average(results):
    no_rw = np.mean([
        results["time"]["no_reweight"]["accuracy"],
        results["site"]["no_reweight"]["accuracy"],
    ])
    rw = np.mean([
        results["time"]["reweight"]["accuracy"],
        results["site"]["reweight"]["accuracy"],
    ])
    assert rw > no_rw, (
        f"reweighted avg accuracy ({rw:.3f}) should beat no-reweight avg ({no_rw:.3f}) "
        "on the distribution-shift (time/site) holdouts"
    )


def test_accuracy_in_reasonable_band(results):
    for split in ("random", "time", "site"):
        acc = results[split]["reweight"]["accuracy"]
        assert 0.60 < acc < 0.97, f"{split} reweighted accuracy {acc:.3f} out of reasonable band"


def test_active_learning_reduces_pehe(df):
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(df))
    cut = int(len(df) * 0.7)
    pool = df.iloc[perm[:cut]]
    ev = df.iloc[perm[cut:]]
    X_pool = pool[FEATURES].to_numpy()
    T_pool = pool["treatment"].to_numpy()
    Y_pool = pool["outcome"].to_numpy()
    X_eval = ev[FEATURES].to_numpy()
    true_cate_eval = ev["true_cate"].to_numpy()

    history = run_active_learning(
        X_pool, T_pool, Y_pool, X_eval, true_cate_eval,
        n_init=300, n_rounds=4, k_per_round=200, seed=3,
    )
    assert history[-1]["pehe"] < history[0]["pehe"], (
        "active learning should reduce CATE error (PEHE) from first to last round"
    )


def test_injected_sanity_check_shuffled_treatment_no_signal(df):
    """
    Sanity check against label leakage: if we shuffle the treatment
    assignment (destroying any real T/Y/X relationship) the X-learner's
    CATE sign predictions should be no better than chance on a random
    holdout. If this test fails, something in the pipeline is leaking
    the true effect label into the features.
    """
    rng = np.random.default_rng(123)
    train_idx, test_idx = make_splits(df)["random"]
    train = df.loc[train_idx].copy()
    test = df.loc[test_idx]

    shuffled_T = rng.permutation(train["treatment"].to_numpy())
    X_train = train[FEATURES].to_numpy()
    Y_train = train["outcome"].to_numpy()
    X_test = test[FEATURES].to_numpy()
    true_cate_test = test["true_cate"].to_numpy()

    model = XLearner().fit(X_train, shuffled_T, Y_train)
    pred = model.predict_cate(X_test)
    acc = np.mean((pred > 0) == (true_cate_test > 0))
    assert acc < 0.62, (
        f"model trained on shuffled treatment got suspiciously high accuracy {acc:.3f}; "
        "possible label leakage"
    )
