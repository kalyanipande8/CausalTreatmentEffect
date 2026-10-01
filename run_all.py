"""
End-to-end pipeline:
  1. generate synthetic multi-site/multi-time causal dataset
  2. run the sequential (active learning) experiment selector and show
     PEHE/accuracy improving over acquisition rounds
  3. evaluate the X-learner CATE model on random / time / site holdouts,
     with and without adaptive importance-weighting reweighting
  4. print a results summary (also used by README numbers)
"""
import json
import numpy as np

from src.data_gen import generate_dataset, N_SITES
from src.active_learning import run_active_learning
from src.evaluation import run_full_evaluation, FEATURES


def main():
    print("=" * 70)
    print("1. Generating synthetic multi-site, multi-period causal dataset")
    print("=" * 70)
    df = generate_dataset(seed=0)
    print(f"Rows: {len(df)}, sites: {df['site'].nunique()}, periods: {df['period'].nunique()}")
    print(f"Treatment rate: {df['treatment'].mean():.3f}")
    print(f"Fraction with true benefit (tau>0): {df['benefit_label'].mean():.3f}")

    print()
    print("=" * 70)
    print("2. Sequential experiment selector (active learning, uncertainty sampling)")
    print("=" * 70)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(df))
    cut = int(len(df) * 0.7)
    pool_idx, eval_idx = perm[:cut], perm[cut:]
    pool = df.iloc[pool_idx]
    eval_df = df.iloc[eval_idx]

    X_pool = pool[FEATURES].to_numpy()
    T_pool = pool["treatment"].to_numpy()
    Y_pool = pool["outcome"].to_numpy()
    X_eval = eval_df[FEATURES].to_numpy()
    true_cate_eval = eval_df["true_cate"].to_numpy()

    history = run_active_learning(
        X_pool, T_pool, Y_pool, X_eval, true_cate_eval,
        n_init=300, n_rounds=6, k_per_round=200, seed=3,
    )
    for h in history:
        print(f"  round {h['round']}: n_acquired={h['n_acquired']:5d}  "
              f"PEHE={h['pehe']:.4f}  benefit-accuracy={h['accuracy']:.4f}")

    first_acc, last_acc = history[0]["accuracy"], history[-1]["accuracy"]
    first_pehe, last_pehe = history[0]["pehe"], history[-1]["pehe"]
    print(f"\n  Active learning improvement: accuracy {first_acc:.4f} -> {last_acc:.4f}, "
          f"PEHE {first_pehe:.4f} -> {last_pehe:.4f}")

    print()
    print("=" * 70)
    print("3. Holdout evaluation: random vs. time vs. site, with/without reweighting")
    print("=" * 70)
    results = run_full_evaluation(df)
    summary = {}
    for split_name, res in results.items():
        nrw = res["no_reweight"]
        rw = res["reweight"]
        print(f"\n[{split_name} holdout]  n_train={nrw['n_train']}  n_test={nrw['n_test']}")
        print(f"  no reweighting : accuracy={nrw['accuracy']:.4f}  PEHE={nrw['pehe']:.4f}")
        print(f"  reweighted     : accuracy={rw['accuracy']:.4f}  PEHE={rw['pehe']:.4f}")
        delta = rw["accuracy"] - nrw["accuracy"]
        print(f"  reweighting delta: {delta:+.4f} accuracy")
        summary[split_name] = dict(no_reweight=nrw, reweight=rw, delta_accuracy=delta)

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Random holdout accuracy (no reweight): {summary['random']['no_reweight']['accuracy']:.4f}")
    print(f"Time holdout accuracy   (no reweight): {summary['time']['no_reweight']['accuracy']:.4f}")
    print(f"Site holdout accuracy   (no reweight): {summary['site']['no_reweight']['accuracy']:.4f}")
    print(f"Time holdout accuracy   (reweighted) : {summary['time']['reweight']['accuracy']:.4f}")
    print(f"Site holdout accuracy   (reweighted) : {summary['site']['reweight']['accuracy']:.4f}")

    with open("results.json", "w") as f:
        json.dump(dict(active_learning=history, holdout_summary=summary), f, indent=2)
    print("\nSaved results.json")


if __name__ == "__main__":
    main()
