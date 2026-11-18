"""Part 0.1: create every fixed, stratified, seeded partition once and save the indices.

Run once:  python scripts/make_splits.py
Re-running reproduces identical files (same seed); the script refuses to
overwrite different existing splits, so partitions can never silently change.

Labels are used here only to stratify (preserve class proportions). They are
never used to compute distances, fit centroids or assign clusters.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASS_NAMES, SEED, SPLITS_DIR, load_official  # noqa: E402


def stratified_pick(indices, labels, n, seed):
    """Return a stratified random subset of `indices` of size n."""
    picked, _ = train_test_split(indices, train_size=n, stratify=labels, random_state=seed)
    return np.sort(picked)


def main():
    _, y_train_official, _, y_test = load_official()
    all_idx = np.arange(len(y_train_official))

    # 0.1.2  50,000 train / 10,000 validation from the official 60,000
    train_idx, val_idx = train_test_split(
        all_idx, test_size=10_000, stratify=y_train_official, random_state=SEED
    )
    train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)

    # 0.1.4  CPU-friendly option: fixed stratified 10k subset of the training partition
    train_subset10k_idx = stratified_pick(train_idx, y_train_official[train_idx], 10_000, SEED)

    # 0.1.5  clustering / feature-extraction subsets (same images for every representation)
    cluster_train_idx = stratified_pick(train_idx, y_train_official[train_idx], 3_000, SEED)
    cluster_val_idx = stratified_pick(val_idx, y_train_official[val_idx], 1_000, SEED)
    test_all = np.arange(len(y_test))
    cluster_test_idx = stratified_pick(test_all, y_test, 1_000, SEED)  # indices into the official test set

    arrays = dict(
        train_idx=train_idx, val_idx=val_idx, train_subset10k_idx=train_subset10k_idx,
        cluster_train_idx=cluster_train_idx, cluster_val_idx=cluster_val_idx,
        cluster_test_idx=cluster_test_idx,
    )

    # sanity checks
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(train_idx) + len(val_idx) == 60_000
    assert np.isin(train_subset10k_idx, train_idx).all()
    assert np.isin(cluster_train_idx, train_idx).all() and np.isin(cluster_val_idx, val_idx).all()

    def counts(y):
        c = np.bincount(y, minlength=10)
        return {CLASS_NAMES[i]: int(c[i]) for i in range(10)}

    summary = {
        "seed": SEED,
        "method": "sklearn.model_selection.train_test_split(stratify=labels, random_state=seed)",
        "source": {
            "train_idx": "official train", "val_idx": "official train",
            "train_subset10k_idx": "official train (subset of train_idx)",
            "cluster_train_idx": "official train (subset of train_idx)",
            "cluster_val_idx": "official train (subset of val_idx)",
            "cluster_test_idx": "official test",
        },
        "sizes": {k: int(len(v)) for k, v in arrays.items()},
        "class_counts": {
            "train_idx": counts(y_train_official[train_idx]),
            "val_idx": counts(y_train_official[val_idx]),
            "train_subset10k_idx": counts(y_train_official[train_subset10k_idx]),
            "cluster_train_idx": counts(y_train_official[cluster_train_idx]),
            "cluster_val_idx": counts(y_train_official[cluster_val_idx]),
            "cluster_test_idx": counts(y_test[cluster_test_idx]),
            "official_test": counts(y_test),
        },
    }

    SPLITS_DIR.mkdir(exist_ok=True)
    npz_path = SPLITS_DIR / f"splits_seed{SEED}.npz"
    if npz_path.exists():
        with np.load(npz_path) as old:
            same = all(np.array_equal(old[k], v) for k, v in arrays.items())
        if not same:
            sys.exit(f"Refusing to overwrite {npz_path}: existing splits differ. Partitions are frozen.")
        print("Existing splits reproduced exactly - nothing changed.")
        return
    np.savez_compressed(npz_path, **arrays)
    with open(SPLITS_DIR / f"splits_seed{SEED}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["sizes"], indent=2))


if __name__ == "__main__":
    main()
