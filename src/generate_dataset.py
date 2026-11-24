"""Generate a reproducible, controlled synthetic pull-request dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler


SEED = 42
GROUP_SIZE = 150
CONTROL_COLUMNS = [
    "no_commits",
    "PR_size",
    "no_changed_files",
    "project_age_month",
]
CATEGORIES = [
    "Enhancement",
    "Bug Fix",
    "Documentation",
    "Implementation",
    "Testing",
    "Other",
]
CATEGORY_PROBABILITIES = [0.18, 0.22, 0.12, 0.24, 0.14, 0.10]
PHASE_TARGET_MEANS = {
    0: {
        "at_submission_h": 0.50,
        "at_review_h": 2.83,
        "at_waiting_change_h": 21.99,
        "at_change_h": 1.06,
        "at_waiting_after_h": 14.17,
    },
    1: {
        "at_submission_h": 0.50,
        "at_review_h": 0.94,
        "at_waiting_change_h": 3.00,
        "at_change_h": 1.02,
        "at_waiting_after_h": 15.49624,
    },
}


def _positive_values_with_mean(
    rng: np.random.Generator, size: int, target_mean: float
) -> np.ndarray:
    """Create positive, right-skewed durations calibrated to a target mean."""
    values = rng.lognormal(mean=0.0, sigma=0.35, size=size)
    return values * (target_mean / values.mean())


def main() -> None:
    rng = np.random.default_rng(SEED)
    n = GROUP_SIZE * 2
    treatment = np.repeat([0, 1], GROUP_SIZE)
    rng.shuffle(treatment)

    data = pd.DataFrame(
        {
            "pr_id": [f"PR-{index:04d}" for index in range(1, n + 1)],
            "is_rag_assisted": treatment,
            "project_age_month": rng.integers(3, 121, size=n),
            "no_commits": np.clip(rng.poisson(4, size=n) + 1, 1, None),
            "PR_size": np.clip(rng.lognormal(4.0, 0.8, size=n).round().astype(int), 2, None),
            "no_changed_files": np.clip(rng.poisson(5, size=n) + 1, 1, None),
        }
    )

    phase_columns = [
        "at_submission_h",
        "at_review_h",
        "at_waiting_change_h",
        "at_change_h",
        "at_waiting_after_h",
    ]
    for group, target_means in PHASE_TARGET_MEANS.items():
        group_indices = np.flatnonzero(data["is_rag_assisted"].to_numpy() == group)
        for column in phase_columns:
            data.loc[group_indices, column] = _positive_values_with_mean(
                rng, len(group_indices), target_means[column]
            )

    data["time_to_merge_h"] = data[phase_columns].sum(axis=1)
    data["vector_cosine_sim"] = np.nan
    assisted_indices = data["is_rag_assisted"].eq(1)
    data.loc[assisted_indices, "vector_cosine_sim"] = rng.uniform(
        0.65, 0.98, size=int(assisted_indices.sum())
    )

    for phase in ("submission", "review", "waiting_change", "change", "waiting_after"):
        data[f"task_{phase}"] = rng.choice(
            CATEGORIES, size=n, p=CATEGORY_PROBABILITIES
        )

    scaled_controls = MinMaxScaler().fit_transform(data[CONTROL_COLUMNS])
    distances = pairwise_distances(scaled_controls, metric="manhattan")
    group_values = data["is_rag_assisted"].to_numpy()
    matching_distances = np.empty(n)
    for row_index in range(n):
        opposite_group = np.flatnonzero(group_values != group_values[row_index])
        matching_distances[row_index] = distances[row_index, opposite_group].min()
    data["manhattan_match_dist"] = matching_distances

    output_path = Path(__file__).resolve().parents[1] / "data" / "pr_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, float_format="%.8f")

    means = data.groupby("is_rag_assisted")[phase_columns + ["time_to_merge_h"]].mean()
    review_reduction = 1 - means.loc[1, "at_review_h"] / means.loc[0, "at_review_h"]
    waiting_reduction = (
        1 - means.loc[1, "at_waiting_change_h"] / means.loc[0, "at_waiting_change_h"]
    )
    merge_reduction = 1 - means.loc[1, "time_to_merge_h"] / means.loc[0, "time_to_merge_h"]
    print(f"Dataset tersimpan: {output_path} ({len(data)} PR)")
    print(
        "Penurunan rerata simulasi: "
        f"review={review_reduction:.2%}, "
        f"waiting_change={waiting_reduction:.2%}, "
        f"time_to_merge={merge_reduction:.2%}"
    )


if __name__ == "__main__":
    main()
