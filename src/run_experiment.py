"""Run regression, non-parametric tests, summaries, and publication-quality charts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "pr_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
CONTROL_COLUMNS = [
    "no_commits",
    "PR_size",
    "no_changed_files",
    "project_age_month",
]
TASK_CATEGORIES = [
    "Enhancement",
    "Bug Fix",
    "Documentation",
    "Implementation",
    "Testing",
    "Other",
]
TASK_PHASES = {
    "Submission": "task_submission",
    "Review": "task_review",
    "Waiting for change": "task_waiting_change",
    "Code change": "task_change",
    "Waiting after change": "task_waiting_after",
}
TEST_METRICS = [
    "time_to_merge_h",
    "at_review_h",
    "at_waiting_change_h",
    "at_change_h",
]


def _validate_dataset(data: pd.DataFrame) -> None:
    required = {
        "is_rag_assisted",
        "time_to_merge_h",
        "at_review_h",
        "at_waiting_change_h",
        "at_change_h",
        *CONTROL_COLUMNS,
        *TASK_PHASES.values(),
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan di dataset: {', '.join(missing)}")
    if data.empty or data["is_rag_assisted"].nunique() != 2:
        raise ValueError("Dataset harus berisi data untuk kedua kelompok perlakuan.")


def _make_charts(data: pd.DataFrame, task_counts: pd.DataFrame) -> None:
    labels = ["Non-assisted", "RAG-assisted"]
    groups = [
        data.loc[data["is_rag_assisted"].eq(group), "time_to_merge_h"]
        for group in (0, 1)
    ]

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(groups, tick_labels=labels, patch_artist=True)
    axis.set(title="Distribusi waktu hingga merge", ylabel="Jam")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart1_ttm_boxplot.png", dpi=300)
    plt.close(fig)

    phase_columns = [
        "at_submission_h",
        "at_review_h",
        "at_waiting_change_h",
        "at_change_h",
        "at_waiting_after_h",
    ]
    phase_labels = [
        "Submission",
        "Review",
        "Waiting\nfor change",
        "Code\nchange",
        "Waiting\nafter change",
    ]
    means = data.groupby("is_rag_assisted")[phase_columns].mean()
    x = np.arange(len(phase_columns))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - width / 2, means.loc[0], width, label=labels[0])
    axis.bar(x + width / 2, means.loc[1], width, label=labels[1])
    axis.set(
        title="Rerata durasi per fase PR",
        ylabel="Jam (rerata)",
        xticks=x,
        xticklabels=phase_labels,
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart2_phases_bar.png", dpi=300)
    plt.close(fig)

    phase_names = list(TASK_PHASES)
    counts = task_counts.pivot(
        index="phase", columns="task_category", values="count"
    ).reindex(index=phase_names, columns=TASK_CATEGORIES, fill_value=0)
    fig, axis = plt.subplots(figsize=(11, 6))
    counts.plot(kind="bar", ax=axis, width=0.82)
    axis.set(
        title="Distribusi kategori tugas code review per fase",
        ylabel="Jumlah PR",
        xlabel="Fase PR",
    )
    axis.tick_params(axis="x", rotation=20)
    axis.legend(title="Kategori", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart3_tasks_bar.png", dpi=300)
    plt.close(fig)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}. Jalankan src/generate_dataset.py terlebih dahulu."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    _validate_dataset(data)

    scaled_controls = MinMaxScaler().fit_transform(data[CONTROL_COLUMNS])
    distance_matrix = manhattan_distances(scaled_controls)
    groups_array = data["is_rag_assisted"].to_numpy()
    nearest_opposite = []
    for index, group in enumerate(groups_array):
        opposite = np.flatnonzero(groups_array != group)
        nearest_opposite.append(distance_matrix[index, opposite].min())
    with (OUTPUT_DIR / "manhattan_matching_summary.txt").open(
        "w", encoding="utf-8"
    ) as report:
        report.write("Jarak Manhattan fitur kontrol setelah Min-Max scaling [0, 1]\n")
        report.write(f"Rerata jarak PR terdekat kelompok berlawanan: {np.mean(nearest_opposite):.6f}\n")
        report.write(f"Median jarak PR terdekat kelompok berlawanan: {np.median(nearest_opposite):.6f}\n")

    regression_data = pd.DataFrame(
        {
            "log1p_time_to_merge_h": np.log1p(data["time_to_merge_h"]),
            "is_rag_assisted": data["is_rag_assisted"],
            "log1p_no_commits": np.log1p(data["no_commits"]),
            "log1p_PR_size": np.log1p(data["PR_size"]),
            "log1p_no_changed_files": np.log1p(data["no_changed_files"]),
            "log1p_project_age_month": np.log1p(data["project_age_month"]),
        }
    )
    predictors = sm.add_constant(
        regression_data.drop(columns="log1p_time_to_merge_h")
    )
    regression = sm.OLS(
        regression_data["log1p_time_to_merge_h"], predictors
    ).fit()
    (OUTPUT_DIR / "regression_summary.txt").write_text(
        regression.summary().as_text(), encoding="utf-8"
    )

    test_rows = []
    for metric in TEST_METRICS:
        non_assisted = data.loc[data["is_rag_assisted"].eq(0), metric].dropna()
        rag_assisted = data.loc[data["is_rag_assisted"].eq(1), metric].dropna()
        result = mannwhitneyu(
            rag_assisted, non_assisted, alternative="two-sided", method="auto"
        )
        test_rows.append(
            {
                "metric": metric,
                "non_assisted_n": len(non_assisted),
                "rag_assisted_n": len(rag_assisted),
                "non_assisted_median_h": non_assisted.median(),
                "rag_assisted_median_h": rag_assisted.median(),
                "u_statistic": result.statistic,
                "p_value_two_sided": result.pvalue,
                "alpha": 0.05,
                "significant": result.pvalue < 0.05,
            }
        )
    pd.DataFrame(test_rows).to_csv(
        OUTPUT_DIR / "mann_whitney_results.csv", index=False, float_format="%.8g"
    )

    task_rows = []
    for phase, column in TASK_PHASES.items():
        counts = data[column].value_counts()
        for category in TASK_CATEGORIES:
            task_rows.append(
                {
                    "phase": phase,
                    "task_category": category,
                    "count": int(counts.get(category, 0)),
                }
            )
    task_counts = pd.DataFrame(task_rows)
    task_counts.to_csv(OUTPUT_DIR / "task_category_counts.csv", index=False)
    _make_charts(data, task_counts)

    print(f"Analisis selesai. Output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
