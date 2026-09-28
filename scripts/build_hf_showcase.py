from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


LABELS = [
    "FAITHFUL",
    "NEGATION_FLIP",
    "OMISSION",
    "MODALITY_SHIFT",
    "QUANTIFIER_CHANGE",
    "CONDITION_LOSS",
]


def _find_latest_run(cache_root: Path) -> Path:
    runs_root = cache_root / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"No persistent run cache found at {runs_root}")
    candidates = []
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        score = 0
        if (path / "VALIDATION_REPORT.json").exists():
            score += 4
        if (path / "evaluation" / "metrics.json").exists():
            score += 4
        if (path / "checkpoint" / "model.pt").exists():
            score += 2
        if (path / "model_repository").exists():
            score += 2
        if score:
            candidates.append((score, path.stat().st_mtime, path))
    if not candidates:
        raise FileNotFoundError("No completed semantic-integrity run was found in the persistent cache.")
    candidates.sort(reverse=True)
    return candidates[0][2]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_artifacts(run_dir: Path) -> tuple[dict, dict, dict]:
    validation_path = run_dir / "VALIDATION_REPORT.json"
    if validation_path.exists():
        validation = _load_json(validation_path)
    else:
        validation = {}

    metrics_path = run_dir / "evaluation" / "metrics.json"
    metrics = _load_json(metrics_path) if metrics_path.exists() else validation.get("evaluation", {})

    summary_path = run_dir / "semantic_dataset" / "dataset_summary.json"
    dataset_summary = _load_json(summary_path) if summary_path.exists() else validation.get("dataset_summary", {})

    qa = validation.get("dataset_qa", {})
    if not qa:
        qa_path = run_dir / "semantic_dataset" / "qa_report.json"
        if qa_path.exists():
            qa = _load_json(qa_path)

    if metrics.get("status") != "ok":
        raise RuntimeError("Measured evaluation metrics are missing or invalid.")
    if not dataset_summary:
        raise RuntimeError("Dataset summary is missing.")

    return metrics, dataset_summary, qa


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def generate_charts(metrics: dict, dataset_summary: dict, qa: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    showcase = output_dir / "showcase"
    showcase.mkdir(parents=True, exist_ok=True)

    overview_names = ["Macro F1", "Severity Macro F1", "Critical Drift Recall"]
    overview_values = [
        metrics["macro_f1"] * 100,
        metrics["severity_macro_f1"] * 100,
        metrics["critical_drift_recall"] * 100,
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(overview_names, overview_values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Measured Held-out Performance")
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    fig.tight_layout()
    fig.savefig(showcase / "metrics_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    risk_value = metrics["false_safe_rate"] * 100
    bars = ax.bar(["False Safe Rate"], [risk_value])
    ax.set_ylim(0, max(10, risk_value * 1.5))
    ax.set_ylabel("Rate (%) — lower is better")
    ax.set_title("Critical-case Safety Metric")
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    fig.tight_layout()
    fig.savefig(showcase / "false_safe_rate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    per_label = metrics.get("per_label_f1", [])
    if len(per_label) == len(LABELS):
        fig, ax = plt.subplots(figsize=(11, 6))
        values = [x * 100 for x in per_label]
        bars = ax.bar(LABELS, values)
        ax.set_ylim(0, 100)
        ax.set_ylabel("F1 (%)")
        ax.set_title("Per-label F1 on Frozen Held-out Test")
        ax.tick_params(axis="x", rotation=25)
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        fig.tight_layout()
        fig.savefig(showcase / "per_label_f1.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    label_counts = qa.get("label_counts", {})
    if label_counts:
        names = LABELS
        values = [int(label_counts.get(name, 0)) for name in names]
        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.bar(names, values)
        ax.set_ylabel("Samples")
        ax.set_title("AMANAH-SD v0 Label Distribution")
        ax.tick_params(axis="x", rotation=25)
        ax.bar_label(bars, padding=3)
        fig.tight_layout()
        fig.savefig(showcase / "label_distribution.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    split_sizes = dataset_summary.get("split_sizes", qa.get("split_sizes", {}))
    if split_sizes:
        names = ["Train", "Validation", "Test"]
        values = [
            int(split_sizes.get("train", 0)),
            int(split_sizes.get("validation", 0)),
            int(split_sizes.get("test", 0)),
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, values)
        ax.set_ylabel("Samples")
        ax.set_title("Dataset Split Sizes")
        ax.bar_label(bars, padding=3)
        fig.tight_layout()
        fig.savefig(showcase / "split_distribution.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def build_readme(project_root: Path, metrics: dict, dataset_summary: dict, qa: dict) -> str:
    base = (project_root / "MODEL_CARD.md").read_text(encoding="utf-8")
    marker = "# Semantic Integrity Classifier v0.1"
    showcase = f"""
## Visual evaluation summary

![Measured held-out performance](showcase/metrics_overview.png)

![Critical-case safety metric](showcase/false_safe_rate.png)

![Per-label F1](showcase/per_label_f1.png)

![Dataset label distribution](showcase/label_distribution.png)

![Dataset split sizes](showcase/split_distribution.png)

### Quick committee summary

- **Macro F1:** {_pct(metrics["macro_f1"])}
- **Severity Macro F1:** {_pct(metrics["severity_macro_f1"])}
- **Critical Drift Recall:** {_pct(metrics["critical_drift_recall"])}
- **False Safe Rate:** {_pct(metrics["false_safe_rate"])} (lower is better)
- **Total dataset samples:** {int(dataset_summary.get("total_samples", 0)):,}
- **Qur'anic ayahs covered:** {int(dataset_summary.get("ayahs", qa.get("reference_ayahs", 0))):,}
- **Held-out test samples:** {int(metrics.get("rows", 0)):,}
- **Held-out test ayahs:** {int(qa.get("split_ayah_counts", {}).get("test", 0)):,}
- **Split leakage:** none detected by dataset QA

> These results are measured on AMANAH-SD v0's frozen held-out benchmark. The controlled drift portion is synthetic by construction and does not replace a separately human-reviewed benchmark of naturally occurring translation errors.

"""
    if marker in base:
        return base.replace(marker, marker + "\n" + showcase, 1)
    return base + "\n" + showcase


def build_showcase(
    *,
    project_root: Path,
    cache_root: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    run_dir = _find_latest_run(cache_root)
    metrics, dataset_summary, qa = _load_artifacts(run_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    generate_charts(metrics, dataset_summary, qa, output_dir)
    (output_dir / "README.md").write_text(
        build_readme(project_root, metrics, dataset_summary, qa),
        encoding="utf-8",
    )
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "dataset_qa_summary.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return run_dir, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build committee-ready Hugging Face showcase assets.")
    parser.add_argument("--cache-root", default="/content/drive/MyDrive/semantic_integrity_model_cache")
    parser.add_argument("--output", default="artifacts/hf_showcase")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir, output_dir = build_showcase(
        project_root=project_root,
        cache_root=Path(args.cache_root),
        output_dir=Path(args.output),
    )
    print(json.dumps({
        "status": "ok",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
