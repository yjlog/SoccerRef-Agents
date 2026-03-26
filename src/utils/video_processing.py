"""
SoccerRef-Agents: Video Processing Utilities
----------------------------------------------
Utilities for analyzing SoccerNet annotation data, filtering invalid samples,
and transforming data into the VQA dataset format.

Usage:
    python -m src.utils.video_processing
"""

import json
import logging
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .sample_subset import extract_subset  # noqa: F401

logger = logging.getLogger(__name__)


def analyze_severity_distribution(json_path: str, output_dir: str = ".") -> None:
    """
    Analyzes and visualizes the distribution of foul severity from a SoccerNet JSON file.

    Args:
        json_path: The file path to the input JSON dataset.
        output_dir: Directory to save generated plots (default: current directory).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    actions = data.get("Actions", {})

    severity_list: list[str] = []
    action_types: list[str] = []

    for key, details in actions.items():
        sev = details.get("Severity", "")
        sev = "Empty" if sev == "" else str(sev)
        severity_list.append(sev)
        action_types.append(details.get("Action class", "Unknown"))

    df = pd.DataFrame({
        "Severity": severity_list,
        "Action": action_types,
    })

    stats = df["Severity"].value_counts().sort_index()
    percentage = df["Severity"].value_counts(normalize=True).sort_index() * 100

    logger.info("--- Severity Distribution Statistics ---")
    summary = pd.DataFrame({"Count": stats, "Percentage (%)": percentage})
    logger.info("\n%s", summary.to_string())

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax = sns.countplot(
        data=df,
        x="Severity",
        order=sorted(df["Severity"].unique()),
        palette="viridis",
    )

    plt.title("Distribution of Foul Severity", fontsize=15)
    plt.xlabel("Severity Level", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="center",
            xytext=(0, 9),
            textcoords="offset points",
        )

    plt.tight_layout()
    severity_plot_path = os.path.join(output_dir, "severity_distribution.png")
    plt.savefig(severity_plot_path, dpi=150)
    plt.close()
    logger.info("Severity distribution plot saved to %s", severity_plot_path)

    plt.figure(figsize=(14, 7))
    sns.countplot(data=df, x="Action", hue="Severity", palette="magma")
    plt.title("Severity Distribution by Action Class", fontsize=15)
    plt.xticks(rotation=45)
    plt.legend(title="Severity", loc="upper right")
    plt.tight_layout()
    action_plot_path = os.path.join(output_dir, "severity_by_action.png")
    plt.savefig(action_plot_path, dpi=150)
    plt.close()
    logger.info("Action class plot saved to %s", action_plot_path)


def advanced_filter_annotations(input_file: str, output_file: str) -> None:
    """
    Filters annotations based on specific rules to remove invalid or ambiguous samples.

    The filtering logic removes:
    1. Invalid actions (Empty or 'Dont know').
    2. 'Dive' actions.
    3. Ambiguous offences marked as 'Between'.
    4. Logical contradictions where Severity is 0 but Offence is 'Offence'.

    Args:
        input_file: Path to the raw source JSON file.
        output_file: Path to save the cleaned JSON file.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_actions = data["Actions"]
    filtered_actions: dict = {}

    stats = {
        "invalid_action": 0,
        "dive": 0,
        "between": 0,
        "contradiction": 0,
    }

    for key, action in original_actions.items():
        action_class = str(action.get("Action class", "")).strip()
        offence = str(action.get("Offence", "")).strip()
        severity = str(action.get("Severity", "")).strip()

        if action_class == "" or action_class.lower() == "dont know":
            stats["invalid_action"] += 1
            continue

        if action_class.lower() == "dive":
            stats["dive"] += 1
            continue

        if offence.lower() == "between":
            stats["between"] += 1
            continue

        if (severity in ("0", "0.0", "")) and (offence.lower() == "offence"):
            stats["contradiction"] += 1
            continue

        filtered_actions[key] = action

    data["Actions"] = filtered_actions
    data["Number of actions"] = len(filtered_actions)

    total_removed = sum(stats.values())
    logger.info("=" * 40)
    logger.info("Data Filtering Report:")
    logger.info("Original Total Samples: %d", len(original_actions))
    logger.info("  - Removed Invalid Action (Empty/Dont know): %d", stats["invalid_action"])
    logger.info("  - Removed Dives: %d", stats["dive"])
    logger.info("  - Removed Ambiguous (Between): %d", stats["between"])
    logger.info("  - Removed Contradictions (Sev=0 & Offence): %d", stats["contradiction"])
    logger.info("Total Removed: %d", total_removed)
    logger.info("Final Valid Samples: %d", len(filtered_actions))
    logger.info("=" * 40)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info("Results saved to: %s", output_file)


def transform_to_vqa_dataset(input_file: str, output_file: str) -> None:
    """
    Transforms the SoccerNet annotation format into a Visual Question Answering (VQA) dataset format.

    Maps severity scores to four decision categories (O1-O4), constructs standard video paths,
    and formats the data for model training/evaluation.

    Args:
        input_file: Path to the cleaned (or raw) JSON file.
        output_file: Path to save the VQA-formatted JSON file.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    vqa_dataset: list[dict] = []

    dataset_split = source_data.get("Set", "train").lower()

    OPTIONS = {
        "O1": "No offence",
        "O2": "Offence with no card",
        "O3": "Offence with yellow card",
        "O4": "Offence with possible red card",
    }

    actions = source_data.get("Actions", {})

    for key, action in actions.items():
        action_class = str(action.get("Action class", "")).strip()
        offence = str(action.get("Offence", "")).strip().lower()
        severity = str(action.get("Severity", "")).strip()

        clip_1_path = f"Dataset/video/SoccerNet/mvfouls/{dataset_split}/action_{key}/clip_1.mp4"

        if action_class == "" or action_class.lower() == "dont know":
            continue
        if action_class.lower() == "dive":
            continue
        if offence == "between":
            continue
        if (severity in ("0", "0.0")) and (offence == "offence"):
            continue

        close_a_key = _severity_to_option(severity)

        vqa_entry = {
            "Q": "Based on the following foul video, what decision do you think the head referee should make?",
            "materials": [
                {
                    "path": clip_1_path,
                    "context": action.get("UrlLocal", ""),
                }
            ],
            "openA": OPTIONS[close_a_key],
            "closeA": close_a_key,
            "O1": OPTIONS["O1"],
            "O2": OPTIONS["O2"],
            "O3": OPTIONS["O3"],
            "O4": OPTIONS["O4"],
        }

        vqa_dataset.append(vqa_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vqa_dataset, f, indent=4, ensure_ascii=False)

    logger.info("Transformation complete! Generated %d samples.", len(vqa_dataset))


def _severity_to_option(severity: str) -> str:
    """Maps a severity string to an option key (O1-O4)."""
    severity_float = float(severity) if severity else 0.0

    if severity_float <= 0:
        return "O1"
    elif severity_float <= 1.5:
        return "O2"
    elif severity_float <= 3.5:
        return "O3"
    else:
        return "O4"
