import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

def analyze_severity_distribution(json_path):
    """
    Analyzes and visualizes the distribution of foul severity from a SoccerNet JSON file.

    This function loads the annotation data, calculates statistics for severity levels,
    and generates bar charts showing the overall severity distribution and the
    distribution by action class.

    Args:
        json_path (str): The file path to the input JSON dataset.
    """
    # 1. Load data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    actions = data.get("Actions", {})

    # 2. Extract all Severity labels
    severity_list = []
    action_types = []

    for key, details in actions.items():
        # Get Severity; mark as 'Missing' if empty
        sev = details.get("Severity", "")
        sev = "Empty" if sev == "" else str(sev)
        severity_list.append(sev)

        # Record Action Class for cross-analysis
        action_types.append(details.get("Action class", "Unknown"))

    # 3. Convert to DataFrame
    df = pd.DataFrame({
        'Severity': severity_list,
        'Action': action_types
    })

    # 4. Calculate statistics
    stats = df['Severity'].value_counts().sort_index()
    percentage = df['Severity'].value_counts(normalize=True).sort_index() * 100

    print("--- Severity Distribution Statistics ---")
    summary = pd.DataFrame({'Count': stats, 'Percentage (%)': percentage})
    print(summary)

    # 5. Visualization
    plt.figure(figsize=(12, 6))

    # Plot Bar Chart
    sns.set_style("whitegrid")
    ax = sns.countplot(
        data=df,
        x='Severity',
        order=sorted(df['Severity'].unique()),
        palette="viridis"
    )

    plt.title('Distribution of Foul Severity', fontsize=15)
    plt.xlabel('Severity Level', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # Add numerical labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')

    plt.tight_layout()
    plt.show()

    # 6. Advanced: Cross-distribution of Action vs. Severity (Optional)
    plt.figure(figsize=(14, 7))
    sns.countplot(data=df, x='Action', hue='Severity', palette="magma")
    plt.title('Severity Distribution by Action Class', fontsize=15)
    plt.xticks(rotation=45)
    plt.legend(title='Severity', loc='upper right')
    plt.tight_layout()
    plt.show()


def advanced_filter_annotations(input_file, output_file):
    """
    Filters annotations based on specific rules to remove invalid or ambiguous samples.

    The filtering logic removes:
    1. Invalid actions (Empty or 'Dont know').
    2. 'Dive' actions.
    3. Ambiguous offences marked as 'Between'.
    4. Logical contradictions where Severity is 0 but Offence is 'Offence'.

    Args:
        input_file (str): Path to the raw source JSON file.
        output_file (str): Path to save the cleaned JSON file.
    """
    # 1. Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_actions = data["Actions"]
    filtered_actions = {}

    # Counter for statistics on removed data
    stats = {
        "invalid_action": 0,    # Action class empty or 'Dont know'
        "dive": 0,              # Simulation/Dive
        "between": 0,           # Ambiguous foul judgment
        "contradiction": 0      # Sev=0 but marked as Offence
    }

    for key, action in original_actions.items():
        # Preprocess field values
        action_class = str(action.get("Action class", "")).strip()
        offence = str(action.get("Offence", "")).strip()
        severity = str(action.get("Severity", "")).strip()

        # --- Filtering Logic Start ---

        # 1. Action class is empty or "Dont know"
        if action_class == "" or action_class.lower() == "dont know":
            stats["invalid_action"] += 1
            continue

        # 2. Action class is "Dive"
        if action_class.lower() == "dive":
            stats["dive"] += 1
            continue

        # 3. Offence is "Between" (Uncertain foul)
        if offence.lower() == "between":
            stats["between"] += 1
            continue

        # 4. Severity is 0 but Offence is "Offence" (Logical contradiction)
        if (severity in ["0", "0.0", ""]) and (offence.lower() == "offence"):
            stats["contradiction"] += 1
            continue

        # --- Filtering Logic End ---

        # Keep the sample if it passes all checks
        filtered_actions[key] = action

    # 2. Update metadata
    data["Actions"] = filtered_actions
    data["Number of actions"] = len(filtered_actions)

    # 3. Print statistics
    total_removed = sum(stats.values())
    print("="*40)
    print(f"Data Filtering Report:")
    print(f"Original Total Samples: {len(original_actions)}")
    print(f"  - Removed Invalid Action (Empty/Dont know): {stats['invalid_action']}")
    print(f"  - Removed Dives: {stats['dive']}")
    print(f"  - Removed Ambiguous (Between): {stats['between']}")
    print(f"  - Removed Contradictions (Sev=0 & Offence): {stats['contradiction']}")
    print(f"Total Removed: {total_removed}")
    print(f"Final Valid Samples: {len(filtered_actions)}")
    print("="*40)

    # 4. Save file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Results saved to: {output_file}")


def transform_to_vqa_dataset(input_file, output_file):
    """
    Transforms the SoccerNet annotation format into a Visual Question Answering (VQA) dataset format.

    It maps severity scores to four decision categories (O1-O4), constructs standard video paths,
    and formats the data for model training/evaluation. It applies the same filtering logic
    as `advanced_filter_annotations` during the process.

    Args:
        input_file (str): Path to the cleaned (or raw) JSON file.
        output_file (str): Path to save the VQA-formatted JSON file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    vqa_dataset = []

    # Get dataset split (train/val/test) from the "Set" field in raw JSON
    dataset_split = source_data.get("Set", "train").lower()

    OPTIONS = {
        "O1": "No offence",
        "O2": "Offence with no card",
        "O3": "Offence with yellow card",
        "O4": "Offence with possible red card"
    }

    actions = source_data.get("Actions", {})

    for key, action in actions.items():
        # --- 1. Extract Core Fields ---
        action_class = str(action.get("Action class", "")).strip()
        offence = str(action.get("Offence", "")).strip().lower()
        severity = str(action.get("Severity", "")).strip()
        url_local = action.get("UrlLocal", "")

        # --- 2. Construct Correct Video Path ---
        # Target format: Dataset/video/SoccerNet/mvfouls/[split]/action_[key]/clip_1.mp4
        clip_1_path = f"Dataset/video/SoccerNet/mvfouls/{dataset_split}/action_{key}/clip_1.mp4"

        # --- 3. Filtering Logic (Preserved) ---
        if action_class == "" or action_class.lower() == "dont know": continue
        if action_class.lower() == "dive": continue
        if offence == "between": continue
        if (severity in ["0", "0.0"]) and (offence == "offence"): continue

        # --- 4. Determine closeA (Severity Mapping) ---
        if severity == "" or severity == "0" or severity == "0.0":
            close_a_key = "O1"
        elif severity == "1.0" or severity == "1":
            close_a_key = "O2"
        elif severity == "3.0" or severity == "3":
            close_a_key = "O3"
        elif severity in ["4", "4.0", "5", "5.0"]:
            close_a_key = "O4"
        else:
            # Fallback (logic from original code)
            close_a_key = "O1"

        # --- 5. Construct VQA Entry ---
        vqa_entry = {
            "Q": "Based on the following foul video, what decision do you think the head referee should make?",
            "materials": [
                {
                    "path": clip_1_path,
                    "context": url_local
                }
            ],
            "openA": OPTIONS[close_a_key],
            "closeA": close_a_key,
            "O1": OPTIONS["O1"],
            "O2": OPTIONS["O2"],
            "O3": OPTIONS["O3"],
            "O4": OPTIONS["O4"]
        }

        vqa_dataset.append(vqa_entry)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vqa_dataset, f, indent=4, ensure_ascii=False)

    print(f"Transformation complete! Generated {len(vqa_dataset)} samples using SoccerNet standard paths.")


def extract_subset(input_file, output_file, sample_size=600):
    """
    Randomly extracts a subset of data from a JSON dataset file.

    This function loads a JSON dataset, checks its size, and randomly samples
    a specified number of entries without replacement. If the dataset is smaller
    than the requested sample size, it retains all data.

    Args:
        input_file (str): Path to the source JSON file.
        output_file (str): Path to save the extracted subset JSON file.
        sample_size (int, optional): The number of items to extract. Defaults to 600.
    """
    # 1. Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: File not found {input_file}")
        return

    print(f"Reading {input_file} ...")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_count = len(data)
        print(f"Original data count: {total_count} entries")

        # 2. Extraction Logic
        if total_count <= sample_size:
            print(
                f"Warning: Data count ({total_count}) is less than the target sample size ({sample_size}). Keeping all data.")
            selected_data = data
        else:
            # Random sampling without replacement
            print(f"Randomly extracting {sample_size} entries...")
            selected_data = random.sample(data, sample_size)

            # If you want the first N entries instead of random sampling, use the line below:
            # selected_data = data[:sample_size]

        # 3. Save the new file
        with open(output_file, 'w', encoding='utf-8') as f:
            # ensure_ascii=False ensures characters are displayed correctly (not \uXXXX)
            # indent=2 makes the format readable
            json.dump(selected_data, f, ensure_ascii=False, indent=2)

        print(f"Success! {len(selected_data)} entries saved to {output_file}")

    except json.JSONDecodeError:
        print("Error: Invalid JSON file format. Please check if it is a standard JSON list.")
    except Exception as e:
        print(f"An unknown error occurred: {e}")