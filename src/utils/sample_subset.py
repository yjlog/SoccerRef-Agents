import json
import random
import os

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
            print(f"Warning: Data count ({total_count}) is less than the target sample size ({sample_size}). Keeping all data.")
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