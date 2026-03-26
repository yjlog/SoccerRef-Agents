"""
SoccerRef-Agents: Data Subset Sampler
--------------------------------------
Randomly extracts a subset of entries from a JSON dataset file.

Usage:
    python -m src.utils.sample_subset
"""

import json
import logging
import os
import random
from typing import List, Any, Optional

logger = logging.getLogger(__name__)


def extract_subset(
    input_file: str,
    output_file: str,
    sample_size: int = 600,
    seed: Optional[int] = None,
) -> None:
    """
    Randomly extracts a subset of data from a JSON dataset file.

    Args:
        input_file: Path to the source JSON file.
        output_file: Path to save the extracted subset JSON file.
        sample_size: The number of items to extract. Defaults to 600.
        seed: Optional random seed for reproducibility.
    """
    if not os.path.exists(input_file):
        logger.error("File not found: %s", input_file)
        return

    logger.info("Reading %s ...", input_file)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data: List[Any] = json.load(f)

        total_count = len(data)
        logger.info("Original data count: %d entries", total_count)

        if total_count <= sample_size:
            logger.warning(
                "Data count (%d) is less than the target sample size (%d). Keeping all data.",
                total_count, sample_size,
            )
            selected_data = data
        else:
            if seed is not None:
                random.seed(seed)
            logger.info("Randomly extracting %d entries...", sample_size)
            selected_data = random.sample(data, sample_size)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(selected_data, f, ensure_ascii=False, indent=2)

        logger.info("Success! %d entries saved to %s", len(selected_data), output_file)

    except json.JSONDecodeError:
        logger.error("Invalid JSON file format. Please check if it is a standard JSON list.")
    except Exception as e:
        logger.error("An unknown error occurred: %s", e)


if __name__ == "__main__":
    extract_subset(
        input_file="Database/Text/text.json",
        output_file="Database/Text/text_subset.json",
    )
