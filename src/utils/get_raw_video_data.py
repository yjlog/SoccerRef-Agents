"""
SoccerRef-Agents: SoccerNet Data Downloader
--------------------------------------------
Downloads raw video data from SoccerNet-MVFoul dataset.

Usage:
    python -m src.utils.get_raw_video_data

Environment variables required:
    SOCCERNET_PASSWORD
"""

import os
import logging
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_mvfouls(local_dir: str = "SoccerNet", split: Optional[List[str]] = None) -> None:
    """
    Downloads the mvfouls task data from SoccerNet.

    Args:
        local_dir: Local directory to store downloaded data.
        split: Dataset splits to download (default: ["valid"]).
    """
    from SoccerNet.Downloader import SoccerNetDownloader as SNdl

    password = os.environ.get("SOCCERNET_PASSWORD", "")
    if not password:
        logger.error("SOCCERNET_PASSWORD is not set. Please configure your .env file.")
        return

    if split is None:
        split = ["valid"]

    downloader = SNdl(LocalDirectory=local_dir)
    downloader.downloadDataTask(task="mvfouls", split=split, password=password)


if __name__ == "__main__":
    download_mvfouls()
