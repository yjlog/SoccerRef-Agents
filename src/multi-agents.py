"""
SoccerRef-Agents: Multi-Agent System Entry Point
-------------------------------------------------
Runs the multi-agent referee system on the text benchmark.

Usage:
    python -m src.multi-agents

Environment variables required (see .env.example):
    OPENAI_API_KEY, OPENAI_BASE_URL, EMBEDDING_API_KEY,
    EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, PROJECT_ROOT, DB_PATH
"""

import json
import os
import logging

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

from agents.orchestrator import MultiAgentRefereeSystem

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting Multi-Agent Referee System...")

    project_root = os.environ.get("PROJECT_ROOT", ".")
    db_path = os.environ.get("DB_PATH", "./referee_vector_db")
    dataset_rel_path = os.environ.get("DATASET_PATH", "Database/Text/text.json")
    output_file = os.environ.get("OUTPUT_FILE", "output.json")
    embedding_api_key = os.environ.get("EMBEDDING_API_KEY")
    embedding_model = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    llm_api_key = os.environ.get("OPENAI_API_KEY")
    llm_base_url = os.environ.get("OPENAI_BASE_URL")
    llm_model_name = os.environ.get("LLM_MODEL_NAME", "gpt-4o")

    if not embedding_api_key:
        logger.error("EMBEDDING_API_KEY is not set. Please configure your .env file.")
        return

    if not llm_api_key:
        logger.error("OPENAI_API_KEY is not set. Please configure your .env file.")
        return

    logger.info("Initializing embedding function...")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=embedding_api_key,
        model_name=embedding_model,
    )

    if not os.path.exists(db_path):
        logger.error(
            "Database path '%s' does not exist. Please run 'python -m src.rag.indexing' first.",
            db_path,
        )
        return

    chroma_client = chromadb.PersistentClient(path=db_path)

    try:
        rules_collection = chroma_client.get_collection("ifab_rules", embedding_function=openai_ef)
        cases_collection = chroma_client.get_collection("history_cases", embedding_function=openai_ef)
    except Exception as e:
        logger.error("Error retrieving collections: %s", e)
        return

    logger.info("Initializing LLM client...")
    openai_client = OpenAI(
        api_key=llm_api_key,
        base_url=llm_base_url,
    )

    logger.info("Initializing orchestrator...")
    system = MultiAgentRefereeSystem(
        video_root_dir=project_root,
        rules_collection=rules_collection,
        cases_collection=cases_collection,
        openai_client=openai_client,
        model_name=llm_model_name,
    )

    dataset_path = dataset_rel_path
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(project_root, dataset_rel_path)

    logger.info("Loading dataset from: %s", dataset_path)
    if not os.path.exists(dataset_path):
        logger.error("Dataset not found at %s", dataset_path)
        return

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.error("Failed to parse dataset JSON.")
        return

    logger.info("Total questions to process: %d", len(data))

    try:
        for i, q in enumerate(data):
            system.process_question(q)

            if (i + 1) % 10 == 0:
                logger.info("Auto-saving progress at question %d...", i + 1)
                system.save_results_to_json(output_file)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Saving current progress...")

    system.get_final_report()
    system.save_results_to_json(output_file)
    logger.info("Done.")


if __name__ == "__main__":
    main()
