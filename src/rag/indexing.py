"""
SoccerRef-Agents: Vector Database Indexing Script
--------------------------------------------------
Builds the ChromaDB vector indices for IFAB rules and historical cases.

Usage:
    python -m src.rag.indexing

Environment variables required (see .env.example):
    EMBEDDING_API_KEY, EMBEDDING_MODEL_NAME, DB_PATH,
    KNOWLEDGE_BASE_PDF_PATH, KNOWLEDGE_BASE_CASES_PATH
"""

import json
import os
import logging

import chromadb
from chromadb.utils import embedding_functions
import pypdf

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 50


def _get_embedding_function() -> embedding_functions.OpenAIEmbeddingFunction:
    api_key = os.environ.get("EMBEDDING_API_KEY")
    model_name = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

    if not api_key:
        raise ValueError(
            "EMBEDDING_API_KEY is not set. "
            "Please set it in your .env file or as an environment variable."
        )

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=model_name,
    )


def _get_chroma_client() -> chromadb.ClientAPI:
    db_path = os.environ.get("DB_PATH", "./referee_vector_db")
    return chromadb.PersistentClient(path=db_path)


def build_rules_db() -> None:
    """
    Reads the PDF rulebook and builds the 'ifab_rules' vector collection.

    Strategy:
        Splits the PDF by page.
        Stores metadata including the page number.
    """
    pdf_path = os.environ.get(
        "KNOWLEDGE_BASE_PDF_PATH",
        "KnowledgeBase/Laws of the Game 2025_26_single pages.pdf",
    )

    logger.info("Building rules index from: %s", pdf_path)

    if not os.path.exists(pdf_path):
        logger.error("PDF file not found at %s", pdf_path)
        return

    openai_ef = _get_embedding_function()
    client = _get_chroma_client()

    collection = client.get_or_create_collection(
        name="ifab_rules",
        embedding_function=openai_ef,
    )

    pdf_chunks: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            total_pages = len(reader.pages)

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    content = f"Page {i + 1}: {text}"
                    pdf_chunks.append(content)
                    metadatas.append({"source": "IFAB 2025", "page": i + 1})
                    ids.append(f"rule_page_{i + 1}")

        for i in range(0, len(pdf_chunks), BATCH_SIZE):
            collection.add(
                documents=pdf_chunks[i : i + BATCH_SIZE],
                metadatas=metadatas[i : i + BATCH_SIZE],
                ids=ids[i : i + BATCH_SIZE],
            )

        logger.info("Rules database built successfully. Total chunks: %d", len(pdf_chunks))

    except Exception as e:
        logger.error("Error processing PDF: %s", e)


def build_cases_db() -> None:
    """
    Reads the JSON case dataset and builds the 'history_cases' vector collection.

    Strategy:
        Formats the case description, decision, and controversiality into a single string.
        Stores the decision and ID in metadata.
    """
    cases_path = os.environ.get(
        "KNOWLEDGE_BASE_CASES_PATH",
        "KnowledgeBase/classic_case_knowledge.json",
    )

    logger.info("Building historical cases index from: %s", cases_path)

    if not os.path.exists(cases_path):
        logger.error("Cases file not found at %s", cases_path)
        return

    openai_ef = _get_embedding_function()
    client = _get_chroma_client()

    collection = client.get_or_create_collection(
        name="history_cases",
        embedding_function=openai_ef,
    )

    try:
        with open(cases_path, "r", encoding="utf-8") as f:
            cases = json.load(f)

        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for idx, case in enumerate(cases):
            text_content = (
                f"Case Description: {case.get('case', '')}\n"
                f"Result: {case.get('decision', '')}\n"
                f"Controversiality: {case.get('controversiality', '')}"
            )

            documents.append(text_content)
            metadatas.append({
                "result": case.get("decision", ""),
                "id": idx,
            })
            ids.append(f"case_{idx}")

        for i in range(0, len(documents), BATCH_SIZE):
            collection.add(
                documents=documents[i : i + BATCH_SIZE],
                metadatas=metadatas[i : i + BATCH_SIZE],
                ids=ids[i : i + BATCH_SIZE],
            )

        logger.info("Cases database built successfully. Total cases: %d", len(documents))

    except Exception as e:
        logger.error("Error processing cases JSON: %s", e)


if __name__ == "__main__":
    build_rules_db()
    build_cases_db()
