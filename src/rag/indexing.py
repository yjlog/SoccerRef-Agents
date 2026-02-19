import json
import os

import chromadb
from chromadb.utils import embedding_functions
import PyPDF2

# ================= Configuration =================

API_KEY = "your_openai_api_key_here"

# File Paths
# Using raw strings or forward slashes for Windows paths
PDF_PATH = "your_path_to_pdf/IFAB_Laws_of_the_Game_2025.pdf"
CASES_PATH = "your_path_to_cases/classic_case_knowledge.json"
DB_PATH = "./referee_vector_db"

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# ================= Initialization =================

# Initialize Embedding Function
# Using OpenAI's embedding model for high quality and cost-effectiveness
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name=EMBEDDING_MODEL_NAME
)

# Initialize ChromaDB Client
client = chromadb.PersistentClient(path=DB_PATH)


# ================= Core Functions =================

def build_rules_db():
    """
    Reads the PDF rulebook and builds the 'ifab_rules' vector collection.

    Strategy:
        Splits the PDF by page (or approx. 500 chars).
        Stores metadata including the page number.
    """
    print("Building rules index...")

    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        return

    # Get or create the collection
    collection = client.get_or_create_collection(
        name="ifab_rules",
        embedding_function=openai_ef
    )

    pdf_chunks = []
    metadatas = []
    ids = []

    try:
        with open(PDF_PATH, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Current Strategy: Simple page-based chunking
                    # Recommendation for future: Split by "Law X" using Regex or overlapping windows
                    content = f"Page {i + 1}: {text}"

                    pdf_chunks.append(content)
                    metadatas.append({"source": "IFAB 2025", "page": i + 1})
                    ids.append(f"rule_page_{i + 1}")

        # Batch processing to avoid payload limits
        batch_size = 50
        for i in range(0, len(pdf_chunks), batch_size):
            collection.add(
                documents=pdf_chunks[i: i + batch_size],
                metadatas=metadatas[i: i + batch_size],
                ids=ids[i: i + batch_size]
            )

        print(f"Rules database built successfully. Total chunks: {len(pdf_chunks)}")

    except Exception as e:
        print(f"Error processing PDF: {e}")


def build_cases_db():
    """
    Reads the JSON case dataset and builds the 'history_cases' vector collection.

    Strategy:
        Formats the case description, decision, and controversiality into a single string.
        Stores the decision and ID in metadata.
    """
    print("Building historical cases index...")

    if not os.path.exists(CASES_PATH):
        print(f"Error: Cases file not found at {CASES_PATH}")
        return

    # Get or create the collection
    collection = client.get_or_create_collection(
        name="history_cases",
        embedding_function=openai_ef
    )

    try:
        with open(CASES_PATH, 'r', encoding='utf-8') as f:
            cases = json.load(f)

        documents = []
        metadatas = []
        ids = []

        for idx, case in enumerate(cases):
            # Convert structured case data into a descriptive string for embedding
            text_content = (
                f"Case Description: {case.get('case', '')}\n"
                f"Result: {case.get('decision', '')}\n"
                f"Controversiality: {case.get('controversiality', '')}"
            )

            documents.append(text_content)

            # Storing key info in metadata for quick retrieval without re-querying JSON
            metadatas.append({
                "result": case.get('decision', ''),
                "id": idx
            })
            ids.append(f"case_{idx}")

        # Batch processing
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[i: i + batch_size],
                metadatas=metadatas[i: i + batch_size],
                ids=ids[i: i + batch_size]
            )

        print(f"Cases database built successfully. Total cases: {len(documents)}")

    except Exception as e:
        print(f"Error processing cases JSON: {e}")


if __name__ == "__main__":
    # Run once to initialize the vector database
    build_rules_db()
    build_cases_db()