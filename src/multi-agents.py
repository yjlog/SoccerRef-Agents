import json
import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

from agents.orchestrator import MultiAgentRefereeSystem

# ================= Configuration =================

CONFIG = {
    # Path Configuration
    "project_root": "Your/Path/To/Project",  # Update this to your actual project root path
    "db_path": "./referee_vector_db",
    "dataset_rel_path": "Dataset/text/text.json",  # Relative to project root or absolute
    "output_file": "output.json",

    # Embedding Configuration (For Vector DB)
    "embedding_api_key": "${input:your-token-key}",
    "embedding_model": "text-embedding-3-small",

    # LLM Configuration (For Agents)
    "llm_api_key": "${input:your-token-key}",
    "llm_base_url": "${input:your-token-key}"
}


# ================= Main Execution =================

def main():
    print("--- Starting Multi-Agent Referee System ---")

    # 1. Initialize OpenAI Embedding Function (For ChromaDB)
    # This uses the specific key provided for embeddings
    print("Initializing Embedding Function...")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=CONFIG["embedding_api_key"],
        model_name=CONFIG["embedding_model"]
    )

    # 2. Initialize ChromaDB Client & Retrieve Collections
    print(f"Connecting to Vector DB at {CONFIG['db_path']}...")
    if not os.path.exists(CONFIG['db_path']):
        print(f"Error: Database path '{CONFIG['db_path']}' does not exist. Please run 'build_vector_db.py' first.")
        return

    chroma_client = chromadb.PersistentClient(path=CONFIG["db_path"])

    try:
        rules_collection = chroma_client.get_collection("ifab_rules", embedding_function=openai_ef)
        cases_collection = chroma_client.get_collection("history_cases", embedding_function=openai_ef)
    except Exception as e:
        print(f"Error retrieving collections: {e}")
        return

    # 3. Initialize OpenAI Client (For LLM Agents)
    # This uses the relay service configuration
    print("Initializing LLM Client...")
    openai_client = OpenAI(
        api_key=CONFIG["llm_api_key"],
        base_url=CONFIG["llm_base_url"]
    )

    # 4. Initialize the Multi-Agent System
    print("Initializing Orchestrator...")
    system = MultiAgentRefereeSystem(
        video_root_dir=CONFIG["project_root"],
        rules_collection=rules_collection,
        cases_collection=cases_collection,
        openai_client=openai_client
    )

    # 5. Load Dataset
    dataset_path = CONFIG["dataset_rel_path"]
    # Handle absolute path fallback if relative path logic fails
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(CONFIG["project_root"], CONFIG["dataset_rel_path"])

    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total questions to process: {len(data)}")

    # 6. Execution Loop
    try:
        for i, q in enumerate(data):
            # Optional: Add a limit for testing, e.g., if i >= 5: break
            system.process_question(q)

            # Optional: Save intermediate results every 10 questions
            if (i + 1) % 10 == 0:
                print(f"   [Auto-Save] Saving progress at question {i + 1}...")
                system.save_results_to_json(CONFIG["output_file"])

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")

    # 7. Final Report & Save
    system.get_final_report()
    system.save_results_to_json(CONFIG["output_file"])
    print("Done.")


if __name__ == "__main__":
    main()