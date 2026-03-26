import json
import logging
from typing import Dict, Any, Optional

from chromadb import Collection

logger = logging.getLogger(__name__)


class RuleAgent:
    """
    An agent responsible for retrieving and analyzing specific regulations from the
    IFAB Laws of the Game based on a given scenario.
    """

    N_RESULTS = 2

    def __init__(self, rules_collection: Collection, client: Any, model_name: str = "gpt-4o") -> None:
        """
        Initializes the RuleAgent.

        Args:
            rules_collection: The ChromaDB collection containing IFAB Laws.
            client: The OpenAI client instance for LLM interactions.
            model_name: The LLM model name to use for analysis.
        """
        self.rules_collection = rules_collection
        self.client = client
        self.model_name = model_name

    def search(self, query_text: str) -> Dict[str, Any]:
        """
        Retrieves relevant rules using RAG and uses an LLM to identify the precise applicable clause.

        Args:
            query_text: The textual description of the soccer scenario.

        Returns:
            A dictionary containing the direct quote, terminology match, and confidence level.
        """
        logger.info("RuleAgent: Searching rules for: %s...", query_text[:50])

        retrieved_rules = self._rag_retrieve(query_text)
        if retrieved_rules is None:
            return {
                "direct_quote": "Error",
                "key_terminology_match": {},
                "confidence": "Low",
                "reasoning": "Database query failed.",
            }

        return self._llm_analyze(query_text, retrieved_rules)

    def _rag_retrieve(self, query_text: str) -> Optional[str]:
        try:
            results = self.rules_collection.query(
                query_texts=[query_text],
                n_results=self.N_RESULTS,
            )

            documents = results.get("documents", [])
            if documents and len(documents) > 0 and len(documents[0]) > 0:
                return (
                    "\n\n[Excerpt Start]\n"
                    + "\n[Excerpt End]\n\n[Excerpt Start]\n".join(documents[0])
                    + "\n[Excerpt End]"
                )
            return "No specific rules found in database."
        except Exception as e:
            logger.error("RuleAgent: DB query error: %s", e)
            return None

    def _llm_analyze(self, query_text: str, retrieved_rules: str) -> Dict[str, Any]:
        system_prompt = (
            "You are an expert AI Legal Analyst specializing in the IFAB Laws of the Game. "
            "Your task is to strictly analyze the provided rule excerpts and identify the exact clause "
            "that governs the user's scenario. "
            "You function as a Librarian of Rules: do not make a final refereeing decision, "
            "but provide the precise legal text required to make that decision."
        )

        user_prompt = (
            f"### CONTEXT (Retrieved IFAB Laws) ###\n{retrieved_rules}\n\n"
            f"### SCENARIO ###\n\"{query_text}\"\n\n"
            f"### INSTRUCTIONS ###\n"
            f"1. Analyze the 'Context' to find the specific Law, Section, and Bullet Point that applies to the 'Scenario'.\n"
            f"2. You MUST prioritize specific offenses (e.g., 'Serious Foul Play', 'DOGSO') over general rules.\n"
            f"3. If the retrieved context contains the exact rule needed, extract it verbatim.\n"
            f"4. If the retrieved context is irrelevant to the scenario, state 'Irrelevant' in the fields.\n\n"
            f"### OUTPUT FORMAT (JSON ONLY) ###\n"
            f"Provide a valid JSON object with the following keys:\n"
            f"- 'direct_quote': The EXACT sentence or phrase from the retrieved text that applies. Do not paraphrase.\n"
            f"- 'key_terminology_match': A dictionary mapping terms from the scenario to terms in the rule.\n"
            f"- 'confidence': 'High' (Exact match), 'Medium' (Related concept), or 'Low' (Irrelevant text)."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            if not response.choices:
                logger.warning("RuleAgent: Empty response from API.")
                return {
                    "direct_quote": "Error",
                    "key_terminology_match": {},
                    "confidence": "Low",
                }
            raw_content = response.choices[0].message.content
            if not raw_content:
                logger.warning("RuleAgent: Empty content from API.")
                return {
                    "direct_quote": "Error",
                    "key_terminology_match": {},
                    "confidence": "Low",
                }
            return json.loads(raw_content)

        except json.JSONDecodeError:
            logger.warning("RuleAgent: LLM returned invalid JSON.")
            return {
                "direct_quote": "Parsing Error",
                "key_terminology_match": {},
                "confidence": "Low",
            }
        except Exception as e:
            logger.error("RuleAgent: LLM call error: %s", e)
            return {
                "direct_quote": "Unknown",
                "key_terminology_match": {},
                "confidence": "Low",
            }
