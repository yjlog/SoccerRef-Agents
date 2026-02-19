import json
from typing import Dict, Any


class RuleAgent:
    """
    An agent responsible for retrieving and analyzing specific regulations from the
    IFAB Laws of the Game based on a given scenario.
    """

    def __init__(self, rules_collection: Any, client: Any):
        """
        Initializes the RuleAgent.

        Args:
            rules_collection: The ChromaDB collection containing IFAB Laws.
            client: The OpenAI client instance for LLM interactions.
        """
        self.rules_collection = rules_collection
        self.client = client

    def search(self, query_text: str) -> Dict[str, Any]:
        """
        Retrieves relevant rules using RAG and uses an LLM to identify the precise applicable clause.

        Args:
            query_text (str): The textual description of the soccer scenario.

        Returns:
            Dict[str, Any]: A dictionary containing the direct quote, terminology match,
                            and confidence level.
        """
        # --- 1. RAG Retrieval ---
        try:
            results = self.rules_collection.query(
                query_texts=[query_text],
                n_results=2
            )

            # Concatenate retrieved documents with clear delimiters for the LLM
            documents = results.get('documents', [])
            if documents and len(documents) > 0 and len(documents[0]) > 0:
                # documents[0] is a list of strings (since we queried 1 text)
                retrieved_rules = "\n\n[Excerpt Start]\n" + \
                                  "\n[Excerpt End]\n\n[Excerpt Start]\n".join(documents[0]) + \
                                  "\n[Excerpt End]"
            else:
                retrieved_rules = "No specific rules found in database."

        except Exception as e:
            print(f"   [RuleAgent] DB Error: {e}")
            return {
                "direct_quote": "Error",
                "key_terminology_match": {},
                "confidence": "Low",
                "reasoning": "Database query failed."
            }

        # --- 2. LLM Analysis & Parsing ---
        system_prompt = (
            "You are an expert AI Legal Analyst specializing in the IFAB Laws of the Game. "
            "Your task is to strictly analyze the provided rule excerpts and identify the exact clause that governs the user's scenario. "
            "You function as a Librarian of Rules: do not make a final refereeing decision, but provide the precise legal text required to make that decision."
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
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0  # Low temperature for strict rule retrieval
            )

            content = response.choices[0].message.content
            rule_info = json.loads(content)

        except json.JSONDecodeError:
            print("   [RuleAgent] LLM returned invalid JSON.")
            rule_info = {
                "direct_quote": "Parsing Error",
                "key_terminology_match": {},
                "confidence": "Low"
            }
        except Exception as e:
            print(f"   [RuleAgent] LLM Parse Error: {e}")
            # Fallback return structure
            rule_info = {
                "direct_quote": "Unknown",
                "key_terminology_match": {},
                "confidence": "Unknown"
            }

        # Optional: Attach raw text for debugging or downstream verification
        # rule_info['raw_text'] = retrieved_rules

        return rule_info