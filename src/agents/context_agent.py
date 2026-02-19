import json
from typing import Dict, Any, Optional
from openai import OpenAI


class ContextAgent:
    """
    An agent responsible for analyzing match context (e.g., derby, final, home/away)
    to determine the appropriate refereeing strictness level.
    """

    def __init__(self, llm_client: OpenAI, model_name: str = "gpt-4o"):
        """
        Initializes the ContextAgent.

        Args:
            llm_client (OpenAI): The OpenAI client instance dedicated to LLM calls.
            model_name (str): The name of the model to use (default: "gpt-4o").
        """
        self.client = llm_client
        self.model_name = model_name

    def run(self, context_str: Optional[str]) -> Dict[str, Any]:
        """
        Analyzes the provided match context string.

        Args:
            context_str (str): A description of the match context (e.g., "Champions League Final").

        Returns:
            Dict[str, Any]: A dictionary containing the recommended strictness level and analysis.
        """
        # 1. Handle empty context
        if not context_str:
            return {
                "strictness": "Neutral",
                "analysis": "No context provided."
            }

        # 2. Construct Prompt
        prompt = (
            f"Match Context: {context_str}\n"
            f"Analyze the match importance (e.g., Derby, Final, League match), home/away factors, and potential team rivalries.\n"
            f"Determine the recommended refereeing strictness (Lenient, Normal, Strict).\n"
            f"Output JSON: {{'strictness': '...', 'analysis': '...'}}"
        )

        # 3. Call LLM API
        try:
            # Use self.client explicitly to avoid confusion with global variables
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            return json.loads(response.choices[0].message.content)

        except json.JSONDecodeError:
            print("   [ContextAgent] Error: LLM returned invalid JSON.")
            return {
                "strictness": "Neutral",
                "analysis": "Error parsing model response."
            }
        except Exception as e:
            print(f"   [ContextAgent] Error: {e}")
            return {
                "strictness": "Neutral",
                "analysis": "Error analyzing context."
            }