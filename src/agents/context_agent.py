import json
import logging
from typing import Dict, Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class ContextAgent:
    """
    An agent responsible for analyzing match context (e.g., derby, final, home/away)
    to determine the appropriate refereeing strictness level.
    """

    def __init__(self, llm_client: OpenAI, model_name: str = "gpt-4o") -> None:
        """
        Initializes the ContextAgent.

        Args:
            llm_client: The OpenAI client instance dedicated to LLM calls.
            model_name: The name of the model to use (default: "gpt-4o").
        """
        self.client = llm_client
        self.model_name = model_name

    def run(self, context_str: Optional[str]) -> Dict[str, Any]:
        """
        Analyzes the provided match context string.

        Args:
            context_str: A description of the match context (e.g., "Champions League Final").

        Returns:
            A dictionary containing the recommended strictness level and analysis.
        """
        if not context_str:
            return {
                "strictness": "Neutral",
                "analysis": "No context provided.",
            }

        prompt = (
            f"Match Context: {context_str}\n"
            f"Analyze the match importance (e.g., Derby, Final, League match), "
            f"home/away factors, and potential team rivalries.\n"
            f"Determine the recommended refereeing strictness (Lenient, Normal, Strict).\n"
            f"Output JSON: {{\"strictness\": \"...\", \"analysis\": \"...\"}}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            if not response.choices:
                logger.warning("ContextAgent: Empty response from API.")
                return {
                    "strictness": "Neutral",
                    "analysis": "API returned empty response.",
                }
            content = response.choices[0].message.content
            if not content:
                logger.warning("ContextAgent: Empty content from API.")
                return {
                    "strictness": "Neutral",
                    "analysis": "API returned empty content.",
                }
            return json.loads(content)

        except json.JSONDecodeError:
            logger.warning("ContextAgent: LLM returned invalid JSON.")
            return {
                "strictness": "Neutral",
                "analysis": "Error parsing model response.",
            }
        except Exception as e:
            logger.error("ContextAgent: Error analyzing context: %s", e)
            return {
                "strictness": "Neutral",
                "analysis": "Error analyzing context.",
            }
