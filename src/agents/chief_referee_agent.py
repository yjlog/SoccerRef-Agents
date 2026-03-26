import logging
from typing import Dict, Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class ChiefRefereeAgent:
    """
    The Chief Referee Agent aggregates outputs from subordinate agents (Rule, Case, Video, Context)
    and makes the final decision based on the combined evidence.
    """

    def __init__(self, openai_client: OpenAI, model_name: str = "gpt-4o") -> None:
        """
        Initializes the ChiefRefereeAgent.

        Args:
            openai_client: The OpenAI client instance.
            model_name: The LLM model name to use for final adjudication.
        """
        self.client = openai_client
        self.model_name = model_name

    def run(self, question_item: Dict[str, Any], mode: str, agent_outputs: Dict[str, Any]) -> str:
        """
        Executes the final decision-making process.

        Args:
            question_item: The dictionary containing the Question ('Q') and Options ('O1'...).
            mode: The operation mode, either 'text' or 'video'.
            agent_outputs: A dictionary containing outputs from 'rule', 'case',
                           'video_analysis', and 'context' agents.

        Returns:
            The raw string response from the LLM (Prediction and Explanation).
        """
        option_keys = [k for k in question_item.keys() if k.startswith("O") and k[1:].isdigit()]
        option_keys.sort(key=lambda x: int(x[1:]))

        if option_keys:
            options_text = "\n".join([f"{k}: {question_item[k]}" for k in option_keys])
            start_opt, end_opt = option_keys[0], option_keys[-1]
            range_str = f"({start_opt} to {end_opt})"
        else:
            options_text = "No specific options provided."
            range_str = ""

        input_summary = self._build_input_summary(mode, agent_outputs)

        final_prompt = (
            f"=== QUESTION DATA ===\n"
            f"Question: {question_item.get('Q', '')}\n"
            f"Options {range_str}:\n{options_text}\n\n"
            f"PS: In some questions, 'A#' means player of team A with jersey number #, "
            f"same for 'B#'.\n\n"
            f"=== SUBORDINATE AGENT REPORTS ===\n"
            f"{input_summary}\n"
            f"=== INSTRUCTIONS ===\n"
            f"1. Analyze the input text carefully.\n"
            f"2. Select the most correct ONE option ID.\n"
            f"3. Provide a brief explanation in English.\n\n"
            f"OUTPUT FORMAT:\n"
            f"Prediction: [Option ID]\n"
            f"Explanation: [Reasoning]"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a professional AI Soccer Referee Assistant."},
                    {"role": "user", "content": final_prompt},
                ],
                temperature=0.0,
            )
            if not response.choices:
                logger.warning("ChiefAgent: Empty response from API.")
                return "Prediction: Error\nExplanation: API returned empty response."
            content = response.choices[0].message.content or ""
            return content

        except Exception as e:
            logger.error("ChiefAgent: Error during final adjudication: %s", e)
            return "Prediction: Error\nExplanation: API call failed."

    def _build_input_summary(self, mode: str, agent_outputs: Dict[str, Any]) -> str:
        """Constructs the agent reports summary section of the prompt."""

        rule_data = agent_outputs.get("rule", {})
        rule_conf = rule_data.get("confidence", "Low")

        if rule_conf != "Low":
            rule_str = (
                f"   - TEXT OF THE LAW: \"{rule_data.get('direct_quote', 'N/A')}\"\n"
                f"   - Match Logic: {rule_data.get('key_terminology_match', 'N/A')}\n"
                f"   - Rule Agent Confidence: {rule_data.get('confidence', 'Unknown')}"
            )
        else:
            rule_str = (
                "   [STATUS: LOW CONFIDENCE] The Rule Agent could not find a highly relevant "
                "specific clause. Rely on general football knowledge."
            )

        case_data = agent_outputs.get("case", {})

        if case_data.get("is_relevant", False):
            case_str = (
                f"   [VALID PRECEDENT FOUND]\n"
                f"   - Similar Case Verdict: {case_data.get('historical_answer')}\n"
                f"   - Why it matches: {case_data.get('similarity_explanation')}\n"
                f"   - Key Differences to Watch: {case_data.get('key_difference')}\n"
            )
        else:
            case_str = (
                "   [NO RELEVANT PRECEDENT]\n"
                "   - The search returned cases that were structurally different.\n"
                "   - IGNORE historical precedents for this decision.\n"
            )

        summary = f"[1] Reference Law:\n{rule_str}\n\n[2] Reference Precedents:\n{case_str}\n\n"

        if mode == "video":
            context_out = agent_outputs.get("context", "N/A")
            summary += f"[3] Reference Context:\n   {context_out}\n\n"

            vid_out = agent_outputs.get("video_analysis", {})
            if isinstance(vid_out, dict):
                desc = vid_out.get("choice_explanation", "N/A")
                pred = vid_out.get("predicted_option", "N/A")
                summary += (
                    f"[4] VISUAL EVIDENCE (Video Agent):\n"
                    f"   - Video Agent's Choice Explanation: {desc}\n"
                    f"   - Video Agent's Initial Intuition: {pred}\n"
                )
            else:
                summary += f"[4] VISUAL EVIDENCE (Video Agent):\n   {str(vid_out)}\n"

        return summary
