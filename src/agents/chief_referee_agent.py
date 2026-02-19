from typing import Dict, Any
from openai import OpenAI


class ChiefRefereeAgent:
    """
    The Chief Referee Agent aggregates outputs from subordinate agents (Rule, Case, Video, Context)
    and makes the final decision based on the combined evidence.
    """

    def __init__(self, openai_client: OpenAI):
        """
        Initializes the ChiefRefereeAgent.

        Args:
            openai_client (OpenAI): The OpenAI client instance.
        """
        self.client = openai_client

    def run(self, question_item: Dict[str, Any], mode: str, agent_outputs: Dict[str, Any]) -> str:
        """
        Executes the final decision-making process.

        Args:
            question_item (Dict): The dictionary containing the Question ('Q') and Options ('O1'...).
            mode (str): The operation mode, either 'text' or 'video'.
            agent_outputs (Dict): A dictionary containing outputs from 'rule', 'case',
                                  'video_analysis', and 'context' agents.

        Returns:
            str: The raw string response from the LLM (Prediction and Explanation).
        """

        # --- 1. Dynamic Option Parsing ---
        option_keys = [k for k in question_item.keys() if k.startswith('O') and k[1:].isdigit()]
        # Sort keys numerically (O1, O2, ..., O10)
        option_keys.sort(key=lambda x: int(x[1:]))

        if option_keys:
            options_text = "\n".join([f"{k}: {question_item[k]}" for k in option_keys])
            start_opt, end_opt = option_keys[0], option_keys[-1]
            range_str = f"({start_opt} to {end_opt})"
        else:
            options_text = "No specific options provided."
            range_str = ""

        # --- 2. Extract and Format Agent Outputs ---

        # [A] Rule Agent Section
        rule_data = agent_outputs.get('rule', {})
        rule_conf = rule_data.get('confidence', 'Low')

        if rule_conf != 'Low':
            rule_str = (
                f"   - TEXT OF THE LAW: \"{rule_data.get('direct_quote', 'N/A')}\"\n"
                f"   - Match Logic: {rule_data.get('key_terminology_match', 'N/A')}\n"
                f"   - Rule Agent Confidence: {rule_data.get('confidence', 'Unknown')}"
            )
        else:
            rule_str = "   [STATUS: LOW CONFIDENCE] The Rule Agent could not find a highly relevant specific clause. Rely on general football knowledge."

        # [B] Case Agent Section
        case_data = agent_outputs.get('case', {})

        if case_data.get('is_relevant', False):
            # If a relevant precedent is found, treat it as strong evidence
            case_str = (
                f"   [VALID PRECEDENT FOUND]\n"
                f"   - Similar Case Verdict: {case_data.get('historical_answer')}\n"
                f"   - Why it matches: {case_data.get('similarity_explanation')}\n"
                f"   - Key Differences to Watch: {case_data.get('key_difference')}\n"
            )
        else:
            case_str = (
                f"   [NO RELEVANT PRECEDENT]\n"
                f"   - The search returned cases that were structurally different.\n"
                f"   - IGNORE historical precedents for this decision.\n"
            )

        # --- 3. Construct Agent Reports Summary ---

        # Start with Rule Agent report
        input_summary = f"[1] reference law:\n{rule_str}\n\n"

        # NOTE: Historical precedents section is currently commented out in logic
        # input_summary += f"[2] reference precedents:\n{case_str}\n\n"

        # [C] Mode-Specific Processing (Video Mode)
        if mode == 'video':
            # Add Context Agent report
            input_summary += f"[3] reference context:\n   {agent_outputs.get('context', 'N/A')}\n\n"

            # Add Video Agent report
            vid_out = agent_outputs.get('video_analysis', {})

            if isinstance(vid_out, dict):
                desc = vid_out.get('choice_explanation', 'N/A')
                pred = vid_out.get('predicted_option', 'N/A')
                input_summary += f"[4] VISUAL EVIDENCE (Video Agent):\n"
                input_summary += f"   - Video Agent's Choice Explanation: {desc}\n"
                input_summary += f"   - Video Agent's Initial Intuition: {pred}\n"
            else:
                # Fallback if video output is raw string or error
                input_summary += f"[4] VISUAL EVIDENCE (Video Agent):\n   {str(vid_out)}\n"

        # --- 4. Construct Final Prompt ---
        final_prompt = (
            f"=== QUESTION DATA ===\n"
            f"Question: {question_item.get('Q', '')}\n"
            f"Options {range_str}:\n{options_text}\n\n"
            f"PS: In some questions, 'A#' means player of team A with jersey number #, same for 'B#'.\n\n"
            f"=== SUBORDINATE AGENT REPORTS ===\n"
            f"{input_summary}\n"
            f"=== INSTRUCTIONS ===\n"
            f"1. Analyze the input text carefully.\n"
            f"2. Select the most correct ONE option ID.\n"
            f"3. Provide a brief explanation in English. \n\n"
            f"OUTPUT FORMAT:\n"
            f"Prediction: [Option ID]\n"
            f"Explanation: [Reasoning]"
        )

        # --- 5. Call LLM for Final Decision ---
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional AI Soccer Referee Assistant."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"   [ChiefAgent] Error: {e}")
            return "Prediction: Error\nExplanation: API call failed."