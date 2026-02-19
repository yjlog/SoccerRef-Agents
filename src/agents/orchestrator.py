import json
import re
import os
from typing import Dict, Any


from rule_agent import RuleAgent
from case_agent import CaseAgent
from context_agent import ContextAgent
from video_agent import VideoAgent
from chief_referee_agent import ChiefRefereeAgent

class MultiAgentRefereeSystem:
    """
    The central orchestrator that coordinates the Rule, Case, Context, Video,
    and Chief Referee agents to solve soccer refereeing problems.
    """

    def __init__(self, video_root_dir: str, rules_collection: Any, cases_collection: Any, openai_client: Any):
        """
        Initializes the Multi-Agent System.

        Args:
            video_root_dir (str): Root directory containing video files.
            rules_collection (Any): ChromaDB collection for IFAB rules.
            cases_collection (Any): ChromaDB collection for historical cases.
            openai_client (Any): Shared OpenAI client instance.
        """
        self.video_root = video_root_dir
        self.openai_client = openai_client

        # Statistical variables
        self.total_processed = 0
        self.correct_count = 0
        self.results_log = []

        print("Initializing Agents...")

        # Dependency Injection: Pass DB collections and clients to respective agents
        self.rule_agent = RuleAgent(rules_collection, openai_client)
        self.case_agent = CaseAgent(cases_collection, openai_client)
        self.context_agent = ContextAgent(openai_client)
        self.video_agent = VideoAgent(openai_client)
        self.chief_agent = ChiefRefereeAgent(openai_client)

        print("Agents Ready.")

    def _detect_mode(self, question_item: Dict[str, Any]) -> tuple:
        """
        Helper method: Detects the question mode (text vs video) and extracts paths.
        """
        materials = question_item.get('materials', [])
        mode = "text"
        video_path = None
        context_str = ""

        if materials and isinstance(materials, list):
            mat = materials[0]
            if isinstance(mat, dict) and "path" in mat:
                mode = "video"
                video_path = mat['path']
                context_str = mat.get('context', "")
            elif isinstance(mat, str) and "path:" in mat:
                mode = "video"
                video_path = mat.replace("path:", "").strip()

        return mode, video_path, context_str

    def _parse_prediction(self, text: str) -> str:
        """
        Extracts the predicted option ID (e.g., O1, O2) from the LLM response.
        """
        if not text:
            return "Unknown"

        # Try strict matching first
        match = re.search(r"(?:Final Answer|Prediction|Decision)[:\s\-\*]*([O]\d+)", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Fallback to loose matching
        match_loose = re.search(r"\b(O\d+)\b", text)
        return match_loose.group(1).upper() if match_loose else "Unknown"

    def _extract_explanation(self, text: str) -> str:
        """Extracts the explanation text."""
        return text.strip() if text else "No explanation provided."

    def process_question(self, question_item: Dict[str, Any]) -> str:
        """
        Main execution flow for a single question.
        """
        self.total_processed += 1
        q_id = self.total_processed
        print(f"\n--- [{q_id}] Processing Question... ---")

        # Container for agent execution traces (for debugging/logging)
        agent_traces = {}

        # --- Step 0: Mode Detection ---
        mode, video_rel_path, context_str = self._detect_mode(question_item)
        full_video_path = None
        if video_rel_path:
            full_video_path = os.path.join(self.video_root, video_rel_path)

        agent_traces['mode'] = mode
        print(f"[0] Detected Mode: {mode.upper()}")

        # --- Step 1: Query Construction (Look -> Search) ---
        search_query = question_item.get('Q', '')
        visual_desc = ""

        if mode == 'video' and full_video_path:
            print("[1] Video Agent: Generating description for RAG...")
            # Run VideoAgent primarily to get a description for the RAG search
            try:
                # Note: VideoAgent.run returns a dict with 'choice_explanation'
                video_out = self.video_agent.run(question_item, full_video_path)
                visual_desc = video_out.get('choice_explanation', 'N/A')
            except Exception as e:
                print(f"    Warning: Video description failed ({e}), using text query only.")
                visual_desc = "Video processing failed."

            agent_traces['visual_description_rag'] = visual_desc
            # Combine Question + Visual Description for better retrieval
            search_query = f"{question_item.get('Q', '')}\nVisual Context: {visual_desc}"
        else:
            print("[1] Text Mode: Using original question for RAG...")

        # --- Step 2: Parallel Retrieval (Rules & Cases) ---
        print(f"[2] Retrieving Knowledge (Rules & Cases)...")

        # Rule Agent Execution
        rule_info = self.rule_agent.search(search_query)
        agent_traces['rule_output'] = rule_info

        # Case Agent Execution
        case_output = self.case_agent.search(search_query)
        agent_traces['case_output'] = case_output

        # --- Step 3: Deep Analysis (Analyze) ---
        # Prepare payload for Chief Referee
        agent_outputs_payload = {
            "rule": rule_info,
            "case": case_output
        }

        if mode == 'video' and full_video_path:
            print("[3] Video Branch: Running Context & Video Analysis...")

            # Context Agent Execution
            context_analysis = self.context_agent.run(context_str)
            agent_outputs_payload['context'] = context_analysis
            agent_traces['context_output'] = context_analysis

            # Video Agent Execution (Second pass for decision making)
            video_analysis = self.video_agent.run(question_item, full_video_path)
            agent_outputs_payload['video_analysis'] = video_analysis
            agent_traces['video_analysis_output'] = video_analysis

        elif mode == 'text':
            print("[3] Text Branch: Relying on Rule/Case logic...")
            # Text specific logic is handled by Rule/Case agents

        # --- Step 4: Final Verdict (Chief Referee) ---
        print("[4] Chief Referee: Deliberating...")
        final_verdict = self.chief_agent.run(question_item, mode, agent_outputs_payload)

        # --- Step 5: Scoring & Logging ---
        ground_truth = question_item.get('closeA', 'Unknown')
        prediction = self._parse_prediction(final_verdict)
        explanation = self._extract_explanation(final_verdict)

        # Determine correctness (ignoring whitespace)
        is_correct = (prediction.strip() == ground_truth.strip())
        if is_correct:
            self.correct_count += 1

        # Construct log entry
        log_entry = {
            "id": q_id,
            "question": question_item.get('Q', ''),
            "mode": mode,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "chief_explanation": explanation,
            "agent_traces": agent_traces  # Contains full traces from all agents
        }

        self.results_log.append(log_entry)

        print(f"=== DECISION: {prediction} (GT: {ground_truth}) | {'✅' if is_correct else '❌'} ===")
        return final_verdict

    def save_results_to_json(self, output_path: str = "evaluation_results.json"):
        """Saves the detailed evaluation report to a JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results_log, f, ensure_ascii=False, indent=4)
            print(f"\n[Success] Detailed report saved to: {os.path.abspath(output_path)}")
        except Exception as e:
            print(f"\n[Error] Failed to save JSON: {e}")

    def get_final_report(self) -> float:
        """Calculates and prints the final accuracy."""
        if self.total_processed == 0:
            return 0.0

        accuracy = (self.correct_count / self.total_processed) * 100
        print(f"\n{'=' * 30}\nFINAL ACCURACY: {accuracy:.2f}%\n{'=' * 30}")
        return accuracy