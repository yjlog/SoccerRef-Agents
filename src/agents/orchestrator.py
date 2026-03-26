import json
import re
import os
import logging
from typing import Dict, Any, List, Optional, Tuple

from chromadb import Collection

from .rule_agent import RuleAgent
from .case_agent import CaseAgent
from .context_agent import ContextAgent
from .video_agent import VideoAgent
from .chief_referee_agent import ChiefRefereeAgent

logger = logging.getLogger(__name__)


class MultiAgentRefereeSystem:
    """
    The central orchestrator that coordinates the Rule, Case, Context, Video,
    and Chief Referee agents to solve soccer refereeing problems.
    """

    def __init__(
        self,
        video_root_dir: str,
        rules_collection: Collection,
        cases_collection: Collection,
        openai_client: Any,
        model_name: str = "gpt-4o",
    ) -> None:
        """
        Initializes the Multi-Agent System.

        Args:
            video_root_dir: Root directory containing video files.
            rules_collection: ChromaDB collection for IFAB rules.
            cases_collection: ChromaDB collection for historical cases.
            openai_client: Shared OpenAI client instance.
            model_name: LLM model name shared across agents.
        """
        self.video_root = video_root_dir
        self.openai_client = openai_client
        self.model_name = model_name

        self.total_processed = 0
        self.correct_count = 0
        self.results_log: List[Dict[str, Any]] = []

        logger.info("Initializing agents...")

        self.rule_agent = RuleAgent(rules_collection, openai_client, model_name=model_name)
        self.case_agent = CaseAgent(cases_collection, openai_client, model_name=model_name)
        self.context_agent = ContextAgent(openai_client, model_name=model_name)
        self.video_agent = VideoAgent(openai_client, model_name=model_name)
        self.chief_agent = ChiefRefereeAgent(openai_client, model_name=model_name)

        logger.info("All agents ready.")

    def _detect_mode(self, question_item: Dict[str, Any]) -> Tuple[str, Optional[str], str]:
        """
        Helper method: Detects the question mode (text vs video) and extracts paths.
        """
        materials = question_item.get("materials", [])
        mode = "text"
        video_path = None
        context_str = ""

        if materials and isinstance(materials, list):
            mat = materials[0]
            if isinstance(mat, dict) and "path" in mat:
                mode = "video"
                video_path = mat["path"]
                context_str = mat.get("context", "")
            elif isinstance(mat, str) and "path:" in mat:
                mode = "video"
                video_path = mat.replace("path:", "").strip()

        return mode, video_path, context_str

    def _parse_prediction(self, text: str) -> str:
        """Extracts the predicted option ID (e.g., O1, O2) from the LLM response."""
        if not text:
            return "Unknown"

        match = re.search(r"(?:Final Answer|Prediction|Decision)[:\s\-\*]*([O]\d+)", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

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
        logger.info("[%d] Processing question...", q_id)

        agent_traces: Dict[str, Any] = {}

        mode, video_rel_path, context_str = self._detect_mode(question_item)
        full_video_path = None
        if video_rel_path:
            full_video_path = os.path.join(self.video_root, video_rel_path)

        agent_traces["mode"] = mode
        logger.info("[Step 0] Detected mode: %s", mode.upper())

        search_query = question_item.get("Q", "")
        video_analysis: Optional[Dict[str, Any]] = None

        if mode == "video" and full_video_path:
            logger.info("[Step 1] Video Agent: Analyzing video...")
            try:
                video_analysis = self.video_agent.run(question_item, full_video_path)
                visual_desc = video_analysis.get("choice_explanation", "N/A")
                search_query = f"{question_item.get('Q', '')}\nVisual Context: {visual_desc}"
                agent_traces["video_analysis"] = video_analysis
            except Exception as e:
                logger.warning("Video description failed (%s), using text query only.", e)
                search_query = question_item.get("Q", "")
                video_analysis = None
        else:
            logger.info("[Step 1] Text mode: Using original question for RAG...")

        logger.info("[Step 2] Retrieving knowledge (Rules & Cases)...")

        rule_info = self.rule_agent.search(search_query)
        agent_traces["rule_output"] = rule_info

        case_output = self.case_agent.search(search_query)
        agent_traces["case_output"] = case_output

        agent_outputs_payload: Dict[str, Any] = {
            "rule": rule_info,
            "case": case_output,
        }

        if mode == "video":
            logger.info("[Step 3] Video branch: Running context analysis...")
            context_analysis = self.context_agent.run(context_str)
            agent_outputs_payload["context"] = context_analysis
            agent_traces["context_output"] = context_analysis

            if video_analysis is not None:
                agent_outputs_payload["video_analysis"] = video_analysis
        else:
            logger.info("[Step 3] Text branch: Relying on Rule/Case logic...")

        logger.info("[Step 4] Chief Referee: Deliberating...")
        final_verdict = self.chief_agent.run(question_item, mode, agent_outputs_payload)

        ground_truth = question_item.get("closeA", "Unknown")
        prediction = self._parse_prediction(final_verdict)
        explanation = self._extract_explanation(final_verdict)

        is_correct = prediction.strip() == ground_truth.strip()
        if is_correct:
            self.correct_count += 1

        log_entry = {
            "id": q_id,
            "question": question_item.get("Q", ""),
            "mode": mode,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "chief_explanation": explanation,
            "agent_traces": agent_traces,
        }

        self.results_log.append(log_entry)
        logger.info(
            "Decision: %s (GT: %s) | %s",
            prediction, ground_truth, "CORRECT" if is_correct else "WRONG",
        )

        return final_verdict

    def save_results_to_json(self, output_path: str = "evaluation_results.json") -> None:
        """Saves the detailed evaluation report to a JSON file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.results_log, f, ensure_ascii=False, indent=4)
            logger.info("Detailed report saved to: %s", os.path.abspath(output_path))
        except Exception as e:
            logger.error("Failed to save JSON: %s", e)

    def get_final_report(self) -> float:
        """Calculates and logs the final accuracy."""
        if self.total_processed == 0:
            return 0.0

        accuracy = (self.correct_count / self.total_processed) * 100
        logger.info("=" * 30)
        logger.info("FINAL ACCURACY: %.2f%%", accuracy)
        logger.info("=" * 30)
        return accuracy
