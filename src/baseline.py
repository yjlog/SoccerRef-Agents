"""
SoccerRef-Agents: Baseline Evaluation Script
---------------------------------------------
Evaluates Multi-modal Large Language Models (MLLMs) on the SoccerRefBench dataset.
Supports OpenAI, Anthropic, and Google Gemini models via native or compatible APIs.

Usage:
    python -m src.baseline

Environment variables required (see .env.example):
    OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL_NAME,
    VIDEO_DATASET_ROOT, DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_PATH
"""

import os
import json
import base64
import time
import re
import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

from utils.frame_extraction import extract_frames_from_video

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CURRENT_MODEL = os.environ.get("BASELINE_MODEL", "gpt-4o")

VIDEO_DATASET_ROOT = os.environ.get("VIDEO_DATASET_ROOT", "Dataset/video_dataset")
DEFAULT_DATASET_PATH = os.environ.get(
    "DEFAULT_DATASET_PATH",
    os.path.join(VIDEO_DATASET_ROOT, "Your_Dataset_File.json"),
)
DEFAULT_OUTPUT_PATH = os.environ.get(
    "DEFAULT_OUTPUT_PATH",
    "Results/model_performance_summary.json",
)
SUMMARY_FILE_PATH = os.environ.get(
    "SUMMARY_FILE_PATH",
    "model_performance_summary.json",
)

CONFIG: Dict[str, Dict[str, Any]] = {
    "gpt-4o": {
        "provider": "openai",
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "model_name": "gpt-4o",
        "base_url": os.environ.get("OPENAI_BASE_URL"),
    },
    "claude-4-5-sonnet": {
        "provider": "anthropic",
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "model_name": "claude-sonnet-4-5-20250929",
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "api_key": os.environ.get("GOOGLE_API_KEY", ""),
        "model_name": "gemini-2.5-flash",
    },
    "qwen3-vl-32b-instruct": {
        "provider": "openai_compatible",
        "api_key": os.environ.get("QWEN_API_KEY", ""),
        "base_url": os.environ.get("QWEN_BASE_URL"),
        "model_name": "qwen3-vl-32b-instruct",
    },
    "qwen3-vl-8b-instruct": {
        "provider": "openai_compatible",
        "api_key": os.environ.get("QWEN_API_KEY", ""),
        "base_url": os.environ.get("QWEN_BASE_URL"),
        "model_name": "qwen3-vl-8b-instruct",
    },
    "deepseek-v3": {
        "provider": "openai_compatible",
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": os.environ.get("DEEPSEEK_BASE_URL"),
        "model_name": "deepseek-v3",
        "vision_supported": False,
    },
}


class ModelDispatcher:
    """Handles API connections and generation requests for different model providers."""

    def __init__(self, model_key: str) -> None:
        if model_key not in CONFIG:
            raise ValueError(f"Model key '{model_key}' not found in configuration.")

        self.config = CONFIG[model_key]
        self.provider = self.config["provider"]
        self.model_name = self.config["model_name"]
        self.vision_supported = self.config.get("vision_supported", True)

        if self.provider in ("openai", "openai_compatible"):
            self.client = OpenAI(
                api_key=self.config["api_key"],
                base_url=self.config.get("base_url"),
            )
        elif self.provider == "anthropic":
            self.client = Anthropic(api_key=self.config["api_key"])
        elif self.provider == "google":
            from google import genai
            self.client = genai.Client(api_key=self.config["api_key"])

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        base64_frames: Optional[List[str]] = None,
    ) -> str:
        """Unified generation function for all providers."""
        if not self.vision_supported and base64_frames:
            logger.warning("This model does not support Vision via API. Using text only.")
            base64_frames = None

        try:
            if self.provider in ("openai", "openai_compatible"):
                return self._call_openai_style(system_prompt, user_prompt, base64_frames)
            elif self.provider == "anthropic":
                return self._call_anthropic(system_prompt, user_prompt, base64_frames)
            elif self.provider == "google":
                return self._call_google(system_prompt, user_prompt, base64_frames)
            else:
                return "Error: Unknown provider."
        except Exception as e:
            logger.error("API call failed: %s", e)
            return "Error: API call failed."

    def _call_openai_style(
        self,
        system_prompt: str,
        user_prompt: str,
        base64_frames: Optional[List[str]],
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        user_content: List[Any] = []
        if base64_frames:
            for b64 in base64_frames:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
        )
        if not response.choices:
            logger.warning("OpenAI: Empty response from API.")
            return "Error: API returned empty response."
        return response.choices[0].message.content or ""

    def _call_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        base64_frames: Optional[List[str]],
    ) -> str:
        messages_content: List[Any] = []
        if base64_frames:
            for b64 in base64_frames:
                messages_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                })
        messages_content.append({"type": "text", "text": user_prompt})

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=500,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": messages_content}],
        )
        return response.content[0].text

    def _call_google(
        self,
        system_prompt: str,
        user_prompt: str,
        base64_frames: Optional[List[str]],
    ) -> str:
        from google import genai
        from google.genai import types

        contents = [system_prompt + "\n\n" + user_prompt]

        if base64_frames:
            for b64 in base64_frames:
                image_bytes = base64.b64decode(b64)
                contents.append(
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=500,
                ),
            )
            return response.text
        except Exception as e:
            logger.error("Gemini API call failed: %s", e)
            return "Error: Gemini API call failed."


def run_evaluation(dataset_path: str, output_path: str) -> None:
    """
    Main evaluation loop: loads data, processes videos, queries models, and saves results.
    """
    logger.info("Initializing Model: %s...", CURRENT_MODEL)
    model_handler = ModelDispatcher(CURRENT_MODEL)

    logger.info("Loading dataset from %s...", dataset_path)
    if not os.path.exists(dataset_path):
        logger.error("Dataset file not found at %s", dataset_path)
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: List[Dict[str, Any]] = []
    total = len(data)

    for index, item in enumerate(data):
        logger.info("[%d/%d] Processing Question ID: %s...", index + 1, total, item.get("id", index))

        option_keys = [k for k in item.keys() if k.startswith("O") and k[1:].isdigit()]
        option_keys.sort(key=lambda x: int(x[1:]))

        if not option_keys:
            logger.warning("No options found, skipping.")
            continue

        options_text = "\n".join([f"{k}: {item[k]}" for k in option_keys])
        start_opt, end_opt = option_keys[0], option_keys[-1]

        materials = item.get("materials", [])
        video_path = None
        is_video_task = False

        if isinstance(materials, list) and len(materials) > 0:
            first_mat = materials[0]
            if isinstance(first_mat, dict) and "path" in first_mat:
                video_path = first_mat["path"]
                is_video_task = True
            elif isinstance(first_mat, str) and "path:" in first_mat:
                video_path = first_mat.replace("path:", "").strip()
                is_video_task = True
            elif first_mat == "none":
                is_video_task = False

        system_instruction = "You are a professional AI Soccer Referee Assistant."

        user_prompt = (
            f"QUESTION: {item['Q']}\n\n"
            f"OPTIONS:\n{options_text}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Analyze the input {'images/video' if is_video_task else 'text'} carefully.\n"
            f"2. Select the ONE correct option ID (from {start_opt} to {end_opt}).\n"
            f"3. Provide a brief explanation in English.\n\n"
            f"OUTPUT FORMAT:\n"
            f"Prediction: [Option ID]\n"
            f"Explanation: [Reasoning]"
        )

        video_frames: List[str] = []
        video_loaded_success = False

        if is_video_task and video_path:
            try:
                full_path = os.path.normpath(os.path.join(VIDEO_DATASET_ROOT, video_path))
                video_frames = extract_frames_from_video(full_path)
                if video_frames:
                    logger.info("Loaded %d video frames.", len(video_frames))
                    video_loaded_success = True
                else:
                    logger.warning("0 frames extracted.")
            except Exception as e:
                logger.error("Video load failed: %s", e)

        try:
            raw_response = model_handler.generate(
                system_instruction,
                user_prompt,
                video_frames if video_loaded_success else None,
            )

            raw_response = str(raw_response).strip()
            prediction = "Unknown"
            explanation = "No explanation."

            pred_match = re.search(r"Prediction:\s*[\*'\"]?(O\d+)[\*'\"]?", raw_response, re.IGNORECASE)
            if pred_match:
                prediction = pred_match.group(1).upper()
            else:
                for opt in reversed(option_keys):
                    if opt in raw_response:
                        prediction = opt
                        break

            expl_match = re.search(r"Explanation:\s*(.*)", raw_response, re.IGNORECASE | re.DOTALL)
            if expl_match:
                explanation = expl_match.group(1).strip()

            ground_truth = item.get("closeA", "N/A")
            is_correct = prediction == ground_truth

            logger.info("GT: %s | Pred: %s | %s", ground_truth, prediction, "CORRECT" if is_correct else "WRONG")

            results.append({
                "id": item.get("id", index),
                "model": CURRENT_MODEL,
                "type": "video" if video_loaded_success else "text",
                "question": item["Q"],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "explanation": explanation,
                "is_correct": is_correct,
                "raw_response": raw_response,
            })

        except Exception as e:
            logger.error("API error: %s", e)
            time.sleep(1)

    if not results:
        logger.info("No results to save.")
        return

    total_count = len(results)
    correct_count = sum(1 for r in results if r["is_correct"])
    acc = (correct_count / total_count) * 100

    summary_data = {
        "model": CURRENT_MODEL,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": total_count,
        "correct_answers": correct_count,
        "accuracy": f"{acc:.2f}%",
    }

    logger.info("Model: %s | Accuracy: %.2f%%", CURRENT_MODEL, acc)

    final_results = {
        "summary": summary_data,
        "details": results,
    }

    detail_output = output_path.replace(".json", f"_{CURRENT_MODEL}_detail.json")
    os.makedirs(os.path.dirname(detail_output) or ".", exist_ok=True)
    with open(detail_output, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    logger.info("Detailed results saved to %s", detail_output)

    all_summaries: List[Dict[str, Any]] = []

    if os.path.exists(SUMMARY_FILE_PATH):
        try:
            with open(SUMMARY_FILE_PATH, "r", encoding="utf-8") as f:
                all_summaries = json.load(f)
        except Exception:
            all_summaries = []

    existing_index = next(
        (i for i, entry in enumerate(all_summaries) if entry["model"] == CURRENT_MODEL),
        -1,
    )
    if existing_index > -1:
        all_summaries[existing_index] = summary_data
    else:
        all_summaries.append(summary_data)

    with open(SUMMARY_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    logger.info("Summary updated in %s", SUMMARY_FILE_PATH)


if __name__ == "__main__":
    output_dir = os.path.dirname(DEFAULT_OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_evaluation(DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_PATH)
