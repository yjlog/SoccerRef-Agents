"""
SoccerRef-Agents Baseline Evaluation Script
-------------------------------------------
This script evaluates Multi-modal Large Language Models (MLLMs) on the Soccer Referee dataset.
It supports OpenAI, Anthropic, and Google Gemini models via native or compatible APIs.

Usage:
    Ensure your API keys are set in the CONFIG dictionary.
    Run the script to process the dataset and generate evaluation reports.
"""

import os
import json
import base64
import time
import re
import cv2
from typing import List, Dict, Any, Optional

# Third-party imports
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

# ================= Configuration =================

# 1. Select the current model to evaluate
CURRENT_MODEL = "gpt-4o"

# 2. File Paths
# NOTE: Update these paths if moving the script to a different environment
VIDEO_DATASET_ROOT = "Your/Path/To/SoccerRef-Agents-GithubVersion/Dataset/video_dataset"
DEFAULT_DATASET_PATH = os.path.join(VIDEO_DATASET_ROOT, "Your_Dataset_File.json")  # Update with actual dataset filename
DEFAULT_OUTPUT_PATH = "Your/Path/To/SoccerRef-Agents-GithubVersion/Results/model_performance_summary.json"  # Update with desired output path
SUMMARY_FILE_PATH = "Test_model_performance_summary.json"

# 3. Video Processing Settings
SECONDS_PER_FRAME = 1  # Extract one frame every X seconds
MAX_FRAMES = 15  # Maximum number of frames to feed into the model

# 4. API Configuration
CONFIG = {
    "gpt-4o": {
        "provider": "openai",
        "api_key": "${input:your-token-key}",
        "model_name": "gpt-4o",
        "base_url": "${input:your-token-key}"
    },
    "claude-4-5-sonnet": {
        "provider": "openai_compatible",
        "api_key": "${input:your-token-key}",
        "base_url": "${input:your-token-key}",
        "model_name": "claude-sonnet-4-5-20250929"
    },
    "gemini-2.5-flash": {
        "provider": "openai_compatible",
        "api_key": "${input:your-token-key}",
        "base_url": "${input:your-token-key}",
        "model_name": "gemini-2.5-flash"
    },
    "qwen3-vl-32b-instruct": {
        "provider": "openai_compatible",
        "api_key": "${input:your-token-key}",
        "base_url": "${input:your-token-key}",
        "model_name": "qwen3-vl-32b-instruct"
    },
    "qwen3-vl-8b-instruct": {
        "provider": "openai_compatible",
        "api_key": "${input:your-token-key}",
        "base_url": "${input:your-token-key}",
        "model_name": "qwen3-vl-8b-instruct"
    },
    "deepseek-v3": {
        "provider": "openai_compatible",
        "api_key": "${input:your-token-key}",
        "base_url": "${input:your-token-key}",
        "model_name": "deepseek-v3",
        "vision_supported": False  # DeepSeek API is currently text-only
    },
    # Add more models as needed, following the same structure
}


# ================= Helper Functions =================

def encode_image_base64(frame: Any) -> str:
    """Encodes an OpenCV image frame to a Base64 string."""
    # Compress properly to speed up transmission and reduce Token consumption
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    _, buffer = cv2.imencode(".jpg", frame, encode_param)
    return base64.b64encode(buffer).decode("utf-8")


def extract_frames_from_video(rel_path: str) -> List[str]:
    """
    Extracts frames from a video file at a specified interval, resizes them,
    and converts them to Base64 strings.
    """
    full_path = os.path.join(VIDEO_DATASET_ROOT, rel_path)
    full_path = os.path.normpath(full_path)

    if not os.path.exists(full_path):
        print(f"   [Error] Video file not found: {full_path}")
        return []

    video = cv2.VideoCapture(full_path)
    base64_frames = []
    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25

    frame_interval = int(fps * SECONDS_PER_FRAME)
    if frame_interval < 1:
        frame_interval = 1

    curr_frame = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        if curr_frame % frame_interval == 0:
            # Resize: Large resolution frames slow down API processing
            # Resize long edge to 720 (or 512)
            h, w = frame.shape[:2]
            scale = 720 / max(h, w)
            if scale < 1:
                new_size = (int(w * scale), int(h * scale))
                frame = cv2.resize(frame, new_size)

            base64_frames.append(encode_image_base64(frame))
        curr_frame += 1

    video.release()

    # Uniform sampling to fit context window if frames exceed limit
    if len(base64_frames) > MAX_FRAMES:
        step = len(base64_frames) / MAX_FRAMES
        indices = [int(i * step) for i in range(MAX_FRAMES)]
        base64_frames = [base64_frames[i] for i in indices]

    return base64_frames


# ================= Model Handler Class =================

class ModelDispatcher:
    """
    Handles API connections and generation requests for different model providers.
    """

    def __init__(self, model_key: str):
        if model_key not in CONFIG:
            raise ValueError(f"Model key '{model_key}' not found in configuration.")

        self.config = CONFIG[model_key]
        self.provider = self.config["provider"]
        self.model_name = self.config["model_name"]
        self.vision_supported = self.config.get("vision_supported", True)

        # Initialize Client based on provider
        if self.provider == "openai" or self.provider == "openai_compatible":
            self.client = OpenAI(
                api_key=self.config["api_key"],
                base_url=self.config.get("base_url")
            )
        elif self.provider == "anthropic":
            self.client = Anthropic(api_key=self.config["api_key"])
        elif self.provider == "google":
            self.client = genai.Client(api_key=self.config["api_key"])

    def generate(self, system_prompt: str, user_prompt: str, base64_frames: Optional[List[str]] = None) -> str:
        """Unified generation function for all providers."""

        # If model does not support vision, force clear images
        if not self.vision_supported and base64_frames:
            print("   [Warn] This model does not support Vision via API. Using text only.")
            base64_frames = None

        try:
            if self.provider in ["openai", "openai_compatible"]:
                return self._call_openai_style(system_prompt, user_prompt, base64_frames)
            elif self.provider == "anthropic":
                return self._call_anthropic(system_prompt, user_prompt, base64_frames)
            elif self.provider == "google":
                return self._call_google(system_prompt, user_prompt, base64_frames)
            else:
                return "Error: Unknown provider."
        except Exception as e:
            return f"Error: {str(e)}"

    def _call_openai_style(self, system_prompt: str, user_prompt: str, base64_frames: Optional[List[str]]) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        user_content = []
        # Ensure correct prefix for Qwen-VL etc.
        if base64_frames:
            for b64 in base64_frames:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })

        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=500
        )
        return response.choices[0].message.content

    def _call_anthropic(self, system_prompt: str, user_prompt: str, base64_frames: Optional[List[str]]) -> str:
        messages_content = []
        if base64_frames:
            for b64 in base64_frames:
                messages_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64
                    }
                })
        messages_content.append({"type": "text", "text": user_prompt})

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=500,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": messages_content}]
        )
        return response.content[0].text

    def _call_google(self, system_prompt: str, user_prompt: str, base64_frames: Optional[List[str]]) -> str:
        contents = [system_prompt + "\n\n" + user_prompt]

        if base64_frames:
            for b64 in base64_frames:
                # Key modification: using types.Part.from_bytes for new Google SDK
                image_bytes = base64.b64decode(b64)
                contents.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    )
                )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=500
                )
            )
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"


# ================= Main Execution Logic =================

def run_evaluation(dataset_path: str, output_path: str):
    """
    Main evaluation loop: loads data, processes videos, queries models, and saves results.
    """
    print(f"Initializing Model: {CURRENT_MODEL}...")
    model_handler = ModelDispatcher(CURRENT_MODEL)

    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    total = len(data)

    for index, item in enumerate(data):
        print(f"\n[{index + 1}/{total}] Processing Question ID: {item.get('id', index)}...")

        # --- 1. Dynamic Option Parsing ---
        option_keys = [k for k in item.keys() if k.startswith('O') and k[1:].isdigit()]
        option_keys.sort(key=lambda x: int(x[1:]))

        if not option_keys:
            print("   [Skip] No options found.")
            continue

        options_text = "\n".join([f"{k}: {item[k]}" for k in option_keys])
        start_opt, end_opt = option_keys[0], option_keys[-1]

        # --- 2. Material/Video Parsing ---
        materials = item.get('materials', [])
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

        # --- 3. Construct Prompt (Strictly Unchanged) ---
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

        # --- 4. Prepare Multimodal Data ---
        video_frames = []
        video_loaded_success = False

        if is_video_task and video_path:
            try:
                video_frames = extract_frames_from_video(video_path)
                if video_frames:
                    print(f"   [Video] Loaded {len(video_frames)} frames.")
                    video_loaded_success = True
                else:
                    print("   [Warning] 0 frames extracted.")
            except Exception as e:
                print(f"   [Error] Video load failed: {e}")

        # --- 5. Call Model ---
        try:
            raw_response = model_handler.generate(
                system_instruction,
                user_prompt,
                video_frames if video_loaded_success else None
            )

            # --- 6. Parse Results ---
            raw_response = str(raw_response).strip()
            prediction = "Unknown"
            explanation = "No explanation."

            # Optimized Regex: Some models like to output **O1** or 'O1'
            pred_match = re.search(r"Prediction:\s*[\*'\"]?(O\d+)[\*'\"]?", raw_response, re.IGNORECASE)
            if pred_match:
                prediction = pred_match.group(1).upper()
            else:
                # Fallback search
                for opt in reversed(option_keys):
                    if opt in raw_response:
                        prediction = opt
                        break

            expl_match = re.search(r"Explanation:\s*(.*)", raw_response, re.IGNORECASE | re.DOTALL)
            if expl_match:
                explanation = expl_match.group(1).strip()

            ground_truth = item.get('closeA', 'N/A')
            is_correct = (prediction == ground_truth)

            print(f"   GT: {ground_truth} | Pred: {prediction} | {'✅' if is_correct else '❌'}")

            results.append({
                "id": item.get('id', index),
                "model": CURRENT_MODEL,
                "type": "video" if video_loaded_success else "text",
                "question": item['Q'],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "explanation": explanation,
                "is_correct": is_correct,
                "raw_response": raw_response
            })

        except Exception as e:
            print(f"   [API Error] {e}")
            time.sleep(1)

    # --- 7. Save Results ---
    if results:
        total_count = len(results)
        correct_count = sum(1 for r in results if r['is_correct'])
        acc = (correct_count / total_count) * 100

        # Construct metadata for this run
        summary_data = {
            "model": CURRENT_MODEL,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": total_count,
            "correct_answers": correct_count,
            "accuracy": f"{acc:.2f}%"
        }

        print(f"\n=== Model: {CURRENT_MODEL} | Accuracy: {acc:.2f}% ===")

        # 1. Save detailed results for this specific model run
        final_results = {
            "summary": summary_data,
            "details": results
        }

        detail_output = output_path.replace(".json", f"_{CURRENT_MODEL}_detail.json")
        with open(detail_output, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to {detail_output}")

        # 2. Append to the global summary table
        all_summaries = []

        # Read old data if exists
        if os.path.exists(SUMMARY_FILE_PATH):
            try:
                with open(SUMMARY_FILE_PATH, 'r', encoding='utf-8') as f:
                    all_summaries = json.load(f)
            except Exception:
                all_summaries = []

        # Check if this model has been run before; update if yes, append if no
        existing_index = next((i for i, item in enumerate(all_summaries) if item["model"] == CURRENT_MODEL), -1)
        if existing_index > -1:
            all_summaries[existing_index] = summary_data
        else:
            all_summaries.append(summary_data)

        with open(SUMMARY_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)

        print(f"Summary updated in {SUMMARY_FILE_PATH}")


if __name__ == "__main__":
    # Ensure the directory exists before running
    output_dir = os.path.dirname(DEFAULT_OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_evaluation(DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_PATH)