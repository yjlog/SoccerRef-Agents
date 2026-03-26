import logging
import json
from typing import List, Dict, Any

from openai import OpenAI

from utils.frame_extraction import extract_frames_from_video

logger = logging.getLogger(__name__)


class VideoAgent:
    """
    An agent responsible for processing video inputs, extracting frames,
    and using a Vision-Language Model (VLM) to analyze soccer scenarios.
    """

    def __init__(self, openai_client: OpenAI, model_name: str = "gpt-4o") -> None:
        """
        Initializes the VideoAgent.

        Args:
            openai_client: The OpenAI client instance.
            model_name: The name of the model to use (default: "gpt-4o").
        """
        self.client = openai_client
        self.model_name = model_name

    def run(self, question_item: Dict[str, Any], video_path: str) -> Dict[str, Any]:
        """
        Analyzes the video to answer a specific multiple-choice question.

        Args:
            question_item: The question object containing 'Q' and options 'O1', 'O2', etc.
            video_path: The file path to the video.

        Returns:
            A dictionary containing the explanation and predicted option.
        """
        frames = extract_frames_from_video(video_path)

        if not frames:
            return {
                "choice_explanation": "Video load failed",
                "predicted_option": "Unknown",
            }

        option_keys = [k for k in question_item.keys() if k.startswith("O") and k[1:].isdigit()]
        option_keys.sort(key=lambda x: int(x[1:]))

        if option_keys:
            options_str = "\n".join([f"{k}: {question_item[k]}" for k in option_keys])
        else:
            options_str = "No specific options provided."

        prompt = (
            f"- Input Type: Broadcast Replay Video\n\n"
            f"Question: {question_item.get('Q', '')}\n\n"
            f"Options:\n{options_str}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Analyze the input video carefully.\n"
            f"2. Select the ONE correct option ID.\n"
            f"3. Provide a brief explanation in English.\n\n"
            f"Output JSON format:\n"
            f"{{\n"
            f"  \"choice_explanation\": \"...\",\n"
            f"  \"predicted_option\": \"O1\"\n"
            f"}}"
        )

        system_instruction = (
            "You are a professional AI Soccer Referee Assistant. "
            "The input contains video frames from a live soccer match broadcast replay. "
            "Output JSON only."
        )

        content: List[Any] = [{"type": "text", "text": prompt}]
        for frame_b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            if not response.choices:
                logger.warning("VideoAgent: Empty response from API.")
                return {
                    "choice_explanation": "API returned empty response",
                    "predicted_option": "Error",
                }
            raw_content = response.choices[0].message.content
            if not raw_content:
                logger.warning("VideoAgent: Empty content from API.")
                return {
                    "choice_explanation": "API returned empty content",
                    "predicted_option": "Error",
                }
            return json.loads(raw_content)

        except json.JSONDecodeError:
            logger.warning("VideoAgent: Invalid JSON returned by model.")
            return {
                "choice_explanation": "Model output invalid JSON",
                "predicted_option": "Error",
            }
        except Exception as e:
            logger.error("VideoAgent: Analysis error: %s", e)
            return {
                "choice_explanation": f"Error: {e}",
                "predicted_option": "Error",
            }
