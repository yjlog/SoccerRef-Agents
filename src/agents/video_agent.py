import os
import cv2
import base64
import json
from typing import List, Dict, Any
from openai import OpenAI


class VideoAgent:
    """
    An agent responsible for processing video inputs, extracting frames,
    and using a Vision-Language Model (VLM) to analyze soccer scenarios.
    """

    def __init__(self, openai_client: OpenAI, model_name: str = "gpt-4o"):
        """
        Initializes the VideoAgent.

        Args:
            openai_client (OpenAI): The OpenAI client instance.
            model_name (str): The name of the model to use (default: "gpt-4o").
        """
        self.client = openai_client
        self.model_name = model_name

    def _encode_image(self, image_content: Any) -> str:
        """Encodes an OpenCV image frame to a Base64 string."""
        _, buffer = cv2.imencode(".jpg", image_content)
        return base64.b64encode(buffer).decode("utf-8")

    def _extract_frames(self, video_path: str, seconds_per_frame: float = 1.0) -> List[str]:
        """
        Extracts frames from a video file at a specified interval.

        Includes logic to limit the total number of frames to avoid Token overflow.

        Args:
            video_path (str): Path to the video file.
            seconds_per_frame (float): Interval in seconds to extract frames.

        Returns:
            List[str]: A list of Base64 encoded image strings.
        """
        if not os.path.exists(video_path):
            print(f"   [VideoAgent] Error: File not found {video_path}")
            return []

        video = cv2.VideoCapture(video_path)
        base64_frames = []

        # Get FPS, default to 25 if retrieval fails
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25

        curr_frame = 0
        target_step = int(fps * seconds_per_frame)
        if target_step < 1:
            target_step = 1

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            # Sample frames based on the interval
            if curr_frame % target_step == 0:
                base64_frames.append(self._encode_image(frame))

            curr_frame += 1

        video.release()

        # Limit maximum frames to prevent Token overflow (Max 15)
        if len(base64_frames) > 15:
            # Uniform sampling if we have too many frames
            step = len(base64_frames) // 15
            if step < 1:
                step = 1
            # Python slicing: [start:stop:step]
            base64_frames = base64_frames[::step][:15]

        return base64_frames

    def run(self, question_item: Dict[str, Any], video_path: str) -> Dict[str, Any]:
        """
        Analyzes the video to answer a specific multiple-choice question.

        Args:
            question_item (Dict): The question object containing 'Q' and options 'O1', 'O2', etc.
            video_path (str): The file path to the video.

        Returns:
            Dict: A dictionary containing the explanation and predicted option.
        """
        # 1. Extract frames from the video
        frames = self._extract_frames(video_path, seconds_per_frame=1.0)

        if not frames:
            return {
                "choice_explanation": "Video load failed",
                "predicted_option": "Unknown"
            }

        # 2. Dynamic Option Parsing
        option_keys = [k for k in question_item.keys() if k.startswith('O') and k[1:].isdigit()]
        option_keys.sort(key=lambda x: int(x[1:]))

        if option_keys:
            options_str = "\n".join([f"{k}: {question_item[k]}" for k in option_keys])
        else:
            options_str = "No specific options provided."

        # 3. Construct Prompt
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
            "The input contains video frames from a **live soccer match broadcast replay**."
            "Output JSON only."
        )

        # 4. Build Message Content (Text + Images)
        content = [{"type": "text", "text": prompt}]
        for f in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f}"}
            })

        # 5. Call API
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": content}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)

        except json.JSONDecodeError:
            print("   [VideoAgent] Analysis Error: Invalid JSON returned.")
            return {
                "choice_explanation": "Model output invalid JSON",
                "predicted_option": "Error"
            }
        except Exception as e:
            print(f"   [VideoAgent] Analysis Error: {e}")
            return {
                "choice_explanation": f"Error: {e}",
                "predicted_option": "Error"
            }