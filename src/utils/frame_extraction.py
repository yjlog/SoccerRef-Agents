"""
SoccerRef-Agents: Frame Extraction Utilities
---------------------------------------------
Shared video frame extraction and encoding logic used by VideoAgent and Baseline.
"""

import os
import logging
import base64
from typing import List, Any

logger = logging.getLogger(__name__)

DEFAULT_SECONDS_PER_FRAME = 1.0
DEFAULT_MAX_FRAMES = 15
DEFAULT_JPEG_QUALITY = 80
DEFAULT_RESIZE_LONG_EDGE = 720


def encode_image_base64(frame: Any, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> str:
    """Encodes an OpenCV image frame to a Base64 string with JPEG compression."""
    import cv2
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, buffer = cv2.imencode(".jpg", frame, encode_param)
    return base64.b64encode(buffer).decode("utf-8")


def extract_frames_from_video(
    video_path: str,
    seconds_per_frame: float = DEFAULT_SECONDS_PER_FRAME,
    max_frames: int = DEFAULT_MAX_FRAMES,
    resize_long_edge: int = DEFAULT_RESIZE_LONG_EDGE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> List[str]:
    """
    Extracts frames from a video file at a specified interval, resizes them,
    and converts them to Base64 strings.

    Uses evenly-spaced downsampling when the number of extracted frames exceeds max_frames.

    Args:
        video_path: Absolute path to the video file.
        seconds_per_frame: Interval in seconds between extracted frames.
        max_frames: Maximum number of frames to return.
        resize_long_edge: Resize the longer edge of each frame to this pixel value (if larger).
        jpeg_quality: JPEG compression quality (0-100).

    Returns:
        A list of Base64 encoded JPEG image strings.
    """
    import cv2

    if not os.path.exists(video_path):
        logger.error("Video file not found: %s", video_path)
        return []

    video = cv2.VideoCapture(video_path)
    base64_frames: List[str] = []

    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25

    frame_interval = int(fps * seconds_per_frame)
    if frame_interval < 1:
        frame_interval = 1

    curr_frame = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        if curr_frame % frame_interval == 0:
            h, w = frame.shape[:2]
            scale = resize_long_edge / max(h, w)
            if scale < 1:
                new_size = (int(w * scale), int(h * scale))
                frame = cv2.resize(frame, new_size)

            base64_frames.append(encode_image_base64(frame, jpeg_quality))

        curr_frame += 1

    video.release()

    if len(base64_frames) > max_frames:
        step = len(base64_frames) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        base64_frames = [base64_frames[i] for i in indices]

    return base64_frames
