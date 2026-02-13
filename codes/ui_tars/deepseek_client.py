# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import os
import base64
import logging
from pathlib import Path

from openai import OpenAI

from ui_tars.action_parser import (
    parse_action_to_structure_output,
    parsing_response_to_pyautogui_code,
)
from ui_tars.prompt import COMPUTER_USE_DOUBAO, MOBILE_USE_DOUBAO

logger = logging.getLogger(__name__)

# Supported DeepSeek models
DEEPSEEK_MODELS = frozenset({
    "deepseek-chat",
    "deepseek-reasoner",
})

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MAX_TOKENS = 1024


def _encode_image(image_path: str) -> str:
    """Read and base64-encode a local image file."""
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    # Restrict to common image formats
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    if path.suffix.lower() not in allowed_suffixes:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Allowed: {', '.join(sorted(allowed_suffixes))}"
        )
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_mime_type(image_path: str) -> str:
    """Return the MIME type for a given image path."""
    suffix = Path(image_path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(suffix, "image/png")


class DeepSeekGUIAgent:
    """A GUI agent powered by DeepSeek API that produces executable pyautogui code.

    Usage:
        agent = DeepSeekGUIAgent()
        result = agent.act(
            instruction="Click the search bar",
            screenshot_path="screenshot.png",
            image_width=1920,
            image_height=1080,
        )
        print(result["pyautogui_code"])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        language: str = "English",
        platform: str = "computer",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """Initialize the DeepSeek GUI agent.

        Args:
            api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
            base_url: API base URL.
            model: Model name (e.g. "deepseek-chat").
            language: Language for the agent's Thought output.
            platform: "computer" or "mobile" - selects the prompt template.
            max_tokens: Maximum tokens in the response.
        """
        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise ValueError(
                "DeepSeek API key is required. Pass api_key= or set "
                "the DEEPSEEK_API_KEY environment variable."
            )

        if model not in DEEPSEEK_MODELS:
            raise ValueError(
                f"Unknown model '{model}'. Supported: {', '.join(sorted(DEEPSEEK_MODELS))}"
            )

        if platform not in ("computer", "mobile"):
            raise ValueError(f"platform must be 'computer' or 'mobile', got '{platform}'")

        self._client = OpenAI(api_key=resolved_key, base_url=base_url)
        self._model = model
        self._language = language
        self._platform = platform
        self._max_tokens = max_tokens

    def _build_system_prompt(self, instruction: str) -> str:
        """Build the system prompt from the template."""
        template = (
            COMPUTER_USE_DOUBAO if self._platform == "computer" else MOBILE_USE_DOUBAO
        )
        return template.format(language=self._language, instruction=instruction)

    def _call_api(
        self,
        system_prompt: str,
        screenshot_b64: str | None = None,
        mime_type: str = "image/png",
        user_message: str = "What is the next action?",
    ) -> str:
        """Call the DeepSeek API and return the raw response text."""
        user_content = []
        if screenshot_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{screenshot_b64}"
                },
            })
        user_content.append({"type": "text", "text": user_message})

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content

    def act(
        self,
        instruction: str,
        image_width: int,
        image_height: int,
        screenshot_path: str | None = None,
        screenshot_b64: str | None = None,
        model_type: str = "doubao",
        factor: int = 1000,
        input_swap: bool = True,
    ) -> dict:
        """Send a screenshot + instruction to DeepSeek and return parsed actions.

        Provide either screenshot_path (local file) or screenshot_b64 (pre-encoded).

        Args:
            instruction: What the agent should do (e.g. "Click the search bar").
            image_width: Original screenshot width in pixels.
            image_height: Original screenshot height in pixels.
            screenshot_path: Path to a local screenshot image.
            screenshot_b64: Base64-encoded screenshot (alternative to path).
            model_type: Coordinate mode - "doubao" (relative) or "qwen25vl" (absolute).
            factor: Coordinate scaling factor (default 1000).
            input_swap: Use clipboard paste for typing (default True).

        Returns:
            Dict with keys: raw_response, parsed_actions, pyautogui_code.
        """
        if not instruction or not instruction.strip():
            raise ValueError("instruction must not be empty")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")

        # Encode screenshot
        mime_type = "image/png"
        if screenshot_path:
            screenshot_b64 = _encode_image(screenshot_path)
            mime_type = _get_mime_type(screenshot_path)
        elif screenshot_b64 is None:
            logger.warning("No screenshot provided; sending text-only request.")

        # Call API
        system_prompt = self._build_system_prompt(instruction)
        raw_response = self._call_api(system_prompt, screenshot_b64, mime_type)
        logger.info("DeepSeek response: %s", raw_response)

        # Parse response into structured actions
        parsed_actions = parse_action_to_structure_output(
            raw_response,
            factor=factor,
            origin_resized_height=image_height,
            origin_resized_width=image_width,
            model_type=model_type,
        )

        # Generate pyautogui code
        pyautogui_code = parsing_response_to_pyautogui_code(
            parsed_actions,
            image_height=image_height,
            image_width=image_width,
            input_swap=input_swap,
        )

        return {
            "raw_response": raw_response,
            "parsed_actions": parsed_actions,
            "pyautogui_code": pyautogui_code,
        }
