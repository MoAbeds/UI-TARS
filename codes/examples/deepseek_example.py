"""Example: Using DeepSeek API as the main LLM for UI-TARS GUI automation.

Prerequisites:
    pip install ui-tars[deepseek]
    export DEEPSEEK_API_KEY="your-api-key-here"
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_tars.deepseek_client import DeepSeekGUIAgent


def main():
    # --- Option 1: Basic usage with a screenshot file ---
    agent = DeepSeekGUIAgent(
        # api_key="sk-..."          # or set DEEPSEEK_API_KEY env var
        model="deepseek-chat",       # use "deepseek-chat" for text
        language="English",
        platform="computer",         # "computer" or "mobile"
    )

    result = agent.act(
        instruction="Click the search bar and type 'hello world'",
        screenshot_path="screenshot.png",  # path to your screenshot
        image_width=1920,
        image_height=1080,
    )

    print("=== Raw LLM Response ===")
    print(result["raw_response"])
    print()
    print("=== Parsed Actions ===")
    for action in result["parsed_actions"]:
        print(f"  {action['action_type']}: {action['action_inputs']}")
    print()
    print("=== Generated PyAutoGUI Code ===")
    print(result["pyautogui_code"])

    # --- Option 2: With pre-encoded base64 screenshot ---
    # import base64
    # with open("screenshot.png", "rb") as f:
    #     b64 = base64.b64encode(f.read()).decode()
    #
    # result = agent.act(
    #     instruction="Click the submit button",
    #     screenshot_b64=b64,
    #     image_width=1920,
    #     image_height=1080,
    # )

    # --- Option 3: Text-only (no screenshot) ---
    # result = agent.act(
    #     instruction="Press Ctrl+S to save",
    #     image_width=1920,
    #     image_height=1080,
    # )

    # --- Option 4: Mobile platform ---
    # mobile_agent = DeepSeekGUIAgent(
    #     model="deepseek-chat",
    #     platform="mobile",
    # )
    # result = mobile_agent.act(
    #     instruction="Tap the settings icon",
    #     screenshot_path="phone_screenshot.png",
    #     image_width=1080,
    #     image_height=2400,
    # )


if __name__ == "__main__":
    main()
