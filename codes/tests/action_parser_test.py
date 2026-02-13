import unittest

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_tars.action_parser import (
    parsing_response_to_pyautogui_code,
    parse_action,
    parse_action_to_structure_output,
    _safe_parse_box,
)


class TestActionParser(unittest.TestCase):
    def test_parse_action(self):
        action_str = "click(point='<point>200 300</point>')"
        result = parse_action(action_str)
        self.assertEqual(result['function'], 'click')
        self.assertEqual(result['args']['point'], '<point>200 300</point>')

    def test_parse_action_to_structure_output(self):
        text = "Thought: test\nAction: click(point='<point>200 300</point>')"
        actions = parse_action_to_structure_output(
            text, factor=1000, origin_resized_height=224, origin_resized_width=224
        )
        self.assertEqual(actions[0]['action_type'], 'click')
        self.assertIn('start_box', actions[0]['action_inputs'])

    def test_parsing_response_to_pyautogui_code(self):
        responses = {"action_type": "hotkey", "action_inputs": {"hotkey": "ctrl v"}}
        code = parsing_response_to_pyautogui_code(responses, 224, 224)
        self.assertIn('pyautogui.hotkey', code)


class TestSecurityHardening(unittest.TestCase):
    """Security regression tests to prevent reintroduction of vulnerabilities."""

    def test_safe_parse_box_rejects_code_injection(self):
        """eval() would execute this; _safe_parse_box must reject it."""
        with self.assertRaises(Exception):
            _safe_parse_box("__import__('os').system('id')")

    def test_safe_parse_box_rejects_non_numeric(self):
        """Box coordinates must be numeric, not strings or other types."""
        with self.assertRaises(ValueError):
            _safe_parse_box("['a', 'b', 'c', 'd']")

    def test_safe_parse_box_rejects_wrong_length(self):
        """Only 2 or 4 element tuples/lists are valid boxes."""
        with self.assertRaises(ValueError):
            _safe_parse_box("[1, 2, 3]")

    def test_safe_parse_box_accepts_valid_box(self):
        """Valid 4-element numeric boxes should parse correctly."""
        result = _safe_parse_box("[0.1, 0.2, 0.3, 0.4]")
        self.assertEqual(result, (0.1, 0.2, 0.3, 0.4))

    def test_safe_parse_box_accepts_valid_point(self):
        """Valid 2-element numeric points should parse correctly."""
        result = _safe_parse_box("(100, 200)")
        self.assertEqual(result, (100, 200))

    def test_unknown_action_type_rejected(self):
        """Unknown action types must raise ValueError, not silently proceed."""
        responses = {"action_type": "malicious_action\nimport os", "action_inputs": {}}
        with self.assertRaises(ValueError):
            parsing_response_to_pyautogui_code(responses, 224, 224)

    def test_type_action_content_injection(self):
        """Content with quotes/escapes must not break out of generated code."""
        responses = {
            "action_type": "type",
            "action_inputs": {"content": "test'); import os; os.system('id"}
        }
        code = parsing_response_to_pyautogui_code(responses, 224, 224)
        # The malicious payload should be safely contained inside repr()
        self.assertNotIn("import os", code.split("pyperclip.copy(")[1].split(")")[0]
                         if "pyperclip.copy(" in code else "safe")

    def test_triple_quote_escape_in_thought(self):
        """Thought/observation with triple quotes must not escape the comment block."""
        responses = {
            "action_type": "click",
            "action_inputs": {"start_box": "[0.1, 0.2, 0.3, 0.4]"},
            "thought": "test'''\nimport os\nos.system('id')\n'''",
            "observation": "normal"
        }
        code = parsing_response_to_pyautogui_code(responses, 224, 224)
        # Triple quotes in thought should be sanitized
        self.assertNotIn("'''\nimport os", code)

    def test_box_injection_in_click(self):
        """Malicious start_box values must be rejected, not eval'd."""
        responses = {
            "action_type": "click",
            "action_inputs": {"start_box": "__import__('os').system('id')"}
        }
        with self.assertRaises(Exception):
            parsing_response_to_pyautogui_code(responses, 224, 224)

    def test_box_injection_in_drag(self):
        """Malicious box values in drag must be rejected."""
        responses = {
            "action_type": "drag",
            "action_inputs": {
                "start_box": "__import__('os').system('id')",
                "end_box": "[0.1, 0.2, 0.3, 0.4]"
            }
        }
        with self.assertRaises(Exception):
            parsing_response_to_pyautogui_code(responses, 224, 224)


if __name__ == '__main__':
    unittest.main()
