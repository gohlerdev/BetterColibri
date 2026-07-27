"""K2 chat-template parity: render_chat_k2 vs the OFFICIAL chat_template.jinja.

tests/k2_template_expected.json holds the official template's own rendering
(produced by real jinja2 via tools/make_k2_template_fixture.py) for a set of
conversations; render_chat_k2 must match BYTE FOR BYTE — including the
template's own whitespace quirk after an injected default system turn. Tool
parsing is tested round-trip: what render_chat_k2 writes, parse_tool_calls_k2
reads back. The GLM family stays pinned by TemplateTest in test_openai_server;
this file never touches the family dispatch (COLI_CHAT_FAMILY default = glm).
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai_server import (APIError, parse_tool_calls_k2,  # noqa: E402
                           render_chat_k2)

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "k2_template_expected.json")


class K2TemplateParityTest(unittest.TestCase):
    """Byte parity with the official jinja on every fixture conversation."""

    @classmethod
    def setUpClass(cls):
        with open(FIXTURE, encoding="utf-8") as f:
            cls.fixture = json.load(f)

    def test_fixture_is_nontrivial(self):
        self.assertGreaterEqual(len(self.fixture), 6)

    def test_byte_parity_with_official_template(self):
        for key, entry in self.fixture.items():
            with self.subTest(case=key):
                case = entry["case"]
                got = render_chat_k2(case["messages"], tools=case.get("tools"))
                self.assertEqual(got, entry["expected"],
                                 f"render_chat_k2 diverges from official jinja on {key!r}")


class K2RenderRulesTest(unittest.TestCase):
    def test_thinking_flags_ignored(self):
        base = render_chat_k2([{"role": "user", "content": "Hi"}])
        self.assertEqual(render_chat_k2([{"role": "user", "content": "Hi"}],
                                        True, "high"), base)

    def test_tool_choice_none_drops_declaration(self):
        prompt = render_chat_k2([{"role": "user", "content": "hi"}],
                                tools=[{"type": "function", "function": {"name": "f"}}],
                                tool_choice="none")
        self.assertNotIn("tool_declare", prompt)

    def test_tool_choice_forced_appends_instruction(self):
        prompt = render_chat_k2([{"role": "user", "content": "hi"}],
                                tools=[{"type": "function", "function": {"name": "f"}},
                                       {"type": "function", "function": {"name": "g"}}],
                                tool_choice={"function": {"name": "f"}})
        self.assertIn("You must call the function `f`", prompt)
        self.assertNotIn('"name":"g"', prompt)

    def test_rejects_bad_role_and_empty(self):
        with self.assertRaises(APIError):
            render_chat_k2([])
        with self.assertRaises(APIError):
            render_chat_k2([{"role": "oracle", "content": "x"}])

    def test_generation_prompt_suffix(self):
        prompt = render_chat_k2([{"role": "user", "content": "Hi"}])
        self.assertTrue(prompt.endswith("<|im_assistant|>assistant<|im_middle|>"))


class K2ParseTest(unittest.TestCase):
    def test_parse_single_call(self):
        reply = ("Sure.<|tool_calls_section_begin|><|tool_call_begin|>"
                 "functions.get_weather:0<|tool_call_argument_begin|>"
                 '{"city":"Rome"}<|tool_call_end|><|tool_calls_section_end|>')
        text, calls = parse_tool_calls_k2(reply)
        self.assertEqual(text, "Sure.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Rome"})

    def test_parse_multiple_calls(self):
        reply = ("<|tool_calls_section_begin|>"
                 "<|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>{}<|tool_call_end|>"
                 '<|tool_call_begin|>functions.b:1<|tool_call_argument_begin|>{"k":1}<|tool_call_end|>'
                 "<|tool_calls_section_end|>")
        text, calls = parse_tool_calls_k2(reply)
        self.assertEqual(text, "")
        self.assertEqual([c["function"]["name"] for c in calls], ["a", "b"])

    def test_render_parse_round_trip(self):
        """What the renderer writes for history, the parser must read back."""
        calls_in = [{"id": "functions.get_weather:0", "type": "function",
                     "function": {"name": "get_weather",
                                  "arguments": '{"city":"Rome"}'}}]
        prompt = render_chat_k2([
            {"role": "user", "content": "w?"},
            {"role": "assistant", "content": "", "tool_calls": calls_in}])
        segment = prompt.split("<|im_assistant|>")[1]
        text, calls = parse_tool_calls_k2(segment)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Rome"})

    def test_non_dict_and_broken_json_args(self):
        reply = ('<|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>'
                 '[1,2]<|tool_call_end|>'
                 '<|tool_call_begin|>functions.g:1<|tool_call_argument_begin|>'
                 '{broken<|tool_call_end|>')
        _text, calls = parse_tool_calls_k2(reply)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"value": [1, 2]})
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {"raw": "{broken"})

    def test_plain_text_untouched(self):
        text, calls = parse_tool_calls_k2("just an answer")
        self.assertEqual((text, calls), ("just an answer", []))


if __name__ == "__main__":
    unittest.main()
