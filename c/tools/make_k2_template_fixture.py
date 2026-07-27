"""Genera tests/k2_template_expected.json: rendering dell'UFFICIALE
chat_template.jinja di Kimi-K2 (tools/k2_chat_template.jinja, scaricato dal
repo HF del modello) su un set di conversazioni, via jinja2 vero.

Il fixture committato permette a tests/test_k2_template.py di verificare la
parita' BYTE-PER-BYTE di render_chat_k2 senza dipendere da jinja2 a runtime
di CI. Rigenerare solo se il template ufficiale cambia:
    python3 tools/make_k2_template_fixture.py   (richiede: pip install jinja2)
"""
import json
import os

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(HERE)

env = jinja2.Environment()
env.filters["tojson"] = lambda v, separators=None: json.dumps(
    v, separators=separators, ensure_ascii=False)
tpl = env.from_string(open(os.path.join(HERE, "k2_chat_template.jinja")).read())

WEATHER_TOOL = [{"type": "function",
                 "function": {"name": "get_weather",
                              "parameters": {"type": "object",
                                             "properties": {"city": {"type": "string"}}}}}]

CASES = {
    "basic_user": dict(
        messages=[{"role": "user", "content": "Hi"}]),
    "system_first": dict(
        messages=[{"role": "system", "content": "Sys"},
                  {"role": "user", "content": "Hi"},
                  {"role": "assistant", "content": "Yo"},
                  {"role": "user", "content": "Again"}]),
    "named_user": dict(
        messages=[{"role": "user", "name": "alice", "content": "hey"}]),
    "tool_round_trip": dict(
        tools=WEATHER_TOOL,
        messages=[{"role": "user", "content": "weather?"},
                  {"role": "assistant", "content": "",
                   "tool_calls": [{"id": "functions.get_weather:0", "type": "function",
                                   "function": {"name": "get_weather",
                                                "arguments": "{\"city\":\"Rome\"}"}}]},
                  {"role": "tool", "tool_call_id": "functions.get_weather:0",
                   "content": "sunny"}]),
    "multi_tool_calls": dict(
        tools=WEATHER_TOOL,
        messages=[{"role": "user", "content": "compare"},
                  {"role": "assistant", "content": "checking",
                   "tool_calls": [{"id": "functions.get_weather:0", "type": "function",
                                   "function": {"name": "get_weather",
                                                "arguments": "{\"city\":\"Rome\"}"}},
                                  {"id": "functions.get_weather:1", "type": "function",
                                   "function": {"name": "get_weather",
                                                "arguments": "{\"city\":\"Oslo\"}"}}]}]),
    "unicode_content": dict(
        messages=[{"role": "user", "content": "你好 world 🙂"}]),
    "developer_role": dict(
        messages=[{"role": "developer", "content": "Rules"},
                  {"role": "user", "content": "Hi"}]),
}

out = {}
for key, case in CASES.items():
    out[key] = {"case": case,
                "expected": tpl.render(add_generation_prompt=True, **case)}

path = os.path.join(C, "tests", "k2_template_expected.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"wrote {path}: {len(out)} cases")
