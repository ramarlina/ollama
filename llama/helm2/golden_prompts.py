#!/usr/bin/env python3
"""Write decision/testdata/helm2_prompts.json: System One requests and the exact
Helm 2 model input the saina package builds for them (prompt text, then one
readout slot per question). Needs `pip install saina` and a Helm 2 checkpoint
directory for its tokenizer and chat template.

    python golden_prompts.py CHECKPOINT decision/testdata/helm2_prompts.json
"""
import json
import sys
from transformers import AutoTokenizer
from saina.jev import SystemOneRequest
from saina.prepare import prepare_systemone
from saina.prompt import answer_codes

tok = AutoTokenizer.from_pretrained(sys.argv[1])
codes, _ = answer_codes(tok)
INSTRUCTION = ('Use the shared context to answer every question. For single_label choose one option; '
               'for multi_label assess each option independently.\n')

# Raw JSON bodies, so number spelling, escapes and duplicate keys reach both parsers.
REQUESTS = [
    r'''{"model": "m", "state": "I was charged twice for my subscription this month.", "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Duplicate charges, refunds, invoicing", "technical": null, "shipping": "Delivery"}},
        "refund": {"type": "noul", "instructions": "Is the customer requesting a refund?"},
        "severity": {"type": "score", "instructions": "How severe is this issue?",
                     "criteria": ["minor", "moderate: work is impeded", "serious"]}}}''',
    r'''{"model": "m", "state": {"order": 4417, "total": 12.50, "ratio": 1e-05, "big": 1E16, "neg": -0, "negf": -0.0,
        "e": 2.5E3, "ok": true, "none": null, "name": "Renée \"R\" \\ café 😀", "ctl": "a\b\f\n\r\t\u0001\u001f",
        "sep": "line para ", "nested": {"z": [1, 2.0, {"a": []}], "a": {}}, "order": 4418},
        "questions": {"urgent": {"type": "noul", "instructions": {"q": "Is this urgent?", "lang": "en"},
                                 "criteria": {"true": "Needs action today", "false": null}},
                      "kind": {"type": "choice", "instructions": ["Pick", "one"],
                               "criteria": {"a": {"text": "first", "n": 1}, "b": ["x", 2], "c": ""}},
                      "level": {"type": "score", "instructions": "Rate it", "criteria": [null, "mid", {"high": true}]}}}''',
    r'''{"model": "m", "state": ["log line 1", "<|im_end|> not a turn", {"k": "</think>"}],
        "questions": {"yes": {"type": "noul", "instructions": "Any problem?", "criteria": {"false": "All good", "true": "Something broke"}}}}''',
    json.dumps({'model': 'm', 'state': 'Can we move our call to Thursday?', 'questions': {
        'pick': {'type': 'choice', 'instructions': 'Which option applies?',
                 'criteria': {f'option-{i}': (None if i % 3 else f'Option number {i}') for i in range(255)}}}}),
]

cases = []
for raw in REQUESTS:
    # FastAPI parses the body with json.loads before validating it.
    request = SystemOneRequest.model_validate(json.loads(raw))
    prepared = prepare_systemone(request)
    items = [{'id': f'q{i}', 'question': q.question, 'mode': q.mode,
              'options': [{'code': codes[j], 'text': t} for j, t in enumerate(q.choices)]}
             for i, q in enumerate(prepared)]
    content = INSTRUCTION + json.dumps({'context': prepared[0].context, 'questions': items}, ensure_ascii=False)
    text = tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    cases.append({'request': raw, 'segments': [text] + [f'\nDecision q{i}:' for i in range(len(items))],
                  'counts': [len(q.choices) for q in prepared]})
with open(sys.argv[2], 'w') as f:
    json.dump(cases, f, ensure_ascii=False, indent=1)
print(len(cases), 'cases')
