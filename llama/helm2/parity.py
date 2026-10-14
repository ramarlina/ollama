#!/usr/bin/env python3
"""Compare Ollama's helm2 encoding with the saina runtime (PyTorch FP32 on CPU).

    python parity.py CHECKPOINT [--ollama http://localhost:11434] [--model run-saina/saina-helm-2-0.8b] [--out parity.json]

Sends the same System One requests to Ollama's /v1/systemone and to saina's own
System One path (saina.jev.evaluate) and reports the largest absolute probability
difference. The cases are the ones the Helm 2 GGUF release is checked on;
multi-label questions are asked as choice questions, since System One has no
multi-label type. Two more cases cover every question type and a
255-option question. Also checks that both sides feed the model the same number of
tokens. Exits nonzero above --limit.
"""
import argparse
import json
import sys
import time
import urllib.request

sys.modules['fla'] = None  # the flash-linear-attention kernels are GPU-only; use the reference path

from saina.jev import SystemOneRequest, evaluate
from saina.loader import load_scorer
from saina.prepare import encode_all, prepare_systemone

LONG = ' '.join(f'Ticket {i}: customer reports the invoice total does not match the order; '
                f'agent {i % 7} asked for a screenshot and promised a callback.' for i in range(150))


def choice(question, options):
    return {'type': 'choice', 'instructions': question, 'criteria': {o: None for o in options}}


CASES = [
    ('I was charged twice for my order and nobody answers the phone.', {
        'team': choice('Which team should handle this?', ['Billing', 'Technical support', 'Shipping']),
        'apply': choice('Which apply?', ['Refund request', 'Support unreachable', 'Praise']),
        'reply': choice('Is a reply needed?', ['Yes', 'No'])}),
    ('Can we move our call to Thursday?', {'reply': choice('Is a reply needed?', ['Yes', 'No'])}),
    ('The app crashes when I upload a photo larger than 10 MB on Android 15.', {
        'component': choice('Which component?', [f'Component {i}' for i in range(39)] + ['Media upload'])}),
    (json.dumps({'order': 4417, 'status': 'delivered', 'damage': True, 'complaints_this_month': 3}), {
        'urgency': choice('How urgent is this?', ['Low', 'Medium', 'High', 'Critical']),
        'issues': choice('Which issues apply?', ['Item damaged', 'Late delivery', 'Repeat complainer'])}),
    (LONG, {
        'topic': choice('What is the main topic?', ['Invoice mismatch', 'Shipping delay', 'Login problem']),
        'apply': choice('Which apply?', ['Callback promised', 'Screenshot requested', 'Refund issued'])}),
    # Every question type, with descriptions and a structured state.
    ({'customer': 'Ana', 'message': 'I was charged twice for my subscription this month.', 'plan': 'pro'}, {
        'team': {'type': 'choice', 'instructions': 'Which team should handle this?',
                 'criteria': {'billing': 'Duplicate charges, refunds, invoicing', 'technical': 'Bugs, outages',
                              'shipping': 'Delivery and tracking'}},
        'refund': {'type': 'noul', 'instructions': 'Is the customer requesting a refund?'},
        'reply': {'type': 'noul', 'instructions': 'Does this need a reply today?',
                  'criteria': {'true': 'Urgent for the customer', 'false': 'Can wait'}},
        'severity': {'type': 'score', 'instructions': 'How severe is this issue?',
                     'criteria': ['minor: customer is annoyed', 'moderate: work is impeded',
                                  'serious: money or data is at risk']}}),
    # The largest choice question System One allows, past the two-letter codes.
    ('Route this: printer on floor 3 is jammed.', {
        'queue': {'type': 'choice', 'instructions': 'Which queue?',
                  'criteria': {**{f'queue-{i}': None for i in range(254)},
                               'facilities-printers': 'Printers and office hardware'}}}),
]


def probabilities(answer):
    return [answer['noul']] if answer['type'] == 'noul' else list(answer['probabilities'].values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--ollama', default='http://localhost:11434')
    ap.add_argument('--model', default='run-saina/saina-helm-2-0.8b')
    ap.add_argument('--limit', type=float, default=0.005)
    ap.add_argument('--out')
    a = ap.parse_args()
    scorer = load_scorer(a.checkpoint, device='cpu')
    worst, rows = 0.0, []
    for i, (state, questions) in enumerate(CASES):
        body = {'model': a.model, 'state': state, 'questions': questions}
        request = SystemOneRequest.model_validate(body)
        tokens = len(encode_all(prepare_systemone(request), scorer)[0])
        native = evaluate(request, scorer)['answers']
        req = urllib.request.Request(a.ollama + '/v1/systemone', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            ollama = json.load(r)
        seconds = time.perf_counter() - t
        diff = 0.0
        for name in questions:
            n, o = probabilities(native[name]), probabilities(ollama['answers'][name])
            assert len(n) == len(o), name
            diff = max(diff, max(abs(x - y) for x, y in zip(n, o)))
        same_tokens = ollama['usage']['input_tokens'] == tokens
        worst = max(worst, diff)
        rows.append({'case': i, 'input_tokens': tokens, 'ollama_input_tokens': ollama['usage']['input_tokens'],
                     'max_abs_diff': diff, 'seconds': round(seconds, 3)})
        print(f'case {i}: {tokens} tokens (ollama {ollama["usage"]["input_tokens"]}), '
              f'max |p diff| {diff:.6f}, {seconds:.2f}s', flush=True)
        if not same_tokens:
            print('  token counts differ: the prompts are not identical', flush=True)
            worst = float('inf')
    report = {'reference': 'saina System One path (saina.jev.evaluate), PyTorch FP32 on CPU',
              'cases': rows, 'max_abs_probability_diff': worst}
    print(json.dumps(report, indent=1))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=1)
    sys.exit(0 if worst <= a.limit else 1)


if __name__ == '__main__':
    main()
