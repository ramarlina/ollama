#!/usr/bin/env python3
"""Compare Ollama's helm2 encoding of Saina Helm 2 Omni (text and images) with the saina runtime (PyTorch FP32, CPU).

    python parity_omni.py CHECKPOINT EXAMPLES_DIR [--ollama URL] [--model NAME] [--limit 0.01] [--out parity.json]

Sends the same System One requests to Ollama's /v1/systemone and to saina's System One path (saina.jev.evaluate):
text only, a screenshot, a chess board, a chart, and an audio clip. Ollama takes images only, so the clip goes to
Ollama as its spectrogram PNG (the image saina reads for it, resized to the model's grid) and to saina as audio. Reports the largest absolute probability difference and checks both sides feed the model the same
number of tokens. Needs saina 0.4 with saina[media]. Exits nonzero above --limit.
"""
import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.modules['fla'] = None  # the flash-linear-attention kernels are GPU-only; use the reference path

from saina import media as M
from saina.jev import SystemOneRequest, evaluate
from saina.loader import load_scorer
from saina.prepare import encode_all, prepare_systemone


def choice(question, options):
    return {'type': 'choice', 'instructions': question, 'criteria': {o: None for o in options}}


def spectrogram_png(audio_bytes):
    """The spectrogram saina reads for a clip, resized to the model's patch grid (PIL bicubic, as saina does), as
    PNG: send this to Ollama as an image in place of the audio."""
    import io
    from PIL import Image
    image = M.audio_image(audio_bytes)
    policy = M.ImagePolicy()
    _, h, w = policy.grid(*image.size)
    buffer = io.BytesIO()
    image.resize((w * policy.patch_size, h * policy.patch_size), Image.BICUBIC).save(buffer, 'PNG')
    return buffer.getvalue()


def cases(ex):
    png = lambda name: base64.b64encode((ex / name).read_bytes()).decode()
    spectrogram = base64.b64encode(spectrogram_png((ex / 'voice-call.wav').read_bytes())).decode()
    wav = 'data:audio/wav;base64,' + base64.b64encode((ex / 'voice-call.wav').read_bytes()).decode()
    return [
        ('text', {'customer': 'Ana', 'message': 'I was charged twice for my subscription this month.'}, {
            'team': {'type': 'choice', 'instructions': 'Which team should handle this?',
                     'criteria': {'billing': 'Duplicate charges, refunds', 'technical': 'Bugs, outages'}},
            'refund': {'type': 'noul', 'instructions': 'Is the customer requesting a refund?'},
            'severity': {'type': 'score', 'instructions': 'How severe is this issue?',
                         'criteria': ['minor', 'moderate', 'serious']}}, [], None),
        ('screenshot', 'Checkout keeps failing, screenshot attached.', {
            'issue': choice('What went wrong?', ['Card declined', 'Site error', 'Login problem'])},
         [png('checkout-error.png')], None),
        ('chess', 'A chess puzzle from an online game.', {
            'move': choice('Black to move. Which move is best?', ['Rf7', 'h6', 'Rxf1', 'a5']),
            'check': {'type': 'noul', 'instructions': 'Is Black in check in this position?'}},
         [png('chess-puzzle.png')], None),
        ('chart', 'The quarterly sales dashboard, as a chart.', {
            'above': choice('How many regions have a value above 60?', ['1', '2', '3', '4'])},
         [png('revenue-chart.png')], None),
        ('audio', 'Inbound call to the bank: a short recording of English speech.', {
            'spoof': {'type': 'noul', 'instructions': 'Is this speech synthesized or voice-converted by a computer (a spoof)?'}},
         [spectrogram], wav),
    ]


def probabilities(answer):
    return [answer['noul']] if answer['type'] == 'noul' else list(answer['probabilities'].values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('examples', type=Path)
    ap.add_argument('--ollama', default='http://localhost:11434')
    ap.add_argument('--model', default='run-saina/saina-helm-2-omni-0.8b')
    ap.add_argument('--limit', type=float, default=0.01)
    ap.add_argument('--out')
    a = ap.parse_args()
    scorer = load_scorer(a.checkpoint, device='cpu')
    worst, rows = 0.0, []
    for name, state, questions, images, audio in cases(a.examples):
        body = {'model': a.model, 'state': state, 'questions': questions}
        native_body = dict(body)
        if audio:
            native_body['audios'] = [audio]
        elif images:
            native_body['images'] = ['data:image/png;base64,' + i for i in images]
        request = SystemOneRequest.model_validate(native_body)
        raw = M.request_media(request, scorer)
        model_tokens = len(encode_all(prepare_systemone(request), scorer, M.decode_media(raw) if raw else None)[0])
        native = evaluate(request, scorer)
        if images:
            body['images'] = images
        req = urllib.request.Request(a.ollama + '/v1/systemone', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            ollama = json.load(r)
        seconds = time.perf_counter() - t
        diff = max(max(abs(x - y) for x, y in zip(probabilities(native['answers'][q]), probabilities(ollama['answers'][q])))
                   for q in questions)
        tokens, otokens = model_tokens, ollama['usage']['input_tokens']  # the model's input length on both sides
        if tokens != otokens:
            print(f'  {name}: token counts differ: the inputs are not identical', flush=True)
            worst = float('inf')
        worst = max(worst, diff)
        rows.append({'case': name, 'input_tokens': tokens, 'ollama_input_tokens': otokens,
                     'max_abs_diff': diff, 'seconds': round(seconds, 3)})
        print(f'case {name}: {tokens} tokens (ollama {otokens}), max |p diff| {diff:.6f}, {seconds:.2f}s', flush=True)
    report = {'reference': 'saina System One path (saina.jev.evaluate), PyTorch FP32 on CPU',
              'cases': rows, 'max_abs_probability_diff': worst}
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    sys.exit(0 if worst <= a.limit else 1)


if __name__ == '__main__':
    main()
