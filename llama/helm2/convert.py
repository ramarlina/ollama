#!/usr/bin/env python3
"""Convert Saina Helm 2 with its routed head, using pinned llama.cpp.

Usage: python convert.py --llama-cpp build/_deps/llama_cpp-src MODEL --outtype f16 --outfile MODEL.gguf
MODEL is a Helm 2 checkpoint (head-config.json with head_type routed_residual_v1).
All remaining arguments are passed to convert_hf_to_gguf.py. Keep the backbone at
F16 or wider: the head reads hidden states directly, and Q8_0 drifts.
"""
import argparse
import json
import re
from pathlib import Path
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--llama-cpp', type=Path, required=True)
args, remaining = parser.parse_known_args()
sys.path[:0] = [str(args.llama_cpp.resolve()), str((args.llama_cpp / 'gguf-py').resolve())]

import numpy as np
from safetensors import safe_open
from conversion.qwen import Qwen3_5TextModel
import convert_hf_to_gguf

HEAD_TENSORS = ('shared.weight', 'norm.weight', 'norm.bias', 'router.weight', 'down', 'up')
# The Go encoding labels options with these codes; the head's rows must follow them.
HELM_GO = Path(__file__).resolve().parents[2] / 'decision' / 'helm.go'

original_tensors = Qwen3_5TextModel.prepare_tensors
original_metadata = Qwen3_5TextModel.set_gguf_parameters


def head_config(model):
    config = json.loads((model.dir_model / 'head-config.json').read_text())
    if config.get('head_type') != 'routed_residual_v1' or config.get('prompt') != 'shared_context_query_slots_v1':
        raise ValueError('not a Helm 2 checkpoint')
    codes = re.search(r'var helmCodes = strings\.Fields\(`([^`]*)`\)', HELM_GO.read_text()).group(1).split()
    if config['codes'] != codes:
        raise ValueError('checkpoint answer codes differ from decision/helm.go')
    return config


def prepare_tensors(self):
    original_tensors(self)
    config = head_config(self)
    with safe_open(self.dir_model / 'experts.safetensors', framework='pt', device='cpu') as head:
        if sorted(head.keys()) != sorted(HEAD_TENSORS):
            raise ValueError(f'unexpected head tensors: {sorted(head.keys())}')
        for name in HEAD_TENSORS:
            # Keep the small head in float32.
            data = head.get_tensor(name).float().numpy()
            self.gguf_writer.add_tensor('helm2.' + name, np.ascontiguousarray(data))
        rows, experts = head.get_tensor('shared.weight').shape[0], head.get_tensor('router.weight').shape[0]
    if rows != len(config['codes']) or experts != config['experts']:
        raise ValueError('head shape does not match head-config.json')


def set_gguf_parameters(self):
    original_metadata(self)
    config = head_config(self)
    arch = self.gguf_writer.arch
    self.gguf_writer.add_string(f'{arch}.decision.type', 'helm2')
    self.gguf_writer.add_float64(f'{arch}.decision.route_progress', float(config['route_progress']))
    self.gguf_writer.add_uint32(f'{arch}.decision.experts', int(config['experts']))
    self.gguf_writer.add_uint32(f'{arch}.decision.bottleneck', int(config['bottleneck']))


Qwen3_5TextModel.prepare_tensors = prepare_tensors
Qwen3_5TextModel.set_gguf_parameters = set_gguf_parameters
sys.argv = [sys.argv[0], '--no-mtp', *remaining]
convert_hf_to_gguf.main()
