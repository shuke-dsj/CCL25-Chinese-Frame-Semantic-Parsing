#!/usr/bin/env python3
"""Precompute frozen Frame--FE textual prototypes for Task3 Experiment D."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from tasks.common.experiment_utils import REPO_ROOT, set_seed
from .frame_utils import load_frame_ontology


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    parser.add_argument('--pretrained_model_dir', default='./chinese_bert_wwm_ext')
    parser.add_argument(
        '--output_file',
        default='./tasks/task3/definition_prototypes/frame_fe_definition_last4_mean.npz',
    )
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--max_length', default=192, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return parser.parse_args()


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    frame_info_file = resolve(args.frame_info_file)
    pretrained_dir = resolve(args.pretrained_model_dir)
    output_file = resolve(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    frame_info, idx2label, label2idx, _, _ = load_frame_ontology(frame_info_file)
    tokenizer = BertTokenizer.from_pretrained(str(pretrained_dir), do_lower_case=True)
    encoder = BertModel.from_pretrained(str(pretrained_dir)).to(device)
    encoder.eval()

    text_a, text_b, pair_frame_ids, pair_role_ids = [], [], [], []
    frame_names = []
    max_candidates = max(len(frame.get('fes', [])) for frame in frame_info)
    frame_pair_indices = np.full((len(frame_info), max_candidates), -1, dtype=np.int32)
    frame_role_ids = np.full((len(frame_info), max_candidates), -1, dtype=np.int32)
    for frame_id, frame in enumerate(frame_info):
        frame_names.append(frame['frame_name'])
        frame_text = f"框架：{frame['frame_name']}。{frame.get('frame_def', '')}"
        for candidate_index, fe in enumerate(frame.get('fes', [])):
            pair_index = len(text_a)
            role_id = label2idx[fe['fe_name']]
            text_a.append(frame_text)
            text_b.append(f"框架元素：{fe['fe_name']}。{fe.get('fe_def', '')}")
            pair_frame_ids.append(frame_id)
            pair_role_ids.append(role_id)
            frame_pair_indices[frame_id, candidate_index] = pair_index
            frame_role_ids[frame_id, candidate_index] = role_id

    chunks = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in tqdm(range(0, len(text_a), args.batch_size), desc='definition-prototypes'):
            stop = min(start + args.batch_size, len(text_a))
            encoded = tokenizer(
                text_a[start:stop],
                text_b[start:stop],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors='pt',
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            output = encoder(**encoded, output_hidden_states=True)
            hidden = sum(output.hidden_states[-4:]) / 4.0
            mask = encoded['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            chunks.append(F.normalize(pooled, dim=-1).cpu().numpy())
    pair_prototypes = np.concatenate(chunks, axis=0).astype(np.float32)

    global_role_prototypes = np.zeros((len(idx2label), pair_prototypes.shape[1]), dtype=np.float32)
    global_counts = np.zeros(len(idx2label), dtype=np.int32)
    for pair_index, role_id in enumerate(pair_role_ids):
        global_role_prototypes[role_id] += pair_prototypes[pair_index]
        global_counts[role_id] += 1
    if (global_counts == 0).any():
        missing = [idx2label[i] for i in np.flatnonzero(global_counts == 0)]
        raise ValueError(f'Global roles without definition prototypes: {missing[:10]}')
    global_role_prototypes /= global_counts[:, None]
    global_role_prototypes /= np.maximum(
        np.linalg.norm(global_role_prototypes, axis=1, keepdims=True), 1e-12
    )

    # float16 reduces a 30k x 768 bank from roughly 94 MB to 47 MB.  The model
    # converts back to float32 once when loading.
    np.savez_compressed(
        output_file,
        pair_prototypes=pair_prototypes.astype(np.float16),
        global_role_prototypes=global_role_prototypes.astype(np.float16),
        frame_pair_indices=frame_pair_indices,
        frame_role_ids=frame_role_ids,
        pair_frame_ids=np.asarray(pair_frame_ids, dtype=np.int32),
        pair_role_ids=np.asarray(pair_role_ids, dtype=np.int32),
        frame_names=np.asarray(frame_names),
        role_names=np.asarray(idx2label),
    )
    metadata = {
        'frame_info_file': str(frame_info_file),
        'pretrained_model_dir': str(pretrained_dir),
        'output_file': str(output_file),
        'frames': len(frame_info),
        'frame_fe_pairs': len(text_a),
        'global_roles': len(idx2label),
        'hidden_size': int(pair_prototypes.shape[1]),
        'pooling': 'last4_mean_then_attention_mean',
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'device': str(device),
        'elapsed_seconds': time.perf_counter() - start_time,
    }
    output_file.with_suffix(output_file.suffix + '.meta.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8'
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
