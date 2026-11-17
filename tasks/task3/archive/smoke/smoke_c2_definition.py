#!/usr/bin/env python3
"""Small forward/backward compatibility test for Task3 C2 and D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from tasks.common.experiment_utils import REPO_ROOT, set_seed
from tasks.task3.dataset_experiments import (
    Task3RoleDataset,
    build_augmented_frame_roles,
    make_role_collate,
    train_role_frequencies,
)
from tasks.task3.train_role_classifier import build_model, forward_outputs, objective_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument(
        '--checkpoint',
        default='./tasks/task3/experiment_record/task3-C-efficient-span-5ep-result/checkpoints/task3_best.bin',
    )
    parser.add_argument(
        '--output_dir', default='./tasks/task3/analysis/task3-C2D-smoke'
    )
    return parser.parse_args()


def synthetic_bank(path, frame_info, idx2label, label2idx, dim=32):
    rng = np.random.default_rng(42)
    pair_count = sum(len(frame.get('fes', [])) for frame in frame_info)
    max_candidates = max(len(frame.get('fes', [])) for frame in frame_info)
    pair = rng.normal(size=(pair_count, dim)).astype(np.float32)
    pair /= np.maximum(np.linalg.norm(pair, axis=1, keepdims=True), 1e-8)
    frame_pair_indices = np.full((len(frame_info), max_candidates), -1, dtype=np.int32)
    frame_role_ids = np.full((len(frame_info), max_candidates), -1, dtype=np.int32)
    global_sum = np.zeros((len(idx2label), dim), dtype=np.float32)
    global_count = np.zeros(len(idx2label), dtype=np.int32)
    pair_index = 0
    for frame_id, frame in enumerate(frame_info):
        for candidate_index, fe in enumerate(frame.get('fes', [])):
            role_id = label2idx[fe['fe_name']]
            frame_pair_indices[frame_id, candidate_index] = pair_index
            frame_role_ids[frame_id, candidate_index] = role_id
            global_sum[role_id] += pair[pair_index]
            global_count[role_id] += 1
            pair_index += 1
    global_sum /= np.maximum(global_count[:, None], 1)
    global_sum /= np.maximum(np.linalg.norm(global_sum, axis=1, keepdims=True), 1e-8)
    np.savez_compressed(
        path,
        pair_prototypes=pair.astype(np.float16),
        global_role_prototypes=global_sum.astype(np.float16),
        frame_pair_indices=frame_pair_indices,
        frame_role_ids=frame_role_ids,
    )


def make_model_args(variant, checkpoint, prototype_file):
    return SimpleNamespace(
        model_variant=variant,
        config_file=str(REPO_ROOT / 'chinese_bert_wwm_ext/config.json'),
        warm_start_checkpoint=str(checkpoint),
        span_dim=256,
        role_dim=256,
        distance_dim=32,
        width_dim=16,
        definition_prototype_file=str(prototype_file),
        definition_scale_init=0.1,
        definition_temperature_init=10.0,
        definition_frame_topk=5,
        frame_probability_smoothing=0.1,
        frame_condition_dropout=0.3,
        freeze_definition_temperature=(variant == 'definition_pair_marginal'),
    )


def gradient_norm(model, names):
    return {
        name: float(dict(model.named_parameters())[name].grad.norm().detach().cpu())
        for name in names
    }


def main():
    args = parse_args()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prototype_file = output_dir / 'synthetic_definition_bank.npz'

    frame_info, idx2label, label2idx, frame_to_role_ids, _ = build_augmented_frame_roles(
        REPO_ROOT / 'dataset/frame_info.json', REPO_ROOT / 'dataset/cfn-train.json'
    )
    synthetic_bank(prototype_file, frame_info, idx2label, label2idx)
    frame_to_id = {
        frame['frame_name']: index for index, frame in enumerate(frame_info)
    }
    tokenizer = BertTokenizer(
        vocab_file=str(REPO_ROOT / 'chinese_bert_wwm_ext/vocab.txt'),
        do_lower_case=True,
    )
    dataset = Task3RoleDataset(
        REPO_ROOT / 'dataset/cfn-train.json', tokenizer, label2idx,
        train_role_frequencies(REPO_ROOT / 'dataset/cfn-train.json'),
        max_examples=args.batch_size,
    )
    collate = make_role_collate(
        frame_to_role_ids, len(idx2label), tokenizer.pad_token_id, device,
        frame_to_id=frame_to_id,
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)))

    c2 = build_model(
        make_model_args('efficient_span_calibrated', checkpoint, prototype_file),
        len(idx2label),
    ).to(device)
    c2_outputs = forward_outputs(c2, 'efficient_span_calibrated', batch)
    c2_loss, _, _, _, _ = objective_loss(c2_outputs, batch, global_loss_weight=0.2)
    c2_loss.backward()
    result = {
        'device': str(device),
        'batch_size': args.batch_size,
        'c2_logits_shape': list(c2_outputs['logits'].shape),
        'c2_loss': float(c2_loss.detach().cpu()),
        'c2_gradient_norms': gradient_norm(
            c2, ['interaction.0.weight', 'role_embeddings']
        ),
    }
    del c2, c2_outputs, c2_loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    d = build_model(
        make_model_args('definition_prototype', checkpoint, prototype_file),
        len(idx2label),
    ).to(device)
    d_outputs = forward_outputs(d, 'definition_prototype', batch)
    d_loss, _, _, _, _ = objective_loss(
        d_outputs, batch, global_loss_weight=0.2, prototype_loss_weight=0.3
    )
    d_loss.backward()
    result.update({
        'd_logits_shape': list(d_outputs['logits'].shape),
        'd_prototype_logits_shape': list(d_outputs['prototype_logits'].shape),
        'd_loss': float(d_loss.detach().cpu()),
        'definition_gate': float(d.definition_gate.detach().cpu()),
        'definition_temperature': float(d.definition_log_temperature.exp().detach().cpu()),
        'd_gradient_norms': gradient_norm(
            d,
            [
                'definition_projection.0.weight',
                'definition_query_projection.0.weight',
                'definition_gate',
            ],
        ),
    })
    del d, d_outputs, d_loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    d2 = build_model(
        make_model_args('definition_pair_marginal', checkpoint, prototype_file),
        len(idx2label),
    ).to(device)
    d2.train()
    batch['prototype_sample_weights'] = torch.ones(
        args.batch_size, dtype=torch.float32, device=device
    )
    d2_outputs = forward_outputs(d2, 'definition_pair_marginal', batch)
    d2_loss, _, _, _, gate_regularization = objective_loss(
        d2_outputs, batch,
        global_loss_weight=0.2,
        prototype_loss_weight=0.2,
        gate_regularization_weight=0.1,
        gate_regularization_target=0.1,
    )
    d2_loss.backward()
    zero_probabilities = torch.zeros(
        (args.batch_size, len(frame_info)), dtype=torch.float32, device=device
    )
    d2.eval()
    with torch.no_grad():
        fallback_outputs = d2(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            target=batch['target'], argument=batch['argument'],
            frame_probabilities=zero_probabilities, frame_topk=5,
        )
    if not torch.isfinite(d2_outputs['prototype_logits']).all():
        raise AssertionError('D2 pair-marginal logits contain non-finite values.')
    if not torch.isfinite(fallback_outputs['prototype_logits']).all():
        raise AssertionError('D2 all-global fallback logits contain non-finite values.')
    result.update({
        'd2_logits_shape': list(d2_outputs['logits'].shape),
        'd2_pair_logits_shape': list(d2_outputs['prototype_logits'].shape),
        'd2_loss': float(d2_loss.detach().cpu()),
        'd2_gate_regularization': float(gate_regularization.detach().cpu()),
        'd2_temperature_requires_grad': bool(
            d2.definition_log_temperature.requires_grad
        ),
        'd2_all_global_fallback_finite': bool(
            torch.isfinite(fallback_outputs['prototype_logits']).all()
        ),
        'd2_gradient_norms': gradient_norm(
            d2,
            [
                'definition_projection.0.weight',
                'definition_query_projection.0.weight',
                'definition_gate',
            ],
        ),
    })
    (output_dir / 'summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8'
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('[OK] Task3 C2/D/D2 forward, backward, fallback and C-checkpoint compatibility passed.')


if __name__ == '__main__':
    main()
