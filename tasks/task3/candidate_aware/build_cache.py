#!/usr/bin/env python3
"""Build aligned B+C2+D2 top-k candidate caches for Task3 D4.

Train uses gold argument boundaries but *predicted* Task1 probabilities and
the frozen B/C2/D2 generator.  Dev reuses the already audited full-Pipeline
caches, including Task2 predicted boundaries.  D4 therefore learns and is
evaluated on the same cross-Frame candidate score space.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from tasks.common.experiment_utils import REPO_ROOT, set_seed
from tasks.task1.dataset import Dataset as Task1Dataset
from tasks.task1.model import Model as Task1Model
from tasks.task1.prototype.model import FramePrototypeModel
from tasks.task3.analyze_pipeline_ensemble import (
    build_gold_mask,
    percentile_confidence,
    zscore_rows,
)
from tasks.task3.dataset_experiments import build_augmented_frame_roles
from tasks.task3.evaluate_full_pipeline import (
    build_model,
    collect_logits,
    collate as role_collate,
    softmax,
)


TASK_DIR = REPO_ROOT / 'tasks' / 'task3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train_gold', 'dev_pipeline'])
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--split_file', default=str(REPO_ROOT / 'dataset/cfn-train.json'))
    parser.add_argument('--train_file', default=str(REPO_ROOT / 'dataset/cfn-train.json'))
    parser.add_argument('--frame_info_file', default=str(REPO_ROOT / 'dataset/frame_info.json'))
    parser.add_argument('--pretrained_model_dir', default=str(REPO_ROOT / 'chinese_bert_wwm_ext'))
    parser.add_argument('--config_file', default=str(REPO_ROOT / 'chinese_bert_wwm_ext/config.json'))
    parser.add_argument('--vocab_file', default=str(REPO_ROOT / 'chinese_bert_wwm_ext/vocab.txt'))
    parser.add_argument('--definition_prototype_file', default=str(
        TASK_DIR / 'definition_prototypes/frame_fe_definition_last4_mean.npz'
    ))
    parser.add_argument('--candidate_topk', default=5, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_rows', default=0, type=int)

    # Frozen current generator.
    parser.add_argument('--b_weight', default=0.585, type=float)
    parser.add_argument('--c2_weight', default=0.10, type=float)
    parser.add_argument('--d2_weight', default=0.315, type=float)
    parser.add_argument('--prior_low', default=0.6, type=float)
    parser.add_argument('--prior_high', default=1.8, type=float)
    parser.add_argument('--prior_power', default=2.0, type=float)
    parser.add_argument('--prior_epsilon', default=1e-8, type=float)
    parser.add_argument('--b_checkpoint', default=str(
        TASK_DIR / 'experiment_record/task3-B-frame-masked-warmstart-2ep-result/checkpoints/task3_best.bin'
    ))
    parser.add_argument('--c2_checkpoint', default=str(
        TASK_DIR / 'experiment_record/task3-C2-mixed-objective-a020-3ep-result/checkpoints/task3_best.bin'
    ))
    parser.add_argument('--d2_checkpoint', default=str(
        TASK_DIR / 'experiment_record/task3-D2-pair-marginal-longtail-2ep-result/checkpoints/task3_epoch_1.bin'
    ))

    # Train-mode Task1 inference.  A temperature >1 deliberately prevents
    # in-sample Task1 predictions from collapsing to one-hot distributions.
    parser.add_argument('--task1_baseline_checkpoint', default=str(
        REPO_ROOT / 'tasks/task1/experiment_record/baseline-scheduler-fix-v2-10ep-result/checkpoints/task1_best.bin'
    ))
    parser.add_argument('--task1_prototype_checkpoint', default=str(
        REPO_ROOT / 'tasks/task1/experiment_record/frame-prototype-mvp-10ep-result/checkpoints/task1_prototype_best.bin'
    ))
    parser.add_argument('--task1_prototype_file', default=str(
        REPO_ROOT / 'tasks/task1/assets/frame_prototypes/name-def-fes-mean.pt'
    ))
    parser.add_argument('--task1_baseline_weight', default=0.5, type=float)
    parser.add_argument('--task1_prototype_gamma', default=2.7, type=float)
    parser.add_argument('--task1_probability_temperature', default=1.5, type=float)

    # Dev-mode audited caches.
    parser.add_argument('--c2_pipeline_cache', default=str(
        TASK_DIR / 'analysis/task3-full-pipeline-B-C2-frame/pipeline_logits.npz'
    ))
    parser.add_argument('--d2_pipeline_cache', default=str(
        TASK_DIR / 'analysis/task3-full-pipeline-B-D2-epoch1/pipeline_logits.npz'
    ))
    return parser.parse_args()


def read_json(path):
    with open(path, 'r', encoding='utf8') as handle:
        return json.load(handle)


def validate_checkpoint_archives(paths):
    """Fail before expensive Task1 inference when an upload is truncated."""
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f'D4 checkpoint is missing: {path}')
        try:
            with zipfile.ZipFile(path, 'r') as archive:
                if not archive.namelist():
                    raise RuntimeError(f'D4 checkpoint archive is empty: {path}')
        except zipfile.BadZipFile as error:
            raise RuntimeError(
                f'D4 checkpoint is truncated/corrupted and must be re-synced: '
                f'{path} (bytes={path.stat().st_size})'
            ) from error
        print(f'[checkpoint OK] {path} bytes={path.stat().st_size}')


def task1_collate(rows, device):
    max_length = max(len(row[0]) for row in rows)
    pad = lambda values: values + [0] * (max_length - len(values))
    return (
        torch.tensor([pad(row[0]) for row in rows], dtype=torch.long, device=device),
        torch.tensor([pad(row[1]) for row in rows], dtype=torch.long, device=device),
        [row[2] for row in rows],
        torch.tensor([row[3] for row in rows], dtype=torch.long, device=device),
        [int(row[4]) for row in rows],
    )


@torch.no_grad()
def task1_probabilities(args, rows, frame_info, device):
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = Task1Dataset(args.split_file, args.frame_info_file, tokenizer)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(task1_collate, device=device),
    )
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = len(frame_info)
    config.model_variant = 'rope'
    config.target_aware_layers = 'last'
    baseline = Task1Model(config)
    baseline.load_state_dict(torch.load(args.task1_baseline_checkpoint, map_location='cpu'), strict=False)
    baseline.to(device).eval()
    baseline_logits = []
    sentence_ids = []
    for input_ids, mask, target, _, ids in tqdm(loader, desc='D4-Task1-baseline'):
        output = baseline(
            input_ids=input_ids, attention_mask=mask, target=target,
            labels=None, device=device, for_test=True,
        )['logits']
        baseline_logits.append(output.detach().cpu().float().numpy())
        sentence_ids.extend(ids)
    del baseline
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    prototype = FramePrototypeModel(config, args.task1_prototype_file)
    prototype.load_state_dict(torch.load(args.task1_prototype_checkpoint, map_location='cpu'), strict=False)
    prototype.to(device).eval()
    prototype_logits = []
    for input_ids, mask, target, _, _ in tqdm(loader, desc='D4-Task1-prototype'):
        output = prototype(input_ids, mask, target, labels=None, for_test=True)['logits']
        prototype_logits.append(output.detach().cpu().float().numpy())
    del prototype
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    baseline_logits = np.concatenate(baseline_logits).astype(np.float32)
    prototype_logits = np.concatenate(prototype_logits).astype(np.float32)
    frame_names = [row['frame_name'] for row in frame_info]
    seen_frames = {row['frame'] for row in read_json(args.train_file)}
    seen_mask = np.asarray([name in seen_frames for name in frame_names], dtype=bool)
    prototype_logits[:, seen_mask] -= args.task1_prototype_gamma
    probability = (
        args.task1_baseline_weight * softmax(baseline_logits)
        + (1.0 - args.task1_baseline_weight) * softmax(prototype_logits)
    )
    temperature = float(args.task1_probability_temperature)
    if temperature <= 0:
        raise ValueError('task1_probability_temperature must be positive.')
    probability = np.power(probability.clip(min=1e-12), 1.0 / temperature)
    probability /= probability.sum(axis=1, keepdims=True)
    expected = [int(row['sentence_id']) for row in rows]
    if sentence_ids != expected:
        raise RuntimeError('Task1 Train inference order differs from split_file.')
    return probability.astype(np.float32)


class GoldSpanDataset(torch.utils.data.Dataset):
    """Task3 gold spans in the same tuple format as full-Pipeline inference."""
    def __init__(self, rows, frame_probabilities, tokenizer, role2idx, max_rows=0):
        from tasks.task3.dataset_experiments import shifted_character_position
        self.shift = shifted_character_position
        self.tokenizer = tokenizer
        self.target_start_id = tokenizer.convert_tokens_to_ids('[unused1]')
        self.target_end_id = tokenizer.convert_tokens_to_ids('[unused2]')
        self.items = []
        for instance_index, source in enumerate(rows):
            target = source['target'][-1]
            for span in source.get('cfn_spans', []):
                self.items.append({
                    'instance_index': int(instance_index),
                    'sentence_id': int(source['sentence_id']),
                    'text': source['text'],
                    'target_start': int(target['start']),
                    'target_end': int(target['end']),
                    'argument_start': int(span['start']),
                    'argument_end': int(span['end']),
                    'gold_role_id': int(role2idx[span['fe_name']]),
                    'frame_prob': frame_probabilities[instance_index],
                })
        if max_rows:
            self.items = self.items[: int(max_rows)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        encoded = self.tokenizer.encode_plus(list(row['text']))
        input_ids = encoded.data['input_ids']
        attention_mask = encoded.data['attention_mask']
        raw_ts, raw_te = row['target_start'], row['target_end']
        ts, te = raw_ts + 1, raw_te + 1
        input_ids = (
            input_ids[:ts] + [self.target_start_id]
            + input_ids[ts:te + 1] + [self.target_end_id]
            + input_ids[te + 1:]
        )
        attention_mask = (
            attention_mask[:ts] + [1]
            + attention_mask[ts:te + 1] + [1]
            + attention_mask[te + 1:]
        )
        target = [ts + 1, te + 1]
        argument = [
            self.shift(row['argument_start'], raw_ts, raw_te),
            self.shift(row['argument_end'], raw_ts, raw_te),
        ]
        return input_ids, attention_mask, target, argument, row


def generator_args(args):
    return SimpleNamespace(
        config_file=args.config_file,
        frame_info_file=args.frame_info_file,
        train_file=args.train_file,
        definition_prototype_file=args.definition_prototype_file,
        definition_frame_condition='soft_topk',
        definition_frame_topk=5,
        span_dim=256, role_dim=256, distance_dim=32, width_dim=16,
    )


def infer_train_logits(args, rows, frame_probs, role_names, role2idx, device):
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = GoldSpanDataset(
        rows, frame_probs, tokenizer, role2idx, max_rows=args.max_rows,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(role_collate, device=device, pad_token_id=tokenizer.pad_token_id),
    )
    cfg = generator_args(args)
    b = collect_logits(cfg, 'B', args.b_checkpoint, loader, len(role_names), device)
    c2 = collect_logits(cfg, 'C', args.c2_checkpoint, loader, len(role_names), device)
    d2 = collect_logits(cfg, 'D2', args.d2_checkpoint, loader, len(role_names), device)
    metadata = {
        'instance_indices': np.asarray([x['instance_index'] for x in dataset.items], np.int32),
        'sentence_ids': np.asarray([x['sentence_id'] for x in dataset.items], np.int64),
        'starts': np.asarray([x['argument_start'] for x in dataset.items], np.int32),
        'ends': np.asarray([x['argument_end'] for x in dataset.items], np.int32),
        'gold_role_ids': np.asarray([x['gold_role_id'] for x in dataset.items], np.int32),
        'frame_probs': np.stack([x['frame_prob'] for x in dataset.items]).astype(np.float32),
    }
    return b, c2, d2, metadata


def aligned_cache(path, reference=None):
    cache = np.load(path, allow_pickle=False)
    required = {'b_logits', 'c_logits', 'frame_probs', 'instance_indices', 'sentence_ids', 'starts', 'ends', 'role_names'}
    missing = required.difference(cache.files)
    if missing:
        raise ValueError(f'Pipeline cache {path} missing {sorted(missing)}')
    if reference is not None:
        for key in ('instance_indices', 'sentence_ids', 'starts', 'ends', 'role_names'):
            if not np.array_equal(cache[key], reference[key]):
                raise ValueError(f'Pipeline caches are not aligned at {key}: {path}')
    return cache


def build_generator_scores(args, b, c2, d2, frame_probs, membership):
    combined = (
        args.b_weight * zscore_rows(b.astype(np.float32))
        + args.c2_weight * zscore_rows(c2.astype(np.float32))
        + args.d2_weight * zscore_rows(d2.astype(np.float32))
    )
    top2 = np.partition(frame_probs, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    confidence = percentile_confidence(margin)
    prior_weight = args.prior_low + (
        args.prior_high - args.prior_low
    ) * np.power(confidence, args.prior_power)
    legal_prior = np.clip(frame_probs @ membership, args.prior_epsilon, 1.0)
    return (
        combined + prior_weight[:, None] * np.log(legal_prior)
    ).astype(np.float32), prior_weight.astype(np.float32)


def main():
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = read_json(args.split_file)
    frame_info, role_names, role2idx, frame_to_roles, augmented = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    frame_names = [row['frame_name'] for row in frame_info]
    frame2idx = {name: index for index, name in enumerate(frame_names)}
    membership = np.zeros((len(frame_names), len(role_names)), np.float32)
    for frame, role_ids in frame_to_roles.items():
        membership[frame2idx[frame], role_ids] = 1.0

    if args.mode == 'train_gold':
        validate_checkpoint_archives([
            args.task1_baseline_checkpoint,
            args.task1_prototype_checkpoint,
            args.b_checkpoint,
            args.c2_checkpoint,
            args.d2_checkpoint,
        ])
        frame_probs = task1_probabilities(args, rows, frame_info, device)
        b, c2, d2, metadata = infer_train_logits(
            args, rows, frame_probs, role_names, role2idx, device,
        )
        frame_probs = metadata['frame_probs']
        gold_role_ids = metadata.pop('gold_role_ids')
        gold_mask_full = None
        boundary_match = np.ones(len(gold_role_ids), dtype=bool)
        gold_tuples = len(gold_role_ids)
        gold_boundaries = len(gold_role_ids)
    else:
        c2_cache = aligned_cache(args.c2_pipeline_cache)
        d2_cache = aligned_cache(args.d2_pipeline_cache, c2_cache)
        b = c2_cache['b_logits'].astype(np.float32)
        c2 = c2_cache['c_logits'].astype(np.float32)
        d2 = d2_cache['c_logits'].astype(np.float32)
        frame_probs = c2_cache['frame_probs'].astype(np.float32)
        metadata = {
            key: c2_cache[key] for key in ('instance_indices', 'sentence_ids', 'starts', 'ends')
        }
        gold_mask_full, boundary_match, gold_tuples, gold_boundaries = build_gold_mask(
            rows, role2idx, metadata['instance_indices'], metadata['starts'],
            metadata['ends'], len(role_names),
        )

    generator, prior_weights = build_generator_scores(
        args, b, c2, d2, frame_probs, membership,
    )
    topk = min(int(args.candidate_topk), generator.shape[1])
    candidate_ids = np.argpartition(generator, -topk, axis=1)[:, -topk:]
    candidate_scores = np.take_along_axis(generator, candidate_ids, axis=1)
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    candidate_ids = np.take_along_axis(candidate_ids, order, axis=1).astype(np.int32)
    candidate_scores = np.take_along_axis(candidate_scores, order, axis=1).astype(np.float32)
    if args.mode == 'train_gold':
        gold_candidate_mask = candidate_ids == gold_role_ids[:, None]
    else:
        gold_candidate_mask = np.take_along_axis(gold_mask_full, candidate_ids, axis=1)
    top1_correct = gold_candidate_mask[:, 0]
    candidate_hit = gold_candidate_mask.any(axis=1)
    matched = boundary_match.astype(bool)
    # Audit whether the actual top-k group really crosses Frame boundaries.
    best_candidate_frames = np.full(candidate_ids.shape, -1, dtype=np.int32)
    for role_id in np.unique(candidate_ids):
        legal_frames = np.flatnonzero(membership[:, int(role_id)])
        row_indices, candidate_columns = np.where(candidate_ids == int(role_id))
        if not len(legal_frames):
            best_candidate_frames[row_indices, candidate_columns] = frame_probs[
                row_indices
            ].argmax(axis=1)
        else:
            local_best = frame_probs[row_indices][:, legal_frames].argmax(axis=1)
            best_candidate_frames[row_indices, candidate_columns] = legal_frames[local_best]
    cross_frame = np.asarray([
        len(set(row.tolist())) > 1 for row in best_candidate_frames
    ], dtype=bool)

    np.savez_compressed(
        output,
        candidate_role_ids=candidate_ids,
        candidate_generator_scores=candidate_scores,
        frame_probs=frame_probs.astype(np.float32),
        instance_indices=np.asarray(metadata['instance_indices'], np.int32),
        sentence_ids=np.asarray(metadata['sentence_ids'], np.int64),
        starts=np.asarray(metadata['starts'], np.int32),
        ends=np.asarray(metadata['ends'], np.int32),
        role_names=np.asarray(role_names),
        frame_names=np.asarray(frame_names),
        gold_candidate_mask=gold_candidate_mask.astype(bool),
        boundary_match=matched,
        prior_weights=prior_weights,
        gold_tuples=np.asarray([gold_tuples], np.int64),
        gold_boundaries=np.asarray([gold_boundaries], np.int64),
    )
    summary = {
        'mode': args.mode,
        'rows': int(len(candidate_ids)),
        'candidate_topk': topk,
        'boundary_matches': int(matched.sum()),
        'generator_top1_tp': int(top1_correct.sum()),
        'candidate_oracle_tp': int(candidate_hit.sum()),
        'top1_acc_on_matched': float(top1_correct.sum() / max(matched.sum(), 1)),
        'candidate_recall_on_matched': float(candidate_hit.sum() / max(matched.sum(), 1)),
        'train_usable_candidate_groups': int((candidate_hit & matched).sum()),
        'generator_wrong_but_candidate_hit': int(((~top1_correct) & candidate_hit & matched).sum()),
        'cross_frame_candidate_groups': int(cross_frame.sum()),
        'cross_frame_candidate_rate': float(cross_frame.mean()),
        'generator_weights': {
            'B': args.b_weight, 'C2': args.c2_weight, 'D2': args.d2_weight,
        },
        'adaptive_prior': {
            'low': args.prior_low, 'high': args.prior_high,
            'power': args.prior_power,
        },
        'augmented_frame_role_licenses': augmented,
        'run_config': vars(args),
    }
    output.with_suffix('.summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
