#!/usr/bin/env python3
"""Candidate-limited D3 reranking for the instance-safe Task3 Dev pipeline.

The expensive Definition Query model never scans all 1009 roles here.  A
Task3 generator cache first supplies top-k role candidates after the Task1
soft legal-role prior.  D3 scores one or more most probable licensed
Frame--FE definitions per candidate.  Post-hoc z-score fusion and a
generator-margin gate are searched without repeating BERT inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from tasks.common.experiment_utils import REPO_ROOT
from tasks.task3.dataset_experiments import build_augmented_frame_roles
from tasks.task3.verifier.dataset import (
    MARKERS,
    encode_with_target_argument_markers,
)

from .dataset import build_definition_maps
from .model import DefinitionQueryCrossEncoder


TASK_DIR = REPO_ROOT / 'tasks' / 'task3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_cache', required=True)
    parser.add_argument('--generator_key', default='c_logits')
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--dev_file', default=str(REPO_ROOT / 'dataset' / 'cfn-dev.json'))
    parser.add_argument('--train_file', default=str(REPO_ROOT / 'dataset' / 'cfn-train.json'))
    parser.add_argument('--frame_info_file', default=str(REPO_ROOT / 'dataset' / 'frame_info.json'))
    parser.add_argument('--pretrained_model_dir', default=str(REPO_ROOT / 'chinese_bert_wwm_ext'))
    parser.add_argument('--candidate_topk', default=5, type=int)
    parser.add_argument('--generator_prior_weight', default=3.0, type=float)
    parser.add_argument('--pair_frames_per_role', default=1, type=int)
    parser.add_argument('--projection_dim', default=512, type=int)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--group_batch_size', default=2, type=int)
    parser.add_argument('--pair_batch_size', default=16, type=int)
    parser.add_argument('--alpha_values', default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.75,1,1.25,1.5,2')
    parser.add_argument('--rerank_fractions', default='0.1,0.2,0.3,0.4,0.5,0.6,0.75,1')
    parser.add_argument('--prior_epsilon', default=1e-8, type=float)
    parser.add_argument('--max_rows', default=0, type=int)
    parser.add_argument('--reuse_score_cache', action='store_true')
    parser.add_argument(
        '--output_dir',
        default=str(TASK_DIR / 'analysis' / 'task3-D3-C-candidate-reranker'),
    )
    return parser.parse_args()


def read_json(path):
    with open(path, 'r', encoding='utf8') as handle:
        return json.load(handle)


def zscore_rows(values):
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / values.std(axis=1, keepdims=True).clip(min=1e-6)


def build_gold_mask(dev_rows, role2idx, instance_indices, starts, ends, num_roles):
    gold_by_boundary = {}
    gold_tuples = 0
    gold_boundaries = 0
    for instance_index, row in enumerate(dev_rows):
        for span in row.get('cfn_spans', []):
            boundary = (instance_index, int(span['start']), int(span['end']))
            if boundary not in gold_by_boundary:
                gold_by_boundary[boundary] = set()
                gold_boundaries += 1
            role_id = int(role2idx[span['fe_name']])
            if role_id not in gold_by_boundary[boundary]:
                gold_by_boundary[boundary].add(role_id)
                gold_tuples += 1
    mask = np.zeros((len(instance_indices), num_roles), dtype=bool)
    boundary_match = np.zeros(len(instance_indices), dtype=bool)
    for index, key in enumerate(zip(instance_indices, starts, ends)):
        roles = gold_by_boundary.get(tuple(int(value) for value in key), ())
        if roles:
            boundary_match[index] = True
            mask[index, list(roles)] = True
    return mask, boundary_match, gold_tuples, gold_boundaries


def prf(tp, predicted, gold):
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(prediction, gold_mask, boundary_match, gold_tuples, gold_boundaries):
    indices = np.arange(len(prediction))
    correct = gold_mask[indices, prediction]
    tp = int(correct.sum())
    precision, recall, f1 = prf(tp, len(prediction), gold_tuples)
    boundary_tp = int(boundary_match.sum())
    bp, br, bf = prf(boundary_tp, len(prediction), gold_boundaries)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'predicted': int(len(prediction)),
        'gold': int(gold_tuples),
        'boundary_precision': bp,
        'boundary_recall': br,
        'boundary_f1': bf,
        'boundary_tp': boundary_tp,
        'gold_boundaries': int(gold_boundaries),
        'role_acc_on_matched_boundaries': (
            tp / boundary_tp if boundary_tp else 0.0
        ),
    }, correct


class PipelineCandidateDataset(Dataset):
    def __init__(
        self, dev_rows, tokenizer, pair_text, global_text, role_names,
        frame_names, frame_role_legal, frame_probs, instance_indices,
        starts, ends, candidate_role_ids, pair_frames_per_role=1,
        max_length=512,
    ):
        self.dev_rows = dev_rows
        self.tokenizer = tokenizer
        self.pair_text = pair_text
        self.global_text = global_text
        self.role_names = role_names
        self.frame_names = frame_names
        self.frame_role_legal = frame_role_legal
        self.frame_probs = frame_probs
        self.instance_indices = instance_indices
        self.starts = starts
        self.ends = ends
        self.candidate_role_ids = candidate_role_ids
        self.pair_frames_per_role = int(pair_frames_per_role)
        self.max_length = int(max_length)
        if self.pair_frames_per_role < 1:
            raise ValueError('pair_frames_per_role must be positive.')
        self.role_to_frames = [
            np.flatnonzero(frame_role_legal[:, role_id]).astype(np.int64)
            for role_id in range(frame_role_legal.shape[1])
        ]

    def __len__(self):
        return len(self.instance_indices)

    def _definition(self, frame_name, role_id):
        value = self.pair_text.get((frame_name, int(role_id)))
        if value is not None:
            return value
        return (
            f'框架：{frame_name}。'
            + self.global_text.get(
                int(role_id), f'框架元素：{self.role_names[int(role_id)]}。'
            )
        )

    def _best_frames(self, row_index, role_id):
        legal_frames = self.role_to_frames[int(role_id)]
        probabilities = self.frame_probs[row_index]
        if not len(legal_frames):
            legal_frames = np.asarray([int(probabilities.argmax())], dtype=np.int64)
        order = legal_frames[np.argsort(probabilities[legal_frames])[::-1]]
        chosen = order[: self.pair_frames_per_role].tolist()
        while len(chosen) < self.pair_frames_per_role:
            chosen.append(chosen[-1])
        return chosen

    def _encode(self, source, start, end, frame_id, role_id):
        target = source['target'][-1]
        sentence_ids, _, positions = encode_with_target_argument_markers(
            self.tokenizer,
            source['text'],
            int(target['start']), int(target['end']),
            int(start), int(end),
            max_length=self.max_length,
        )
        definition = self._definition(self.frame_names[int(frame_id)], role_id)
        definition_ids = self.tokenizer.encode(
            definition, add_special_tokens=False
        )
        available = self.max_length - len(sentence_ids) - 1
        if available <= 0:
            raise ValueError(
                f'No room for D3 definition instance={source["sentence_id"]}.'
            )
        definition_ids = definition_ids[:available]
        input_ids = sentence_ids + definition_ids + [self.tokenizer.sep_token_id]
        return {
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'token_type_ids': (
                [0] * len(sentence_ids) + [1] * (len(definition_ids) + 1)
            ),
            'marker_positions': positions,
            'frame_id': int(frame_id),
            'role_id': int(role_id),
        }

    def __getitem__(self, index):
        instance_index = int(self.instance_indices[index])
        source = self.dev_rows[instance_index]
        pairs = []
        pair_frame_ids = []
        for role_id in self.candidate_role_ids[index]:
            frames = self._best_frames(index, int(role_id))
            pair_frame_ids.append(frames)
            for frame_id in frames:
                pairs.append(self._encode(
                    source, self.starts[index], self.ends[index],
                    frame_id, int(role_id),
                ))
        return {
            'pairs': pairs,
            'candidate_role_ids': self.candidate_role_ids[index].tolist(),
            'pair_frame_ids': pair_frame_ids,
            'row_index': int(index),
        }


def collate_candidates(items, pad_token_id=0):
    pairs = [pair for item in items for pair in item['pairs']]
    max_len = max(len(pair['input_ids']) for pair in pairs)
    pad = lambda values, value: values + [value] * (max_len - len(values))
    marker_positions = {
        name: torch.tensor([
            pair['marker_positions'][name] for pair in pairs
        ], dtype=torch.long)
        for name in MARKERS
    }
    return {
        'input_ids': torch.tensor([
            pad(pair['input_ids'], pad_token_id) for pair in pairs
        ], dtype=torch.long),
        'attention_mask': torch.tensor([
            pad(pair['attention_mask'], 0) for pair in pairs
        ], dtype=torch.long),
        'token_type_ids': torch.tensor([
            pad(pair['token_type_ids'], 0) for pair in pairs
        ], dtype=torch.long),
        'marker_positions': marker_positions,
        'group_pair_sizes': [len(item['pairs']) for item in items],
        'candidate_role_ids': [item['candidate_role_ids'] for item in items],
        'pair_frame_ids': [item['pair_frame_ids'] for item in items],
        'row_indices': [item['row_index'] for item in items],
    }


def score_pairs(model, batch, device, pair_batch_size):
    values = []
    total = len(batch['input_ids'])
    for start in range(0, total, pair_batch_size):
        end = min(start + pair_batch_size, total)
        values.append(model(
            input_ids=batch['input_ids'][start:end].to(device),
            attention_mask=batch['attention_mask'][start:end].to(device),
            token_type_ids=batch['token_type_ids'][start:end].to(device),
            marker_positions={
                name: positions[start:end].to(device)
                for name, positions in batch['marker_positions'].items()
            },
        ).detach().cpu().to(torch.float32))
    return torch.cat(values, dim=0)


def create_candidate_score_cache(args, output_file):
    pipeline = np.load(args.pipeline_cache, allow_pickle=False)
    required = {
        args.generator_key, 'frame_probs', 'instance_indices', 'sentence_ids',
        'starts', 'ends', 'role_names',
    }
    missing = required.difference(pipeline.files)
    if missing:
        raise ValueError(f'Pipeline cache missing fields: {sorted(missing)}')
    logits = pipeline[args.generator_key].astype(np.float32)
    frame_probs = pipeline['frame_probs'].astype(np.float32)
    instance_indices = pipeline['instance_indices'].astype(np.int64)
    sentence_ids = pipeline['sentence_ids'].astype(np.int64)
    starts = pipeline['starts'].astype(np.int64)
    ends = pipeline['ends'].astype(np.int64)
    role_names = pipeline['role_names'].tolist()
    if args.max_rows > 0:
        limit = min(args.max_rows, len(logits))
        logits, frame_probs = logits[:limit], frame_probs[:limit]
        instance_indices, sentence_ids = instance_indices[:limit], sentence_ids[:limit]
        starts, ends = starts[:limit], ends[:limit]

    frame_info, ontology_roles, role2idx, frame_to_role_ids, augmented = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    if ontology_roles != role_names:
        raise ValueError('Pipeline Role order differs from frame ontology.')
    frame_names = [row['frame_name'] for row in frame_info]
    frame2idx = {name: index for index, name in enumerate(frame_names)}
    membership = np.zeros((len(frame_names), len(role_names)), dtype=np.float32)
    for frame, role_ids in frame_to_role_ids.items():
        membership[frame2idx[frame], role_ids] = 1.0
    legal_prior = np.clip(
        frame_probs @ membership, args.prior_epsilon, 1.0
    )
    # Keep candidate generation exactly consistent with
    # evaluate_full_pipeline.py.  Normalisation is applied only after the
    # fixed top-k has been cached for cross-model D3 fusion.
    generator_full_scores = (
        logits + float(args.generator_prior_weight) * np.log(legal_prior)
    )
    candidate_topk = min(int(args.candidate_topk), logits.shape[1])
    partial = np.argpartition(
        generator_full_scores, -candidate_topk, axis=1
    )[:, -candidate_topk:]
    partial_scores = np.take_along_axis(generator_full_scores, partial, axis=1)
    order = np.argsort(partial_scores, axis=1)[:, ::-1]
    candidate_role_ids = np.take_along_axis(partial, order, axis=1).astype(np.int64)
    candidate_generator_scores = np.take_along_axis(
        generator_full_scores, candidate_role_ids, axis=1
    ).astype(np.float32)

    dev_rows = read_json(args.dev_file)
    for index, instance_index in enumerate(instance_indices):
        source = dev_rows[int(instance_index)]
        if int(source['sentence_id']) != int(sentence_ids[index]):
            raise ValueError(f'Pipeline cache SID mismatch at row {index}.')

    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_dir, do_lower_case=True
    )
    pair_text, global_text, _ = build_definition_maps(frame_info, role2idx)
    dataset = PipelineCandidateDataset(
        dev_rows=dev_rows,
        tokenizer=tokenizer,
        pair_text=pair_text,
        global_text=global_text,
        role_names=role_names,
        frame_names=frame_names,
        frame_role_legal=membership,
        frame_probs=frame_probs,
        instance_indices=instance_indices,
        starts=starts,
        ends=ends,
        candidate_role_ids=candidate_role_ids,
        pair_frames_per_role=args.pair_frames_per_role,
        max_length=args.max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.group_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: collate_candidates(
            items, pad_token_id=tokenizer.pad_token_id
        ),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BertConfig.from_pretrained(args.pretrained_model_dir)
    model = DefinitionQueryCrossEncoder(
        config, projection_dim=args.projection_dim
    )
    state = torch.load(args.checkpoint_path, map_location='cpu')
    message = model.load_state_dict(state, strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(
            f'D3 checkpoint mismatch: missing={message.missing_keys} '
            f'unexpected={message.unexpected_keys}'
        )
    model = model.to(device).eval()
    d3_scores = np.zeros(candidate_role_ids.shape, dtype=np.float32)
    pair_frame_ids = np.zeros(
        candidate_role_ids.shape + (args.pair_frames_per_role,), dtype=np.int32
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc='D3-pipeline-rerank-score'):
            scores = score_pairs(
                model, batch, device=device,
                pair_batch_size=args.pair_batch_size,
            )
            offset = 0
            for local_index, size in enumerate(batch['group_pair_sizes']):
                row_index = batch['row_indices'][local_index]
                group = scores[offset: offset + size].numpy().reshape(
                    candidate_topk, args.pair_frames_per_role
                )
                d3_scores[row_index] = group.max(axis=1)
                pair_frame_ids[row_index] = np.asarray(
                    batch['pair_frame_ids'][local_index], dtype=np.int32
                )
                offset += size
            if offset != len(scores):
                raise ValueError('D3 reranker group sizes do not cover scores.')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        candidate_role_ids=candidate_role_ids,
        candidate_generator_scores=candidate_generator_scores,
        candidate_d3_scores=d3_scores,
        pair_frame_ids=pair_frame_ids,
        frame_probs=frame_probs,
        instance_indices=instance_indices,
        sentence_ids=sentence_ids,
        starts=starts,
        ends=ends,
        role_names=np.asarray(role_names),
    )
    return {
        'device': str(device),
        'num_rows': len(dataset),
        'candidate_topk': candidate_topk,
        'pairs_per_role': int(args.pair_frames_per_role),
        'total_cross_encoder_pairs': int(
            len(dataset) * candidate_topk * args.pair_frames_per_role
        ),
        'augmented_frame_role_licenses': augmented,
    }


def analyze_score_cache(args, score_file, scoring_metadata=None):
    cache = np.load(score_file, allow_pickle=False)
    candidates = cache['candidate_role_ids'].astype(np.int64)
    generator_scores = cache['candidate_generator_scores'].astype(np.float32)
    d3_scores = cache['candidate_d3_scores'].astype(np.float32)
    instance_indices = cache['instance_indices'].astype(np.int64)
    starts = cache['starts'].astype(np.int64)
    ends = cache['ends'].astype(np.int64)
    role_names = cache['role_names'].tolist()
    _, ontology_roles, role2idx, _, augmented = build_augmented_frame_roles(
        args.frame_info_file, args.train_file
    )
    if role_names != ontology_roles:
        raise ValueError('D3 score-cache Role order differs from ontology.')
    dev_rows = read_json(args.dev_file)
    gold_mask, boundary_match, gold_tuples, gold_boundaries = build_gold_mask(
        dev_rows, role2idx, instance_indices, starts, ends, len(role_names)
    )

    generator_norm = zscore_rows(generator_scores)
    d3_norm = zscore_rows(d3_scores)
    baseline_prediction = candidates[:, 0]
    baseline_metrics, baseline_correct = evaluate(
        baseline_prediction, gold_mask, boundary_match,
        gold_tuples, gold_boundaries,
    )
    candidate_hit = np.take_along_axis(
        gold_mask, candidates, axis=1
    ).any(axis=1)
    oracle_metrics, _ = evaluate(
        candidates[np.arange(len(candidates)), np.argmax(
            np.take_along_axis(gold_mask, candidates, axis=1), axis=1
        )],
        gold_mask, boundary_match, gold_tuples, gold_boundaries,
    )
    # The evaluate call above chooses index zero when no gold candidate; make
    # its TP explicit so the candidate ceiling cannot be misread.
    oracle_tp = int(candidate_hit.sum())
    op, ore, of1 = prf(oracle_tp, len(candidates), gold_tuples)
    oracle_metrics.update({'precision': op, 'recall': ore, 'f1': of1, 'tp': oracle_tp})

    if generator_scores.shape[1] > 1:
        generator_margin = generator_scores[:, 0] - generator_scores[:, 1]
    else:
        generator_margin = np.full(len(generator_scores), np.inf, np.float32)
    alphas = [float(value) for value in args.alpha_values.split(',') if value]
    fractions = [
        float(value) for value in args.rerank_fractions.split(',') if value
    ]
    grid = []
    best = None
    best_prediction = baseline_prediction
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError('rerank_fractions values must be in (0,1].')
        threshold = float(np.quantile(generator_margin, fraction))
        selected = generator_margin <= threshold
        for alpha in alphas:
            fused = generator_norm.copy()
            fused[selected] += alpha * d3_norm[selected]
            candidate_index = fused.argmax(axis=1)
            prediction = candidates[np.arange(len(candidates)), candidate_index]
            metrics, correct = evaluate(
                prediction, gold_mask, boundary_match,
                gold_tuples, gold_boundaries,
            )
            row = {
                'rerank_fraction': fraction,
                'margin_threshold': threshold,
                'alpha': alpha,
                'selected': int(selected.sum()),
                'changed': int((prediction != baseline_prediction).sum()),
                'corrected_errors': int((correct & ~baseline_correct).sum()),
                'damaged_correct': int((~correct & baseline_correct).sum()),
                **metrics,
            }
            grid.append(row)
            if best is None or (row['f1'], row['tp']) > (best['f1'], best['tp']):
                best = row
                best_prediction = prediction.copy()

    output_dir = Path(args.output_dir)
    with (output_dir / 'rerank_grid.csv').open(
        'w', encoding='utf8', newline=''
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    prediction_rows = []
    for index, role_id in enumerate(best_prediction):
        prediction_rows.append({
            'instance_index': int(instance_indices[index]),
            'sentence_id': int(cache['sentence_ids'][index]),
            'start': int(starts[index]),
            'end': int(ends[index]),
            'role_id': int(role_id),
            'role': role_names[int(role_id)],
        })
    with (output_dir / 'best_reranked_predictions.json').open(
        'w', encoding='utf8'
    ) as handle:
        json.dump(prediction_rows, handle, ensure_ascii=False, indent=2)
    summary = {
        'baseline_generator': baseline_metrics,
        'candidate_oracle_ceiling': {
            **oracle_metrics,
            'candidate_hit_on_matched_boundaries': (
                oracle_tp / int(boundary_match.sum())
                if boundary_match.any() else 0.0
            ),
        },
        'best_d3_reranker': best,
        'num_rows': int(len(candidates)),
        'candidate_topk': int(candidates.shape[1]),
        'scoring_metadata': scoring_metadata or {},
        'augmented_frame_role_licenses': augmented,
        'run_config': vars(args),
        'note': (
            'D3 searches only generator top-k candidates. The margin gate '
            'reranks low-confidence generator rows; all metrics use '
            'instance-safe exact Task3 tuples.'
        ),
    }
    with (output_dir / 'summary.json').open('w', encoding='utf8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_file = output_dir / 'd3_candidate_scores.npz'
    metadata = None
    if args.reuse_score_cache:
        if not score_file.exists():
            raise FileNotFoundError(
                f'--reuse_score_cache requested but cache is absent: {score_file}'
            )
    else:
        if not Path(args.checkpoint_path).exists():
            raise FileNotFoundError(
                f'D3 checkpoint is missing: {args.checkpoint_path}. '
                'Transfer the Epoch-1 task3_best.bin before scoring.'
            )
        metadata = create_candidate_score_cache(args, score_file)
    analyze_score_cache(args, score_file, scoring_metadata=metadata)


if __name__ == '__main__':
    main()
