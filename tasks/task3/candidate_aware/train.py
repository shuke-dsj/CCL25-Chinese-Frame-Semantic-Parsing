#!/usr/bin/env python3
"""Train/evaluate Task3 D4 candidate-aware Definition reranker."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, get_linear_schedule_with_warmup

from tasks.common.experiment_utils import (
    ExperimentLogger,
    REPO_ROOT,
    save_checkpoint,
    set_seed,
)
from tasks.task3.dataset_experiments import build_augmented_frame_roles
from .dataset import CandidateAwareDataset, collate_candidate_groups
from .model import CandidateAwareDefinitionReranker


TASK_DIR = REPO_ROOT / 'tasks' / 'task3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--result_dir', default=None)
    parser.add_argument('--train_cache', default=str(
        TASK_DIR / 'analysis/task3-D4-candidate-cache/train_gold_top5.npz'
    ))
    parser.add_argument('--dev_cache', default=str(
        TASK_DIR / 'analysis/task3-D4-candidate-cache/dev_pipeline_top5.npz'
    ))
    parser.add_argument('--train_file', default=str(REPO_ROOT / 'dataset/cfn-train.json'))
    parser.add_argument('--dev_file', default=str(REPO_ROOT / 'dataset/cfn-dev.json'))
    parser.add_argument('--frame_info_file', default=str(REPO_ROOT / 'dataset/frame_info.json'))
    parser.add_argument('--pretrained_model_dir', default=str(REPO_ROOT / 'chinese_bert_wwm_ext'))
    parser.add_argument('--encoder_initialization_checkpoint', required=True,
                        help='D4-A: C checkpoint; D4-B: D3 checkpoint. Only bert.* is loaded.')
    parser.add_argument('--initialization_label', required=True,
                        choices=['C-BERT-clean-head', 'D3-BERT-clean-head'])
    parser.add_argument('--num_train_epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Candidate groups per loader batch.')
    parser.add_argument('--eval_batch_size', default=2, type=int)
    parser.add_argument('--pair_batch_size', default=16, type=int)
    parser.add_argument('--accumulate_gradients', default=1, type=int)
    parser.add_argument('--pair_frames_per_role', default=2, type=int)
    parser.add_argument('--projection_dim', default=512, type=int)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--encoder_lr', default=1e-5, type=float)
    parser.add_argument('--head_lr', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--definition_ce_weight', default=0.3, type=float)
    parser.add_argument('--distill_weight', default=0.1, type=float)
    parser.add_argument('--distill_temperature', default=1.0, type=float)
    parser.add_argument('--hard_example_extra_weight', default=2.0, type=float,
                        help='Generator-wrong groups receive weight 1+this value.')
    parser.add_argument('--train_generator_perturb_probability', default=0.35, type=float)
    parser.add_argument('--train_generator_perturb_min_margin', default=0.1, type=float)
    parser.add_argument('--train_generator_perturb_max_margin', default=0.6, type=float)
    parser.add_argument('--freeze_bottom_layers', default=8, type=int)
    parser.add_argument('--unfreeze_all_epoch', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--max_train_groups', default=0, type=int)
    parser.add_argument('--max_dev_groups', default=0, type=int)
    return parser.parse_args()


def to_device(batch, device):
    output = dict(batch)
    for key in (
        'input_ids', 'attention_mask', 'token_type_ids', 'generator_scores',
        'frame_log_weights', 'group_features', 'gold_mask',
        'candidate_role_ids', 'boundary_match', 'generator_perturbed',
    ):
        output[key] = output[key].to(device)
    output['marker_positions'] = {
        name: value.to(device)
        for name, value in batch['marker_positions'].items()
    }
    return output


def multi_target_nll(scores, gold_mask):
    if not gold_mask.any(dim=1).all():
        raise ValueError('Every D4 training group must contain a gold candidate.')
    log_probability = F.log_softmax(scores, dim=1)
    masked = log_probability.masked_fill(~gold_mask, float('-inf'))
    return -torch.logsumexp(masked, dim=1)


def objective(outputs, batch, args):
    row_index = torch.arange(
        len(batch['generator_scores']), device=batch['generator_scores'].device
    )
    generator_index = batch['generator_scores'].argmax(dim=1)
    generator_correct = batch['gold_mask'][row_index, generator_index]
    sample_weight = 1.0 + float(args.hard_example_extra_weight) * (~generator_correct).float()
    final_per_sample = multi_target_nll(outputs['final_scores'], batch['gold_mask'])
    definition_per_sample = multi_target_nll(
        outputs['definition_scores'], batch['gold_mask']
    )
    final_loss = (final_per_sample * sample_weight).sum() / sample_weight.sum()
    definition_loss = (
        definition_per_sample * sample_weight
    ).sum() / sample_weight.sum()

    temperature = float(args.distill_temperature)
    teacher = F.softmax(batch['generator_scores'] / temperature, dim=1)
    student_log = F.log_softmax(outputs['final_scores'] / temperature, dim=1)
    distill_per_sample = F.kl_div(
        student_log, teacher, reduction='none'
    ).sum(dim=1) * temperature ** 2
    # Preserve the base primarily where it is already correct.  Incorrect
    # generator groups are free to move under the supervised ranking loss.
    correct_weight = generator_correct.float()
    distill_loss = (
        (distill_per_sample * correct_weight).sum()
        / correct_weight.sum().clamp_min(1.0)
    )
    loss = (
        final_loss
        + float(args.definition_ce_weight) * definition_loss
        + float(args.distill_weight) * distill_loss
    )
    return loss, final_loss, definition_loss, distill_loss


def forward_model(model, batch, pair_batch_size):
    return model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        token_type_ids=batch['token_type_ids'],
        marker_positions=batch['marker_positions'],
        generator_scores=batch['generator_scores'],
        frame_log_weights=batch['frame_log_weights'],
        group_features=batch['group_features'],
        pair_batch_size=pair_batch_size,
    )


def prf(tp, predicted, gold):
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@torch.no_grad()
def evaluate(model, loader, dataset, device, pair_batch_size):
    model.eval()
    totals = defaultdict(float)
    score_rows = []
    prediction_rows = []
    for raw_batch in tqdm(loader, desc='D4-candidate-aware-eval'):
        batch = to_device(raw_batch, device)
        outputs = forward_model(model, batch, pair_batch_size)
        final_index = outputs['final_scores'].argmax(dim=1)
        generator_index = batch['generator_scores'].argmax(dim=1)
        row_index = torch.arange(len(final_index), device=device)
        final_correct = batch['gold_mask'][row_index, final_index]
        generator_correct = batch['gold_mask'][row_index, generator_index]
        changed = final_index != generator_index
        totals['tp'] += int(final_correct.sum())
        totals['generator_tp'] += int(generator_correct.sum())
        totals['corrected'] += int(((~generator_correct) & final_correct).sum())
        totals['damaged'] += int((generator_correct & (~final_correct)).sum())
        totals['changed'] += int(changed.sum())
        totals['candidate_hit'] += int(batch['gold_mask'].any(dim=1).sum())
        totals['boundary_match'] += int(batch['boundary_match'].sum())
        totals['gate_sum'] += float(outputs['gate'].sum())
        totals['groups'] += len(final_index)

        final_cpu = outputs['final_scores'].detach().cpu().float().numpy()
        definition_cpu = outputs['definition_scores'].detach().cpu().float().numpy()
        candidates_cpu = batch['candidate_role_ids'].detach().cpu().numpy()
        final_index_cpu = final_index.detach().cpu().numpy()
        generator_index_cpu = generator_index.detach().cpu().numpy()
        for index, row in enumerate(raw_batch['rows']):
            candidate_ids = candidates_cpu[index]
            predicted_role = int(candidate_ids[final_index_cpu[index]])
            prediction_rows.append({
                'cache_index': int(row['cache_index']),
                'instance_index': int(row['instance_index']),
                'sentence_id': int(row['sentence_id']),
                'start': int(row['start']), 'end': int(row['end']),
                'predicted_role_id': predicted_role,
                'generator_role_id': int(candidate_ids[generator_index_cpu[index]]),
                'changed': bool(final_index_cpu[index] != generator_index_cpu[index]),
                'correct': bool(final_correct[index].item()),
                'generator_correct': bool(generator_correct[index].item()),
                'gate': float(outputs['gate'][index].detach().cpu()),
            })
            score_rows.append((
                int(row['cache_index']), candidate_ids.astype(np.int32),
                final_cpu[index], definition_cpu[index],
            ))

    tp = int(totals['tp'])
    predicted = int(totals['groups'])
    precision, recall, f1 = prf(tp, predicted, dataset.gold_tuples)
    generator_precision, generator_recall, generator_f1 = prf(
        int(totals['generator_tp']), predicted, dataset.gold_tuples,
    )
    metrics = {
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1,
        'val_tp': tp,
        'val_predicted': predicted,
        'val_gold': dataset.gold_tuples,
        'val_role_acc_on_matched': tp / max(int(totals['boundary_match']), 1),
        'val_generator_f1': generator_f1,
        'val_generator_tp': int(totals['generator_tp']),
        'val_candidate_oracle_tp': int(totals['candidate_hit']),
        'val_corrected_errors': int(totals['corrected']),
        'val_damaged_correct': int(totals['damaged']),
        'val_changed': int(totals['changed']),
        'val_gate_mean': totals['gate_sum'] / max(totals['groups'], 1),
        'val_residual_scale': float(outputs['residual_scale'].detach().cpu()),
    }
    score_rows.sort(key=lambda value: value[0])
    score_cache = {
        'cache_indices': np.asarray([row[0] for row in score_rows], np.int32),
        'candidate_role_ids': np.stack([row[1] for row in score_rows]),
        'final_scores': np.stack([row[2] for row in score_rows]).astype(np.float32),
        'definition_scores': np.stack([row[3] for row in score_rows]).astype(np.float32),
    }
    return metrics, prediction_rows, score_cache


def load_model(args, device):
    config = BertConfig.from_pretrained(args.pretrained_model_dir)
    model = CandidateAwareDefinitionReranker(
        config,
        projection_dim=args.projection_dim,
        pair_frames_per_role=args.pair_frames_per_role,
    )
    path = Path(args.encoder_initialization_checkpoint)
    with path.open('rb') as handle:
        state = torch.load(handle, map_location='cpu')
    bert_state = {
        name[len('bert.'):]: value
        for name, value in state.items()
        if name.startswith('bert.')
    }
    if not bert_state:
        raise ValueError(f'No bert.* tensors found in initialization checkpoint: {path}')
    message = model.pair_encoder.bert.load_state_dict(bert_state, strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(f'D4 BERT initialization mismatch: {message}')
    print(
        f'[D4 init] {args.initialization_label}: loaded {len(bert_state)} bert.* '
        f'tensors from {path}; candidate projection/gate/head are random.'
    )
    return model.to(device)


def set_bottom_layer_freeze(model, count, frozen):
    bert = model.pair_encoder.bert
    count = max(0, min(int(count), len(bert.encoder.layer)))
    # Keep token embeddings trainable: D4-A must learn the four previously
    # unused Target/Argument marker embeddings from scratch.  Only the bottom
    # Transformer blocks are staged/frozen.
    modules = list(bert.encoder.layer[:count])
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = not frozen
    return count


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_dir, do_lower_case=True,
    )
    frame_info, role_names, role2idx, frame_to_roles, augmented = (
        build_augmented_frame_roles(args.frame_info_file, args.train_file)
    )
    common = dict(
        frame_info=frame_info,
        frame_to_role_ids=frame_to_roles,
        tokenizer=tokenizer,
        role2idx=role2idx,
        role_names=role_names,
        pair_frames_per_role=args.pair_frames_per_role,
        max_length=args.max_length,
    )
    train_dataset = CandidateAwareDataset(
        args.train_file, args.train_cache, train_mode=True,
        max_groups=args.max_train_groups,
        generator_perturb_probability=args.train_generator_perturb_probability,
        generator_perturb_min_margin=args.train_generator_perturb_min_margin,
        generator_perturb_max_margin=args.train_generator_perturb_max_margin,
        perturb_seed=args.seed,
        **common,
    )
    dev_dataset = CandidateAwareDataset(
        args.dev_file, args.dev_cache, train_mode=False,
        max_groups=args.max_dev_groups, **common,
    )
    collate = partial(
        collate_candidate_groups, pad_token_id=tokenizer.pad_token_id,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=0, collate_fn=collate,
    )
    model = load_model(args, device)
    frozen_count = set_bottom_layer_freeze(
        model, args.freeze_bottom_layers, frozen=True,
    )
    encoder_parameters = list(model.pair_encoder.bert.parameters())
    head_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith('pair_encoder.bert.')
    ]
    optimizer = torch.optim.AdamW([
        {'params': encoder_parameters, 'lr': args.encoder_lr,
         'initial_lr': args.encoder_lr, 'weight_decay': args.weight_decay},
        {'params': head_parameters, 'lr': args.head_lr,
         'initial_lr': args.head_lr, 'weight_decay': args.weight_decay},
    ])
    updates_per_epoch = math.ceil(
        len(train_loader) / max(args.accumulate_gradients, 1)
    )
    total_steps = updates_per_epoch * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_proportion),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.fp16 and device.type == 'cuda'
    )
    fields = [
        'epoch', 'train_loss', 'train_final_ce', 'train_definition_ce',
        'train_distill', 'train_synthetic_hard_rate',
        'val_precision', 'val_recall', 'val_f1', 'val_tp',
        'val_predicted', 'val_gold', 'val_role_acc_on_matched',
        'val_generator_f1', 'val_generator_tp', 'val_candidate_oracle_tp',
        'val_corrected_errors', 'val_damaged_correct', 'val_changed',
        'val_gate_mean', 'val_residual_scale', 'best_val_f1',
        'epoch_seconds', 'encoder_lr', 'head_lr', 'peak_gpu_memory_mb',
    ]
    logger = ExperimentLogger(args, 'task3', fields)
    summary = {
        'architecture': 'candidate_aware_cross_frame_definition_residual_reranker',
        'initialization': args.initialization_label,
        'encoder_checkpoint': args.encoder_initialization_checkpoint,
        'new_components_randomly_initialized': [
            'definition_projection', 'cross_frame_aggregator',
            'residual_gate', 'residual_scale',
        ],
        'train_groups_total': int(len(train_dataset.candidate_role_ids)),
        'train_groups_usable': len(train_dataset),
        'dev_pipeline_groups': len(dev_dataset),
        'candidate_topk': int(train_dataset.candidate_role_ids.shape[1]),
        'pair_frames_per_role': args.pair_frames_per_role,
        'initially_frozen_bottom_bert_layers': frozen_count,
        'unfreeze_all_epoch': args.unfreeze_all_epoch,
        'train_generator_perturbation': {
            'probability': args.train_generator_perturb_probability,
            'min_margin': args.train_generator_perturb_min_margin,
            'max_margin': args.train_generator_perturb_max_margin,
            'policy': 'prefer strongest cross-frame non-gold candidate',
        },
        'augmented_frame_role_licenses': augmented,
        'device': str(device),
    }
    (logger.result_dir / 'candidate_aware_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8',
    )
    print(json.dumps({'D4_config': summary}, ensure_ascii=False))

    best_f1 = -1.0
    optimizer.zero_grad()
    for epoch in range(1, args.num_train_epochs + 1):
        start_time = time.perf_counter()
        train_dataset.set_epoch(epoch)
        if epoch == args.unfreeze_all_epoch:
            set_bottom_layer_freeze(model, args.freeze_bottom_layers, frozen=False)
            print(f'[D4 schedule] epoch={epoch}: unfroze all BERT layers.')
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        model.train()
        totals = defaultdict(float)
        for step, raw_batch in enumerate(tqdm(train_loader, desc=f'D4-epoch_{epoch}')):
            batch = to_device(raw_batch, device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = forward_model(model, batch, args.pair_batch_size)
                loss, final_ce, definition_ce, distill = objective(
                    outputs, batch, args,
                )
                scaled_loss = loss / max(args.accumulate_gradients, 1)
            scaler.scale(scaled_loss).backward()
            totals['loss'] += float(loss.detach())
            totals['final'] += float(final_ce.detach())
            totals['definition'] += float(definition_ce.detach())
            totals['distill'] += float(distill.detach())
            totals['perturbed'] += int(batch['generator_perturbed'].sum())
            totals['train_groups'] += len(batch['generator_perturbed'])
            should_step = (
                (step + 1) % max(args.accumulate_gradients, 1) == 0
                or step + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_ran = (
                    not scaler.is_enabled() or scaler.get_scale() >= scale_before
                )
                if optimizer_ran:
                    scheduler.step()
                optimizer.zero_grad()
            if args.log_interval and (step + 1) % args.log_interval == 0:
                print(
                    f'[D4 train] epoch={epoch} step={step + 1}/{len(train_loader)} '
                    f'loss={totals["loss"] / (step + 1):.6f} '
                    f'scale={float(outputs["residual_scale"].detach()):.4f} '
                    f'gate={float(outputs["gate"].mean().detach()):.4f}'
                )

        metrics, predictions, score_cache = evaluate(
            model, dev_loader, dev_dataset, device, args.pair_batch_size,
        )
        improved = metrics['val_f1'] > best_f1
        best_f1 = max(best_f1, metrics['val_f1'])
        save_checkpoint(
            model.state_dict(),
            logger.checkpoint_dir / f'task3_d4_epoch_{epoch}.bin',
        )
        np.savez_compressed(
            logger.result_dir / f'dev_candidate_scores_epoch_{epoch}.npz',
            **score_cache,
        )
        if improved:
            save_checkpoint(model.state_dict(), logger.checkpoint_path('task3_d4'))
            (logger.result_dir / 'dev_predictions_best.json').write_text(
                json.dumps(predictions, ensure_ascii=False, indent=2),
                encoding='utf8',
            )
        elapsed = time.perf_counter() - start_time
        row = {
            **metrics,
            'epoch': epoch,
            'train_loss': totals['loss'] / max(len(train_loader), 1),
            'train_final_ce': totals['final'] / max(len(train_loader), 1),
            'train_definition_ce': totals['definition'] / max(len(train_loader), 1),
            'train_distill': totals['distill'] / max(len(train_loader), 1),
            'train_synthetic_hard_rate': (
                totals['perturbed'] / max(totals['train_groups'], 1)
            ),
            'best_val_f1': best_f1,
            'epoch_seconds': elapsed,
            'encoder_lr': optimizer.param_groups[0]['lr'],
            'head_lr': optimizer.param_groups[1]['lr'],
            'peak_gpu_memory_mb': (
                torch.cuda.max_memory_allocated() / 1024 ** 2
                if device.type == 'cuda' else 0.0
            ),
        }
        logger.log(row)
        print('[D4 eval] ' + json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
