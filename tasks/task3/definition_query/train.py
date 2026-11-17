#!/usr/bin/env python3
"""Train Task3 Experiment D3 Definition Query Cross-Encoder."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from tasks.common.experiment_utils import (
    ExperimentLogger,
    REPO_ROOT,
    save_checkpoint,
    set_seed,
)
from tasks.task3.dataset_experiments import (
    build_augmented_frame_roles,
    train_frame_frequencies,
    train_role_frequencies,
)
from .dataset import DefinitionQueryDataset, collate_groups
from .model import DefinitionQueryCrossEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--result_dir', default=None)
    parser.add_argument('--num_train_epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Number of positive argument groups per update batch.')
    parser.add_argument('--negative_count', default=4, type=int)
    parser.add_argument('--hard_negative_pool', default=12, type=int)
    parser.add_argument('--eval_pair_batch_size', default=12, type=int)
    parser.add_argument('--accumulate_gradients', default=1, type=int)
    parser.add_argument('--encoder_lr', default=1e-5, type=float)
    parser.add_argument('--head_lr', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--bce_weight', default=0.2, type=float)
    parser.add_argument('--projection_dim', default=512, type=int)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--max_train_examples', default=None, type=int)
    parser.add_argument('--max_dev_examples', default=None, type=int)
    parser.add_argument('--pretrained_model_dir',
                        default=str(REPO_ROOT / 'chinese_bert_wwm_ext'))
    parser.add_argument('--bert_warm_start_checkpoint', default=None,
                        help='Optional C/C2 checkpoint; only bert.* tensors are loaded.')
    parser.add_argument('--prototype_file', default=str(
        REPO_ROOT / 'tasks/task3/definition_prototypes/frame_fe_definition_last4_mean.npz'
    ))
    parser.add_argument('--train_file', default=str(REPO_ROOT / 'dataset/cfn-train.json'))
    parser.add_argument('--dev_file', default=str(REPO_ROOT / 'dataset/cfn-dev.json'))
    parser.add_argument('--frame_info_file',
                        default=str(REPO_ROOT / 'dataset/frame_info.json'))
    return parser.parse_args()


def frequency_bucket(value):
    if value == 0:
        return 'unseen_0'
    if value <= 5:
        return 'rare_1_5'
    if value <= 15:
        return 'few_6_15'
    if value <= 50:
        return 'medium_16_50'
    return 'head_gt_50'


def to_device(batch, device):
    output = dict(batch)
    for key in ('input_ids', 'attention_mask', 'token_type_ids'):
        output[key] = output[key].to(device)
    output['marker_positions'] = {
        key: value.to(device) for key, value in batch['marker_positions'].items()
    }
    return output


def score_pairs(model, batch, pair_batch_size=None):
    total = len(batch['input_ids'])
    step = total if not pair_batch_size else int(pair_batch_size)
    values = []
    for start in range(0, total, step):
        stop = min(start + step, total)
        values.append(model(
            input_ids=batch['input_ids'][start:stop],
            attention_mask=batch['attention_mask'][start:stop],
            token_type_ids=batch['token_type_ids'][start:stop],
            marker_positions={
                key: value[start:stop]
                for key, value in batch['marker_positions'].items()
            },
        ))
    return torch.cat(values, dim=0)


def grouped_loss(scores, group_sizes, bce_weight=0.2):
    ranking_losses = []
    binary_targets = torch.zeros_like(scores)
    offset = 0
    for size in group_sizes:
        group = scores[offset: offset + size]
        if not len(group):
            raise ValueError('D3 group cannot be empty.')
        ranking_losses.append(-group[0] + torch.logsumexp(group, dim=0))
        binary_targets[offset] = 1.0
        offset += size
    if offset != len(scores):
        raise ValueError('D3 group sizes do not cover all flattened scores.')
    ranking_loss = torch.stack(ranking_losses).mean()
    binary_loss = F.binary_cross_entropy_with_logits(scores, binary_targets)
    return ranking_loss + float(bce_weight) * binary_loss, ranking_loss, binary_loss


@torch.no_grad()
def evaluate(model, loader, device, pair_batch_size):
    model.eval()
    total = correct1 = correct3 = correct5 = 0
    reciprocal_rank = 0.0
    role_buckets = defaultdict(lambda: [0, 0])
    frame_buckets = defaultdict(lambda: [0, 0])
    all_scores = []
    all_rows = []
    for batch in tqdm(loader, desc='D3-definition-query-eval'):
        batch = to_device(batch, device)
        scores = score_pairs(model, batch, pair_batch_size=pair_batch_size)
        offset = 0
        for group_index, size in enumerate(batch['group_sizes']):
            group_scores = scores[offset: offset + size]
            candidate_ids = batch['candidate_role_ids'][group_index]
            gold = batch['gold_role_ids'][group_index]
            order = torch.argsort(group_scores, descending=True).tolist()
            ranked_roles = [int(candidate_ids[index]) for index in order]
            rank = ranked_roles.index(int(gold)) + 1
            is_correct = int(rank == 1)
            correct1 += is_correct
            correct3 += int(rank <= 3)
            correct5 += int(rank <= 5)
            reciprocal_rank += 1.0 / rank
            total += 1
            row = batch['records'][group_index]
            role_bucket = frequency_bucket(row['train_role_frequency'])
            frame_bucket = frequency_bucket(row['train_frame_frequency'])
            role_buckets[role_bucket][0] += is_correct
            role_buckets[role_bucket][1] += 1
            frame_buckets[frame_bucket][0] += is_correct
            frame_buckets[frame_bucket][1] += 1
            all_scores.append({
                'instance_index': row['instance_index'],
                'sentence_id': row['sentence_id'],
                'start': row['argument_start'],
                'end': row['argument_end'],
                'gold_role_id': gold,
                'predicted_role_id': ranked_roles[0],
                'gold_rank': rank,
                'top_score': float(group_scores[order[0]].cpu()),
            })
            all_rows.append(row)
            offset += size
    metrics = {
        'val_query_top1': correct1 / max(total, 1),
        'val_query_top3': correct3 / max(total, 1),
        'val_query_top5': correct5 / max(total, 1),
        'val_query_mrr': reciprocal_rank / max(total, 1),
        'val_query_count': total,
    }
    names = ('unseen_0', 'rare_1_5', 'few_6_15', 'medium_16_50', 'head_gt_50')
    for name in names:
        correct, count = role_buckets[name]
        metrics[f'val_role_{name}_acc'] = correct / max(count, 1)
        metrics[f'val_role_{name}_count'] = count
        correct, count = frame_buckets[name]
        metrics[f'val_framefreq_{name}_acc'] = correct / max(count, 1)
        metrics[f'val_framefreq_{name}_count'] = count
    return metrics, all_scores


def load_model(args, device):
    config = BertConfig.from_pretrained(args.pretrained_model_dir)
    model = DefinitionQueryCrossEncoder(
        config, projection_dim=args.projection_dim
    )
    pretrained = BertModel.from_pretrained(args.pretrained_model_dir)
    model.bert.load_state_dict(pretrained.state_dict(), strict=True)
    del pretrained
    if args.bert_warm_start_checkpoint:
        path = Path(args.bert_warm_start_checkpoint)
        with path.open('rb') as handle:
            state = torch.load(handle, map_location='cpu')
        bert_state = {
            name[len('bert.'):]: value
            for name, value in state.items() if name.startswith('bert.')
        }
        message = model.bert.load_state_dict(bert_state, strict=True)
        if message.missing_keys or message.unexpected_keys:
            raise RuntimeError(f'D3 BERT warm-start mismatch: {message}')
        print(f'[D3 warm start] loaded {len(bert_state)} C/C2 BERT tensors from {path}')
    return model.to(device)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_dir, do_lower_case=True
    )
    frame_info, role_names, label2idx, _, augmented = build_augmented_frame_roles(
        args.frame_info_file, args.train_file
    )
    role_frequency = train_role_frequencies(args.train_file)
    frame_frequency = train_frame_frequencies(args.train_file)
    common = dict(
        frame_info=frame_info,
        tokenizer=tokenizer,
        label2idx=label2idx,
        role_names=role_names,
        prototype_file=args.prototype_file,
        seed=args.seed,
        max_length=args.max_length,
        train_role_frequency=role_frequency,
        train_frame_frequency=frame_frequency,
    )
    train_dataset = DefinitionQueryDataset(
        args.train_file,
        negative_count=args.negative_count,
        hard_negative_pool=args.hard_negative_pool,
        all_legal_candidates=False,
        max_examples=args.max_train_examples,
        **common,
    )
    dev_dataset = DefinitionQueryDataset(
        args.dev_file,
        negative_count=args.negative_count,
        hard_negative_pool=args.hard_negative_pool,
        all_legal_candidates=True,
        max_examples=args.max_dev_examples,
        **common,
    )
    collate = partial(collate_groups, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate,
    )
    # One gold argument group at a time; candidate pairs are micro-batched in
    # evaluate(), preventing large-frame OOM.
    dev_loader = DataLoader(
        dev_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate,
    )
    model = load_model(args, device)
    encoder_parameters = list(model.bert.parameters())
    head_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith('bert.')
    ]
    optimizer = torch.optim.AdamW([
        {'params': encoder_parameters, 'lr': args.encoder_lr,
         'initial_lr': args.encoder_lr, 'weight_decay': args.weight_decay},
        {'params': head_parameters, 'lr': args.head_lr,
         'initial_lr': args.head_lr, 'weight_decay': args.weight_decay},
    ])
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulate_gradients)
    total_steps = updates_per_epoch * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_proportion),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == 'cuda')
    bucket_fields = []
    for axis in ('role', 'framefreq'):
        for name in ('unseen_0', 'rare_1_5', 'few_6_15', 'medium_16_50', 'head_gt_50'):
            bucket_fields.extend([
                f'val_{axis}_{name}_acc', f'val_{axis}_{name}_count'
            ])
    logger = ExperimentLogger(args, 'task3', [
        'epoch', 'train_loss', 'train_ranking_loss', 'train_binary_loss',
        'val_query_top1', 'val_query_top3', 'val_query_top5',
        'val_query_mrr', 'val_query_count', *bucket_fields,
        'best_val_query_top1', 'epoch_seconds', 'encoder_lr', 'head_lr',
        'peak_gpu_memory_mb',
    ])
    (logger.result_dir / 'definition_query_summary.json').write_text(
        json.dumps({
            'architecture': 'sentence_definition_cross_encoder',
            'train_groups': len(train_dataset),
            'dev_groups': len(dev_dataset),
            'train_candidates_per_group': args.negative_count + 1,
            'dev_candidates': 'all gold-frame legal FE',
            'augmented_frame_role_licenses': augmented,
            'initialization': (
                args.bert_warm_start_checkpoint or args.pretrained_model_dir
            ),
            'device': str(device),
        }, ensure_ascii=False, indent=2), encoding='utf8'
    )
    print(json.dumps({
        'D3_config': {
            'device': str(device),
            'train_groups': len(train_dataset),
            'dev_groups': len(dev_dataset),
            'negative_count': args.negative_count,
            'epochs': args.num_train_epochs,
            'batch_size_groups': args.batch_size,
            'effective_pair_batch': args.batch_size * (args.negative_count + 1),
        }
    }, ensure_ascii=False))

    best_top1 = -1.0
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(1, args.num_train_epochs + 1):
        start_time = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        model.train()
        totals = defaultdict(float)
        for step, batch in enumerate(tqdm(train_loader, desc=f'D3-epoch_{epoch}')):
            batch = to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                scores = score_pairs(model, batch)
                loss, ranking_loss, binary_loss = grouped_loss(
                    scores, batch['group_sizes'], args.bce_weight
                )
                scaled_loss = loss / args.accumulate_gradients
            scaler.scale(scaled_loss).backward()
            totals['loss'] += float(loss.item())
            totals['ranking'] += float(ranking_loss.item())
            totals['binary'] += float(binary_loss.item())
            should_step = (
                (step + 1) % args.accumulate_gradients == 0
                or step + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # With fp16, GradScaler deliberately skips optimizer.step()
                # when the first batches overflow.  Advancing the scheduler in
                # that case triggers PyTorch's "scheduler before optimizer"
                # warning and consumes an LR step without an update.
                optimizer_was_run = (
                    not scaler.is_enabled()
                    or scaler.get_scale() >= scale_before_step
                )
                if optimizer_was_run:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad()
            if args.log_interval and (step + 1) % args.log_interval == 0:
                print(
                    f'[D3 train] epoch={epoch} step={step + 1}/{len(train_loader)} '
                    f'loss={totals["loss"] / (step + 1):.6f}'
                )

        metrics, predictions = evaluate(
            model, dev_loader, device, args.eval_pair_batch_size
        )
        improved = metrics['val_query_top1'] > best_top1
        best_top1 = max(best_top1, metrics['val_query_top1'])
        if improved:
            save_checkpoint(model.state_dict(), logger.checkpoint_path('task3_d3'))
            (logger.result_dir / 'dev_predictions_best.json').write_text(
                json.dumps(predictions, ensure_ascii=False, indent=2),
                encoding='utf8',
            )
        save_checkpoint(
            model.state_dict(), logger.checkpoint_dir / f'task3_d3_epoch_{epoch}.bin'
        )
        epoch_seconds = time.perf_counter() - start_time
        row = {
            **metrics,
            'epoch': epoch,
            'train_loss': totals['loss'] / max(len(train_loader), 1),
            'train_ranking_loss': totals['ranking'] / max(len(train_loader), 1),
            'train_binary_loss': totals['binary'] / max(len(train_loader), 1),
            'best_val_query_top1': best_top1,
            'epoch_seconds': epoch_seconds,
            'encoder_lr': optimizer.param_groups[0]['lr'],
            'head_lr': optimizer.param_groups[1]['lr'],
            'peak_gpu_memory_mb': (
                torch.cuda.max_memory_allocated() / 1024 ** 2
                if device.type == 'cuda' else 0.0
            ),
        }
        logger.log(row)
        print('[D3 eval] ' + json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
