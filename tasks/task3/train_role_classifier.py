#!/usr/bin/env python3
"""Train Task3 Experiment B (frame-masked baseline) or C (efficient span)."""

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
from transformers import BertConfig, BertTokenizer

from tasks.common.experiment_utils import ExperimentLogger, REPO_ROOT, save_checkpoint, set_seed
from .dataset_experiments import (
    Task3RoleDataset,
    build_augmented_frame_roles,
    load_json,
    make_role_collate,
    train_frame_frequencies,
    train_role_frequencies,
)
from .model import Model as LegacyRoleModel
from .model_c3 import UncertaintyAwareRoleClassifier
from .model_definition import DefinitionPrototypeRoleClassifier
from .model_efficient import EfficientSpanRoleClassifier
from .verifier.dataset import load_task1_hybrid_probabilities


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_variant', required=True,
                        choices=[
                            'frame_masked_baseline',
                            'efficient_span',
                            'efficient_span_calibrated',
                            'uncertainty_aware_c3',
                            'definition_prototype',
                            'definition_pair_marginal',
                        ])
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--result_dir', default=None)
    parser.add_argument('--num_train_epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--accumulate_gradients', default=1, type=int)
    parser.add_argument('--encoder_lr', default=5e-6, type=float)
    parser.add_argument('--head_lr', default=2e-5, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--max_grad_norm', default=2.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--use_fgm', action='store_true')
    parser.add_argument('--fgm_epsilon', default=1.0, type=float)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--max_train_examples', default=None, type=int)
    parser.add_argument('--max_dev_examples', default=None, type=int)
    parser.add_argument('--span_dim', default=256, type=int)
    parser.add_argument('--role_dim', default=256, type=int)
    parser.add_argument('--distance_dim', default=32, type=int)
    parser.add_argument('--width_dim', default=16, type=int)
    parser.add_argument(
        '--global_loss_weight', default=0.0, type=float,
        help='Weight alpha for global CE; C2/D use 0.2 by default in commands.',
    )
    parser.add_argument(
        '--prototype_loss_weight', default=0.0, type=float,
        help='Weight beta for prototype-only frame CE in Experiment D.',
    )
    parser.add_argument(
        '--definition_prototype_file',
        default='./tasks/task3/definition_prototypes/frame_fe_definition_last4_mean.npz',
    )
    parser.add_argument('--definition_scale_init', default=0.1, type=float)
    parser.add_argument('--definition_temperature_init', default=10.0, type=float)
    parser.add_argument('--definition_frame_topk', default=5, type=int)
    parser.add_argument('--frame_probability_smoothing', default=0.1, type=float)
    parser.add_argument('--frame_condition_dropout', default=0.3, type=float)
    parser.add_argument('--c3_gold_frame_mass', default=0.8, type=float)
    parser.add_argument('--c3_distractor_frame_mass', default=0.15, type=float)
    parser.add_argument('--c3_legal_prior_scale_init', default=0.5, type=float)
    parser.add_argument('--c3_role_balance_power', default=0.25, type=float)
    parser.add_argument('--c3_role_balance_max', default=3.0, type=float)
    parser.add_argument(
        '--task1_dev_logits',
        default='./tasks/task1/analysis/baseline-prototype-10ep/dev_logits.npz',
    )
    parser.add_argument('--task1_prototype_gamma', default=2.7, type=float)
    parser.add_argument('--task1_baseline_weight', default=0.5, type=float)
    parser.add_argument('--frequency_weight_power', default=0.5, type=float)
    parser.add_argument('--frequency_weight_max', default=5.0, type=float)
    parser.add_argument('--freeze_definition_temperature', action='store_true')
    parser.add_argument('--gate_regularization_weight', default=0.0, type=float)
    parser.add_argument('--gate_regularization_target', default=0.1, type=float)
    parser.add_argument(
        '--save_every_epoch', action='store_true',
        help='Save an epoch-specific checkpoint and Dev-logit cache for pipeline selection.',
    )
    parser.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    parser.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    parser.add_argument('--train_file', default='./dataset/cfn-train.json')
    parser.add_argument('--dev_file', default='./dataset/cfn-dev.json')
    parser.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    parser.add_argument(
        '--warm_start_checkpoint',
        default='./tasks/task3/experiment_record/legacy-saves/checkpoints/model_task3_best.bin',
    )
    return parser.parse_args()


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.last_attack_count = 0

    def attack(self, epsilon=1.0, emb_name='embeddings'):
        self.last_attack_count = 0
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None and emb_name in name:
                norm = torch.norm(parameter.grad)
                if torch.isfinite(norm) and norm > 0:
                    self.backup[name] = parameter.data.clone()
                    parameter.data.add_(epsilon * parameter.grad / norm)
                    self.last_attack_count += 1

    def restore(self):
        named = dict(self.model.named_parameters())
        for name, value in self.backup.items():
            named[name].data.copy_(value)
        self.backup = {}


def masked_logits(logits, candidate_mask):
    return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)


def gold_smoothed_frame_probabilities(model, batch):
    """Construct D2 Train/Dev routing without leaking role labels.

    During training only, complete rows are randomly dropped to the global FE
    fallback.  Dev remains deterministic and uses the smoothed gold frame so
    epoch-to-epoch numbers are comparable.
    """
    mass = 1.0 - float(model.frame_probability_smoothing)
    if not 0.0 <= mass <= 1.0:
        raise ValueError('frame_probability_smoothing must be within [0,1].')
    probabilities = torch.zeros(
        (len(batch['frame_ids']), model.frame_pair_indices.shape[0]),
        dtype=torch.float32,
        device=batch['frame_ids'].device,
    )
    probabilities.scatter_(1, batch['frame_ids'][:, None], mass)
    if model.training and model.frame_condition_dropout > 0:
        keep = torch.rand(
            len(probabilities), device=probabilities.device
        ) >= model.frame_condition_dropout
        probabilities = probabilities * keep[:, None]
    return probabilities


def c3_frame_probabilities(model, batch):
    """Create label-free Task1-style noisy frame conditions for C3.

    Train rows receive a dominant gold frame, one random distractor frame and
    explicit omitted/global mass.  Complete frame dropout teaches the exact
    global fallback.  Evaluation uses a deterministic one-hot gold frame so
    oracle-frame metrics remain comparable with C/C2/D.
    """
    frame_ids = batch['frame_ids']
    probabilities = torch.zeros(
        (len(frame_ids), model.num_frames),
        dtype=torch.float32,
        device=frame_ids.device,
    )
    if not model.training:
        if batch.get('frame_probabilities') is not None:
            return batch['frame_probabilities']
        probabilities.scatter_(1, frame_ids[:, None], 1.0)
        return probabilities
    gold_mass = float(model.c3_gold_frame_mass)
    distractor_mass = float(model.c3_distractor_frame_mass)
    if gold_mass < 0 or distractor_mass < 0 or gold_mass + distractor_mass > 1:
        raise ValueError(
            'C3 gold+distractor frame mass must be non-negative and <= 1.'
        )
    probabilities.scatter_(1, frame_ids[:, None], gold_mass)
    if distractor_mass > 0 and model.num_frames > 1:
        distractors = torch.randint(
            0, model.num_frames - 1, (len(frame_ids),), device=frame_ids.device
        )
        distractors = distractors + (distractors >= frame_ids).long()
        probabilities.scatter_add_(
            1,
            distractors[:, None],
            torch.full(
                (len(frame_ids), 1), distractor_mass,
                dtype=probabilities.dtype, device=probabilities.device,
            ),
        )
    if model.frame_condition_dropout > 0:
        keep = torch.rand(len(probabilities), device=probabilities.device) >= (
            model.frame_condition_dropout
        )
        probabilities = probabilities * keep[:, None]
    return probabilities


def forward_outputs(model, variant, batch):
    if variant == 'frame_masked_baseline':
        return model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            target=batch['argument'],
            labels=None,
            device=batch['input_ids'].device,
            for_test=True,
        )
    if variant == 'definition_prototype':
        return model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            target=batch['target'],
            argument=batch['argument'],
            frame_ids=batch['frame_ids'],
        )
    if variant == 'definition_pair_marginal':
        return model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            target=batch['target'],
            argument=batch['argument'],
            frame_probabilities=gold_smoothed_frame_probabilities(model, batch),
            frame_topk=model.definition_frame_topk,
        )
    if variant == 'uncertainty_aware_c3':
        return model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            target=batch['target'],
            argument=batch['argument'],
            frame_probabilities=c3_frame_probabilities(model, batch),
        )
    return model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        target=batch['target'],
        argument=batch['argument'],
    )


def forward_logits(model, variant, batch):
    return forward_outputs(model, variant, batch)['logits']


def objective_loss(
    outputs, batch, global_loss_weight=0.0, prototype_loss_weight=0.0,
    gate_regularization_weight=0.0, gate_regularization_target=0.1,
):
    logits = outputs['logits']
    frame_per_sample = F.cross_entropy(
        masked_logits(logits, batch['candidate_mask']), batch['labels'],
        reduction='none',
    )
    global_per_sample = F.cross_entropy(
        logits, batch['labels'], reduction='none'
    )
    sample_weights = batch.get('role_sample_weights')
    if sample_weights is not None:
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
        frame_loss = (frame_per_sample * sample_weights).mean()
        global_loss = (global_per_sample * sample_weights).mean()
    else:
        frame_loss = frame_per_sample.mean()
        global_loss = global_per_sample.mean()
    total = frame_loss + global_loss_weight * global_loss
    prototype_loss = logits.new_zeros(())
    if prototype_loss_weight:
        if 'prototype_logits' not in outputs:
            raise ValueError('prototype_loss_weight requires definition_prototype model outputs.')
        prototype_per_sample = F.cross_entropy(
            masked_logits(outputs['prototype_logits'], batch['candidate_mask']),
            batch['labels'],
            reduction='none',
        )
        weights = batch.get('prototype_sample_weights')
        prototype_loss = (
            (prototype_per_sample * weights).mean()
            if weights is not None else prototype_per_sample.mean()
        )
        total = total + prototype_loss_weight * prototype_loss
    gate_regularization = logits.new_zeros(())
    if gate_regularization_weight:
        if 'definition_gate' not in outputs:
            raise ValueError('gate_regularization_weight requires a definition model.')
        gate_regularization = (
            outputs['definition_gate'] - float(gate_regularization_target)
        ).pow(2)
        total = total + gate_regularization_weight * gate_regularization
    return total, frame_loss, global_loss, prototype_loss, gate_regularization


def frequency_bucket(frequency):
    if frequency == 0:
        return 'unseen_0'
    if frequency <= 5:
        return 'rare_1_5'
    if frequency <= 15:
        return 'few_6_15'
    if frequency <= 50:
        return 'medium_16_50'
    return 'head_gt_50'


def evaluate(
    model, loader, device, variant, collect_logits=True,
    global_loss_weight=0.0, prototype_loss_weight=0.0,
    gate_regularization_weight=0.0, gate_regularization_target=0.1,
):
    model.eval()
    total_loss = total_frame_loss = total_global_loss = total_prototype_loss = 0.0
    total_gate_regularization = 0.0
    global_correct = frame_correct = top3_correct = top5_correct = total = 0
    learned_frame_correct = prototype_frame_correct = 0
    c3_global_frame_correct = c3_local_frame_correct = 0
    c3_gate_sum = c3_component_count = 0
    component_count = 0
    role_buckets = defaultdict(lambda: [0, 0])
    frame_buckets = defaultdict(lambda: [0, 0])
    cache = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(loader, desc='task3-eval'):
            outputs = forward_outputs(model, variant, batch)
            logits = outputs['logits']
            frame_logits = masked_logits(logits, batch['candidate_mask'])
            loss, frame_loss, global_loss, prototype_loss, gate_regularization = objective_loss(
                outputs, batch, global_loss_weight, prototype_loss_weight,
                gate_regularization_weight, gate_regularization_target,
            )
            batch_size = len(batch['labels'])
            total_loss += float(loss.item()) * batch_size
            total_frame_loss += float(frame_loss.item()) * batch_size
            total_global_loss += float(global_loss.item()) * batch_size
            total_prototype_loss += float(prototype_loss.item()) * batch_size
            total_gate_regularization += float(gate_regularization.item()) * batch_size
            global_prediction = logits.argmax(dim=-1)
            top = torch.topk(frame_logits, k=min(5, logits.shape[-1]), dim=-1).indices
            frame_prediction = top[:, 0]
            labels = batch['labels']
            global_correct += int((global_prediction == labels).sum().item())
            frame_correct += int((frame_prediction == labels).sum().item())
            top3_correct += int((top[:, :3] == labels[:, None]).any(dim=1).sum().item())
            top5_correct += int((top == labels[:, None]).any(dim=1).sum().item())
            total += len(labels)
            if 'prototype_logits' in outputs:
                learned_prediction = masked_logits(
                    outputs['learned_logits'], batch['candidate_mask']
                ).argmax(dim=-1)
                prototype_prediction = masked_logits(
                    outputs['prototype_logits'], batch['candidate_mask']
                ).argmax(dim=-1)
                learned_frame_correct += int((learned_prediction == labels).sum().item())
                prototype_frame_correct += int((prototype_prediction == labels).sum().item())
                component_count += len(labels)
            if 'global_logits' in outputs and 'local_residual' in outputs:
                c3_global_prediction = masked_logits(
                    outputs['global_logits'], batch['candidate_mask']
                ).argmax(dim=-1)
                c3_local_prediction = masked_logits(
                    outputs['local_residual'], batch['candidate_mask']
                ).argmax(dim=-1)
                c3_global_frame_correct += int(
                    (c3_global_prediction == labels).sum().item()
                )
                c3_local_frame_correct += int(
                    (c3_local_prediction == labels).sum().item()
                )
                c3_gate_sum += float(outputs['dynamic_gate'].sum().item())
                c3_component_count += len(labels)
            for i, row in enumerate(batch['rows']):
                role_bucket = frequency_bucket(row['train_frequency'])
                frame_bucket = frequency_bucket(row['frame_train_frequency'])
                role_buckets[role_bucket][0] += int(frame_prediction[i] == labels[i])
                role_buckets[role_bucket][1] += 1
                frame_buckets[frame_bucket][0] += int(frame_prediction[i] == labels[i])
                frame_buckets[frame_bucket][1] += 1
            if collect_logits:
                cache['logits'].append(logits.detach().cpu().to(torch.float32).numpy())
                cache['labels'].append(labels.detach().cpu().numpy())
                cache['sentence_ids'].extend(row['sentence_id'] for row in batch['rows'])
                cache['instance_indices'].extend(row['instance_index'] for row in batch['rows'])
                cache['starts'].extend(row['argument_start'] for row in batch['rows'])
                cache['ends'].extend(row['argument_end'] for row in batch['rows'])
    metrics = {
        'val_loss': total_loss / max(total, 1),
        'val_frame_loss': total_frame_loss / max(total, 1),
        'val_global_loss': total_global_loss / max(total, 1),
        'val_prototype_loss': total_prototype_loss / max(total, 1),
        'val_gate_regularization': total_gate_regularization / max(total, 1),
        'val_global_acc': global_correct / max(total, 1),
        'val_frame_acc': frame_correct / max(total, 1),
        'val_frame_top3': top3_correct / max(total, 1),
        'val_frame_top5': top5_correct / max(total, 1),
        'val_learned_frame_acc': learned_frame_correct / max(component_count, 1),
        'val_prototype_frame_acc': prototype_frame_correct / max(component_count, 1),
        'val_c3_global_frame_acc': (
            c3_global_frame_correct / max(c3_component_count, 1)
        ),
        'val_c3_local_frame_acc': (
            c3_local_frame_correct / max(c3_component_count, 1)
        ),
        'val_c3_dynamic_gate_mean': c3_gate_sum / max(c3_component_count, 1),
    }
    for name in ('unseen_0', 'rare_1_5', 'few_6_15', 'medium_16_50', 'head_gt_50'):
        correct, count = role_buckets[name]
        metrics[f'val_role_{name}_acc'] = correct / max(count, 1)
        metrics[f'val_role_{name}_count'] = count
        # Preserve old keys for historical CSV readers; they are role buckets.
        metrics[f'val_{name}_acc'] = correct / max(count, 1)
        metrics[f'val_{name}_count'] = count
        correct, count = frame_buckets[name]
        metrics[f'val_framefreq_{name}_acc'] = correct / max(count, 1)
        metrics[f'val_framefreq_{name}_count'] = count
    if collect_logits:
        cache = {
            'logits': np.concatenate(cache['logits'], axis=0),
            'labels': np.concatenate(cache['labels'], axis=0),
            'sentence_ids': np.asarray(cache['sentence_ids'], dtype=np.int64),
            'instance_indices': np.asarray(cache['instance_indices'], dtype=np.int32),
            'starts': np.asarray(cache['starts'], dtype=np.int32),
            'ends': np.asarray(cache['ends'], dtype=np.int32),
        }
    return metrics, cache


def build_model(
    args, num_labels, frame_to_role_ids=None, frame_to_id=None,
    frame_frequency=None,
):
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = num_labels
    if args.model_variant == 'frame_masked_baseline':
        model = LegacyRoleModel(config)
        # The legacy LSTM is declared but never used in forward().
        for parameter in model.lstm.parameters():
            parameter.requires_grad = False
    elif args.model_variant in ('efficient_span', 'efficient_span_calibrated'):
        model = EfficientSpanRoleClassifier(
            config,
            span_dim=args.span_dim,
            role_dim=args.role_dim,
            distance_dim=args.distance_dim,
            width_dim=args.width_dim,
        )
    elif args.model_variant == 'uncertainty_aware_c3':
        if frame_to_role_ids is None or frame_to_id is None or frame_frequency is None:
            raise ValueError('C3 requires frame ontology and Train frame frequencies.')
        legal = np.zeros((len(frame_to_id), num_labels), dtype=np.float32)
        for frame, role_ids in frame_to_role_ids.items():
            legal[frame_to_id[frame], role_ids] = 1.0
        raw_frequency = np.asarray([
            frame_frequency.get(frame, 0)
            for frame, _ in sorted(frame_to_id.items(), key=lambda item: item[1])
        ], dtype=np.float32)
        log_frequency = np.log1p(raw_frequency)
        if log_frequency.max() > 0:
            log_frequency /= log_frequency.max()
        model = UncertaintyAwareRoleClassifier(
            config,
            num_frames=len(frame_to_id),
            frame_role_legal=legal,
            frame_log_frequency=log_frequency,
            span_dim=args.span_dim,
            role_dim=args.role_dim,
            distance_dim=args.distance_dim,
            width_dim=args.width_dim,
            legal_prior_scale_init=args.c3_legal_prior_scale_init,
        )
        model.c3_gold_frame_mass = args.c3_gold_frame_mass
        model.c3_distractor_frame_mass = args.c3_distractor_frame_mass
        model.frame_condition_dropout = args.frame_condition_dropout
    else:
        model = DefinitionPrototypeRoleClassifier(
            config,
            prototype_file=args.definition_prototype_file,
            span_dim=args.span_dim,
            role_dim=args.role_dim,
            distance_dim=args.distance_dim,
            width_dim=args.width_dim,
            definition_scale_init=args.definition_scale_init,
            definition_temperature_init=args.definition_temperature_init,
            definition_aggregation=(
                'pair_logsumexp'
                if args.model_variant == 'definition_pair_marginal' else 'residual'
            ),
            freeze_definition_temperature=args.freeze_definition_temperature,
        )
        if args.model_variant == 'definition_pair_marginal':
            model.frame_probability_smoothing = args.frame_probability_smoothing
            model.frame_condition_dropout = args.frame_condition_dropout
            model.definition_frame_topk = args.definition_frame_topk
    state = torch.load(args.warm_start_checkpoint, map_location='cpu')
    if args.model_variant == 'frame_masked_baseline':
        message = model.load_state_dict(state, strict=False)
        if message.missing_keys or message.unexpected_keys:
            raise RuntimeError(
                f'Experiment B checkpoint mismatch: missing={message.missing_keys} '
                f'unexpected={message.unexpected_keys}'
            )
        print('[warm start B] full legacy checkpoint loaded: missing=0 unexpected=0')
    elif args.model_variant == 'efficient_span':
        bert_state = {name: value for name, value in state.items() if name.startswith('bert.')}
        message = model.load_state_dict(bert_state, strict=False)
        missing_bert = [name for name in message.missing_keys if name.startswith('bert.')]
        if missing_bert or message.unexpected_keys:
            raise RuntimeError(
                f'Experiment C BERT warm-start mismatch: missing_bert={missing_bert} '
                f'unexpected={message.unexpected_keys}'
            )
        print(f'[warm start C] loaded {len(bert_state)} BERT tensors; '
              f'new head tensors={len(message.missing_keys)}')
    elif args.model_variant == 'efficient_span_calibrated':
        message = model.load_state_dict(state, strict=True)
        print('[warm start C2] full Experiment C checkpoint loaded: missing=0 unexpected=0')
    elif args.model_variant == 'uncertainty_aware_c3':
        message = model.load_state_dict(state, strict=False)
        allowed_prefixes = (
            'frame_role_legal', 'frame_log_frequency', 'frame_embeddings',
            'local_interaction.', 'local_role_embeddings', 'local_role_bias',
            'gate_network.', 'legal_prior_raw_scale',
        )
        bad_missing = [
            name for name in message.missing_keys
            if not name.startswith(allowed_prefixes)
        ]
        if bad_missing or message.unexpected_keys:
            raise RuntimeError(
                f'Experiment C3 C warm-start mismatch: bad_missing={bad_missing} '
                f'unexpected={message.unexpected_keys}'
            )
        print(
            f'[warm start C3] full C checkpoint loaded; '
            f'new C3 tensors={len(message.missing_keys)}'
        )
    else:
        message = model.load_state_dict(state, strict=False)
        allowed_prefixes = (
            'definition_projection.', 'definition_query_projection.',
            'definition_gate', 'definition_log_temperature',
        )
        bad_missing = [
            name for name in message.missing_keys
            if not name.startswith(allowed_prefixes)
        ]
        if bad_missing or message.unexpected_keys:
            raise RuntimeError(
                f'Experiment D C warm-start mismatch: bad_missing={bad_missing} '
                f'unexpected={message.unexpected_keys}'
            )
        print(
            f'[warm start {"D2" if args.model_variant == "definition_pair_marginal" else "D"}] '
            f'full C checkpoint loaded; '
            f'new definition tensors={len(message.missing_keys)}'
        )
    return model


def build_optimizer(model, args):
    groups = []
    for encoder in (True, False):
        for decay in (True, False):
            parameters = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                is_encoder = name.startswith('bert.')
                no_decay = (
                    name.endswith('bias') or 'LayerNorm.weight' in name
                    or 'LayerNorm.bias' in name or name == 'role_bias'
                )
                if is_encoder == encoder and (not no_decay) == decay:
                    parameters.append(parameter)
            if parameters:
                learning_rate = args.encoder_lr if encoder else args.head_lr
                groups.append({
                    'params': parameters,
                    'lr': learning_rate,
                    'initial_lr': learning_rate,
                    'weight_decay': args.weight_decay if decay else 0.0,
                    'is_encoder': encoder,
                })
    return torch.optim.AdamW(groups)


def lr_factor(step, total_steps, warmup_proportion):
    progress = step / max(total_steps, 1)
    if progress < warmup_proportion:
        return progress / max(warmup_proportion, 1e-9)
    return max((1.0 - progress) / max(1.0 - warmup_proportion, 1e-9), 0.0)


def save_dev_cache(cache, path):
    np.savez_compressed(path, **cache)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    frame_info, idx2label, label2idx, frame_to_role_ids, augmented = build_augmented_frame_roles(
        args.frame_info_file, args.train_file
    )
    frame_to_id = {
        frame['frame_name']: frame_id for frame_id, frame in enumerate(frame_info)
    }
    if args.model_variant in ('definition_prototype', 'definition_pair_marginal'):
        prototype_bank = np.load(args.definition_prototype_file, allow_pickle=False)
        if 'frame_names' in prototype_bank.files:
            prototype_frames = prototype_bank['frame_names'].tolist()
            ontology_frames = [frame['frame_name'] for frame in frame_info]
            if prototype_frames != ontology_frames:
                raise ValueError('Definition prototype frame order does not match frame_info.json.')
        if 'role_names' in prototype_bank.files:
            if prototype_bank['role_names'].tolist() != idx2label:
                raise ValueError('Definition prototype role order does not match Task3 labels.')
    role_frequency = train_role_frequencies(args.train_file)
    frame_frequency = train_frame_frequencies(args.train_file)
    max_role_frequency = max(role_frequency.values())
    role_sample_weights = {
        role: min(
            max(
                (max_role_frequency / (frequency + 1.0))
                ** args.c3_role_balance_power,
                1.0,
            ),
            args.c3_role_balance_max,
        )
        for role in idx2label
        for frequency in [role_frequency.get(role, 0)]
    }
    max_frame_frequency = max(frame_frequency.values())
    frame_sample_weights = {
        frame: min(
            max(
                (max_frame_frequency / (frequency + 1.0))
                ** args.frequency_weight_power,
                1.0,
            ),
            args.frequency_weight_max,
        )
        for frame in frame_to_id
        for frequency in [frame_frequency.get(frame, 0)]
    }
    train_dataset = Task3RoleDataset(
        args.train_file, tokenizer, label2idx, role_frequency,
        train_frame_frequency=frame_frequency,
        max_examples=args.max_train_examples, max_length=args.max_length,
    )
    dev_dataset = Task3RoleDataset(
        args.dev_file, tokenizer, label2idx, role_frequency,
        train_frame_frequency=frame_frequency,
        max_examples=args.max_dev_examples, max_length=args.max_length,
    )
    train_collate = make_role_collate(
        frame_to_role_ids, len(idx2label), tokenizer.pad_token_id, device,
        frame_to_id=(
            frame_to_id
            if args.model_variant in (
                'definition_prototype', 'definition_pair_marginal',
                'uncertainty_aware_c3',
            )
            else None
        ),
        frame_sample_weights=(
            frame_sample_weights
            if args.model_variant == 'definition_pair_marginal' else None
        ),
        role_sample_weights=(
            role_sample_weights
            if args.model_variant == 'uncertainty_aware_c3' else None
        ),
    )
    c3_dev_probabilities = None
    if args.model_variant == 'uncertainty_aware_c3':
        c3_dev_probabilities = load_task1_hybrid_probabilities(
            args.task1_dev_logits,
            load_json(args.dev_file),
            load_json(args.train_file),
            [frame['frame_name'] for frame in frame_info],
            gamma=args.task1_prototype_gamma,
            baseline_weight=args.task1_baseline_weight,
        )
    dev_collate = make_role_collate(
        frame_to_role_ids, len(idx2label), tokenizer.pad_token_id, device,
        frame_to_id=(
            frame_to_id
            if args.model_variant in (
                'definition_prototype', 'definition_pair_marginal',
                'uncertainty_aware_c3',
            )
            else None
        ),
        frame_sample_weights=(
            frame_sample_weights
            if args.model_variant == 'definition_pair_marginal' else None
        ),
        role_sample_weights=(
            role_sample_weights
            if args.model_variant == 'uncertainty_aware_c3' else None
        ),
        instance_frame_probabilities=c3_dev_probabilities,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=train_collate, drop_last=False,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=dev_collate, drop_last=False,
    )
    dev_gold_loader = None
    if args.model_variant == 'uncertainty_aware_c3':
        gold_collate = make_role_collate(
            frame_to_role_ids, len(idx2label), tokenizer.pad_token_id, device,
            frame_to_id=frame_to_id,
            role_sample_weights=role_sample_weights,
        )
        dev_gold_loader = DataLoader(
            dev_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=gold_collate, drop_last=False,
        )
    model = build_model(
        args,
        len(idx2label),
        frame_to_role_ids=frame_to_role_ids,
        frame_to_id=frame_to_id,
        frame_frequency=frame_frequency,
    ).to(device)
    optimizer = build_optimizer(model, args)
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulate_gradients)
    total_steps = updates_per_epoch * args.num_train_epochs
    fgm = FGM(model)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    encoder_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if name.startswith('bert.')
    )
    head_parameters = total_parameters - encoder_parameters
    print(json.dumps({
        'variant': args.model_variant,
        'device': str(device),
        'train_examples': len(train_dataset),
        'dev_examples': len(dev_dataset),
        'train_target_overlaps': train_dataset.target_overlap_count,
        'dev_target_overlaps': dev_dataset.target_overlap_count,
        'augmented_frame_role_licenses': augmented,
        'total_parameters': total_parameters,
        'trainable_parameters': trainable_parameters,
        'head_parameters': head_parameters,
        'epochs': args.num_train_epochs,
        'batch_size': args.batch_size,
        'use_fgm': args.use_fgm,
        'global_loss_weight': args.global_loss_weight,
        'prototype_loss_weight': args.prototype_loss_weight,
        'gate_regularization_weight': args.gate_regularization_weight,
        'gate_regularization_target': args.gate_regularization_target,
        'frame_probability_smoothing': args.frame_probability_smoothing,
        'frame_condition_dropout': args.frame_condition_dropout,
        'definition_frame_topk': args.definition_frame_topk,
        'freeze_definition_temperature': args.freeze_definition_temperature,
    }, ensure_ascii=False, indent=2))

    fieldnames = [
        'epoch', 'train_loss', 'train_frame_loss', 'train_global_loss',
        'train_prototype_loss', 'train_gate_regularization',
        'val_loss', 'val_frame_loss', 'val_global_loss',
        'val_prototype_loss', 'val_gate_regularization', 'val_global_acc', 'val_frame_acc',
        'val_frame_top3', 'val_frame_top5',
        'val_learned_frame_acc', 'val_prototype_frame_acc',
        'val_c3_global_frame_acc', 'val_c3_local_frame_acc',
        'val_c3_dynamic_gate_mean',
        'val_c3_task1_global_acc', 'val_c3_task1_frame_acc',
        'val_c3_oracle_global_acc', 'val_c3_oracle_frame_acc',
        'val_c3_oracle_role_unseen_acc', 'val_c3_oracle_role_rare_acc',
        'val_c3_oracle_framefreq_unseen_acc',
        'val_c3_oracle_framefreq_rare_acc',
        'val_unseen_0_acc', 'val_unseen_0_count',
        'val_rare_1_5_acc', 'val_rare_1_5_count',
        'val_few_6_15_acc', 'val_few_6_15_count',
        'val_medium_16_50_acc', 'val_medium_16_50_count',
        'val_head_gt_50_acc', 'val_head_gt_50_count',
        'val_role_unseen_0_acc', 'val_role_unseen_0_count',
        'val_role_rare_1_5_acc', 'val_role_rare_1_5_count',
        'val_role_few_6_15_acc', 'val_role_few_6_15_count',
        'val_role_medium_16_50_acc', 'val_role_medium_16_50_count',
        'val_role_head_gt_50_acc', 'val_role_head_gt_50_count',
        'val_framefreq_unseen_0_acc', 'val_framefreq_unseen_0_count',
        'val_framefreq_rare_1_5_acc', 'val_framefreq_rare_1_5_count',
        'val_framefreq_few_6_15_acc', 'val_framefreq_few_6_15_count',
        'val_framefreq_medium_16_50_acc', 'val_framefreq_medium_16_50_count',
        'val_framefreq_head_gt_50_acc', 'val_framefreq_head_gt_50_count',
        'best_frame_acc', 'best_global_acc', 'epoch_seconds', 'samples_per_second',
        'encoder_lr', 'head_lr', 'peak_gpu_memory_mb',
        'fgm_attack_tensors',
        'definition_gate', 'definition_temperature',
        'c3_legal_prior_scale',
        'total_parameters', 'trainable_parameters', 'head_parameters',
    ]
    logger = ExperimentLogger(args, 'task3', fieldnames)
    (logger.result_dir / 'frame_role_augmentation.json').write_text(
        json.dumps(augmented, ensure_ascii=False, indent=2), encoding='utf8'
    )
    model_summary = {
        'variant': args.model_variant,
        'total_parameters': total_parameters,
        'trainable_parameters': trainable_parameters,
        'encoder_parameters': encoder_parameters,
        'head_parameters': head_parameters,
        'train_examples': len(train_dataset),
        'dev_examples': len(dev_dataset),
        'train_target_overlaps': train_dataset.target_overlap_count,
        'dev_target_overlaps': dev_dataset.target_overlap_count,
        'augmented_frame_role_licenses': augmented,
        'global_loss_weight': args.global_loss_weight,
        'prototype_loss_weight': args.prototype_loss_weight,
        'gate_regularization_weight': args.gate_regularization_weight,
        'gate_regularization_target': args.gate_regularization_target,
        'frame_probability_smoothing': (
            args.frame_probability_smoothing
            if args.model_variant == 'definition_pair_marginal' else None
        ),
        'frame_condition_dropout': (
            args.frame_condition_dropout
            if args.model_variant == 'definition_pair_marginal' else None
        ),
        'definition_frame_topk': args.definition_frame_topk,
        'freeze_definition_temperature': args.freeze_definition_temperature,
        'frequency_weight_power': args.frequency_weight_power,
        'frequency_weight_max': args.frequency_weight_max,
        'c3_gold_frame_mass': args.c3_gold_frame_mass,
        'c3_distractor_frame_mass': args.c3_distractor_frame_mass,
        'c3_legal_prior_scale_init': args.c3_legal_prior_scale_init,
        'c3_role_balance_power': args.c3_role_balance_power,
        'c3_role_balance_max': args.c3_role_balance_max,
        'c3_checkpoint_selection_metric': (
            'task1_conditioned_global_role_acc'
            if args.model_variant == 'uncertainty_aware_c3' else None
        ),
        'task1_dev_logits': (
            args.task1_dev_logits
            if args.model_variant == 'uncertainty_aware_c3' else None
        ),
        'definition_prototype_file': (
            args.definition_prototype_file
            if args.model_variant in ('definition_prototype', 'definition_pair_marginal')
            else None
        ),
    }
    (logger.result_dir / 'model_summary.json').write_text(
        json.dumps(model_summary, ensure_ascii=False, indent=2), encoding='utf8'
    )
    if args.model_variant == 'frame_masked_baseline':
        pretrain_metrics, _ = evaluate(
            model, dev_loader, device, args.model_variant, collect_logits=False,
            global_loss_weight=args.global_loss_weight,
            prototype_loss_weight=args.prototype_loss_weight,
            gate_regularization_weight=args.gate_regularization_weight,
            gate_regularization_target=args.gate_regularization_target,
        )
        (logger.result_dir / 'pretrain_metrics.json').write_text(
            json.dumps(pretrain_metrics, ensure_ascii=False, indent=2), encoding='utf8'
        )
        print('[pretrain B] ' + json.dumps(pretrain_metrics, ensure_ascii=False))

    best_frame_acc = -1.0
    best_global_acc = -1.0
    best_frame_epoch = 0
    best_global_epoch = 0
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(1, args.num_train_epochs + 1):
        epoch_start = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        model.train()
        total_train_loss = 0.0
        total_train_frame_loss = 0.0
        total_train_global_loss = 0.0
        total_train_prototype_loss = 0.0
        total_train_gate_regularization = 0.0
        fgm_attack_tensors = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f'task3-{args.model_variant}-epoch_{epoch}')):
            outputs = forward_outputs(model, args.model_variant, batch)
            loss, frame_loss, global_loss, prototype_loss, gate_regularization = objective_loss(
                outputs, batch, args.global_loss_weight, args.prototype_loss_weight,
                args.gate_regularization_weight, args.gate_regularization_target,
            )
            total_train_loss += float(loss.item())
            total_train_frame_loss += float(frame_loss.item())
            total_train_global_loss += float(global_loss.item())
            total_train_prototype_loss += float(prototype_loss.item())
            total_train_gate_regularization += float(gate_regularization.item())
            (loss / args.accumulate_gradients).backward()
            if args.use_fgm:
                fgm.attack(args.fgm_epsilon)
                fgm_attack_tensors += fgm.last_attack_count
                adversarial_outputs = forward_outputs(model, args.model_variant, batch)
                adversarial_loss, _, _, _, _ = objective_loss(
                    adversarial_outputs, batch,
                    args.global_loss_weight, args.prototype_loss_weight,
                    args.gate_regularization_weight, args.gate_regularization_target,
                )
                adversarial_loss = adversarial_loss / args.accumulate_gradients
                adversarial_loss.backward()
                fgm.restore()
            should_step = (
                (step + 1) % args.accumulate_gradients == 0
                or step + 1 == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                global_step += 1
                factor = lr_factor(global_step, total_steps, args.warmup_proportion)
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * factor
                optimizer.step()
                optimizer.zero_grad()
            if args.log_interval and (step + 1) % args.log_interval == 0:
                print(f'[train] epoch={epoch} step={step + 1}/{len(train_loader)} '
                      f'loss={total_train_loss / (step + 1):.6f}')

        metrics, cache = evaluate(
            model, dev_loader, device, args.model_variant, collect_logits=True,
            global_loss_weight=args.global_loss_weight,
            prototype_loss_weight=args.prototype_loss_weight,
            gate_regularization_weight=args.gate_regularization_weight,
            gate_regularization_target=args.gate_regularization_target,
        )
        if args.model_variant == 'uncertainty_aware_c3':
            metrics['val_c3_task1_global_acc'] = metrics['val_global_acc']
            metrics['val_c3_task1_frame_acc'] = metrics['val_frame_acc']
            oracle_metrics, _ = evaluate(
                model, dev_gold_loader, device, args.model_variant,
                collect_logits=False,
                global_loss_weight=args.global_loss_weight,
                prototype_loss_weight=args.prototype_loss_weight,
                gate_regularization_weight=args.gate_regularization_weight,
                gate_regularization_target=args.gate_regularization_target,
            )
            metrics['val_c3_oracle_global_acc'] = oracle_metrics['val_global_acc']
            metrics['val_c3_oracle_frame_acc'] = oracle_metrics['val_frame_acc']
            metrics['val_c3_oracle_role_unseen_acc'] = (
                oracle_metrics['val_role_unseen_0_acc']
            )
            metrics['val_c3_oracle_role_rare_acc'] = (
                oracle_metrics['val_role_rare_1_5_acc']
            )
            metrics['val_c3_oracle_framefreq_unseen_acc'] = (
                oracle_metrics['val_framefreq_unseen_0_acc']
            )
            metrics['val_c3_oracle_framefreq_rare_acc'] = (
                oracle_metrics['val_framefreq_rare_1_5_acc']
            )
        selection_value = (
            metrics['val_global_acc']
            if args.model_variant == 'uncertainty_aware_c3'
            else metrics['val_frame_acc']
        )
        frame_improved = selection_value > best_frame_acc
        global_improved = metrics['val_global_acc'] > best_global_acc
        if frame_improved:
            best_frame_epoch = epoch
        if global_improved:
            best_global_epoch = epoch
        best_frame_acc = max(best_frame_acc, selection_value)
        best_global_acc = max(best_global_acc, metrics['val_global_acc'])
        epoch_seconds = time.perf_counter() - epoch_start
        encoder_lr = next(group['lr'] for group in optimizer.param_groups if group['is_encoder'])
        head_lr = next(group['lr'] for group in optimizer.param_groups if not group['is_encoder'])
        row = {
            **metrics,
            'epoch': epoch,
            'train_loss': total_train_loss / max(len(train_loader), 1),
            'train_frame_loss': total_train_frame_loss / max(len(train_loader), 1),
            'train_global_loss': total_train_global_loss / max(len(train_loader), 1),
            'train_prototype_loss': total_train_prototype_loss / max(len(train_loader), 1),
            'train_gate_regularization': (
                total_train_gate_regularization / max(len(train_loader), 1)
            ),
            'best_frame_acc': best_frame_acc,
            'best_global_acc': best_global_acc,
            'epoch_seconds': epoch_seconds,
            'samples_per_second': len(train_dataset) / max(epoch_seconds, 1e-9),
            'encoder_lr': encoder_lr,
            'head_lr': head_lr,
            'fgm_attack_tensors': fgm_attack_tensors,
            'peak_gpu_memory_mb': (
                torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == 'cuda' else 0.0
            ),
            'definition_gate': (
                float(model.definition_gate.detach().cpu())
                if hasattr(model, 'definition_gate') else 0.0
            ),
            'definition_temperature': (
                float(model.definition_log_temperature.detach().exp().cpu())
                if hasattr(model, 'definition_log_temperature') else 0.0
            ),
            'c3_legal_prior_scale': (
                float(torch.nn.functional.softplus(
                    model.legal_prior_raw_scale.detach()
                ).cpu())
                if hasattr(model, 'legal_prior_raw_scale') else 0.0
            ),
            'total_parameters': total_parameters,
            'trainable_parameters': trainable_parameters,
            'head_parameters': head_parameters,
        }
        logger.log(row)
        save_all_epochs = (
            args.save_every_epoch or args.model_variant in (
                'definition_pair_marginal', 'uncertainty_aware_c3'
            )
        )
        if save_all_epochs:
            save_checkpoint(
                model.state_dict(), logger.checkpoint_dir / f'task3_epoch_{epoch}.bin'
            )
            save_dev_cache(
                cache, logger.result_dir / f'dev_logits_epoch_{epoch}.npz'
            )
        if frame_improved and args.model_variant != 'definition_pair_marginal':
            save_checkpoint(model.state_dict(), logger.checkpoint_path('task3'))
            save_dev_cache(cache, logger.result_dir / 'dev_logits_best_frame.npz')
        if global_improved and args.model_variant != 'definition_pair_marginal':
            save_checkpoint(
                model.state_dict(), logger.checkpoint_dir / 'task3_best_global.bin'
            )
        if args.model_variant == 'definition_pair_marginal':
            (logger.result_dir / 'checkpoint_manifest.json').write_text(
                json.dumps({
                    'selection_warning': (
                        'Oracle-frame/global metrics are diagnostic only; select the '
                        'final D2 epoch with full-pipeline tuple F1.'
                    ),
                    'best_oracle_frame_epoch': best_frame_epoch,
                    'best_oracle_frame_acc': best_frame_acc,
                    'best_global_epoch': best_global_epoch,
                    'best_global_acc': best_global_acc,
                    'available_epoch_checkpoints': [
                        f'checkpoints/task3_epoch_{index}.bin'
                        for index in range(1, epoch + 1)
                    ],
                }, ensure_ascii=False, indent=2),
                encoding='utf8',
            )
        print(
            f'[eval] epoch={epoch} global_acc={metrics["val_global_acc"]:.6f} '
            f'frame_acc={metrics["val_frame_acc"]:.6f} '
            f'top3={metrics["val_frame_top3"]:.6f} top5={metrics["val_frame_top5"]:.6f} '
            f'best_frame={best_frame_acc:.6f} seconds={epoch_seconds:.1f}'
        )


if __name__ == '__main__':
    main()
