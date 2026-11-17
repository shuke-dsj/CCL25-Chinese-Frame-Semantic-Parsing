#!/usr/bin/env python3
"""Memory-safe B/D/E/H centered-logit ensemble prediction for Task2.

Models are loaded and evaluated one at a time. Only one accumulated float32
score matrix per test example is kept in CPU memory, so four BERT checkpoints
never coexist on GPU. H uses its calibrated start/end boundary residual.
"""
import argparse
import gc
import json
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from .model_factory import build_task2_model


class Task2TestDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        with open(json_file, encoding='utf8') as handle:
            self.rows = json.load(handle)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer.encode_plus(list(row['text']))
        input_ids = encoded.data['input_ids']
        attention_mask = encoded.data['attention_mask']
        target = [row['target'][-1]['start'] + 1, row['target'][-1]['end'] + 1]
        # Preserve the exact baseline Task2 marker convention.
        input_ids = (
            input_ids[:target[0]] + [1]
            + input_ids[target[0]:target[1] + 1] + [2]
            + input_ids[target[1] + 1:]
        )
        attention_mask = attention_mask + [1, 1]
        return input_ids, attention_mask, target, row['sentence_id'], index


def collate(batch, device=None):
    max_length = max(len(item[0]) for item in batch)
    pad = lambda values: values + [0] * (max_length - len(values))
    return (
        torch.tensor([pad(item[0]) for item in batch], dtype=torch.long, device=device),
        torch.tensor([pad(item[1]) for item in batch], dtype=torch.long, device=device),
        [item[2] for item in batch],
        [item[3] for item in batch],
        [item[4] for item in batch],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--baseline_checkpoint',
        default='./tasks/task2/experiment_record/task2-valid-mask-fgm-5ep-result/checkpoints/task2_best.bin',
    )
    parser.add_argument(
        '--target_checkpoint',
        default=('./tasks/task2/experiment_record/'
                 'task2-D-target-conditioned-valid-mask-fgm-5ep-result/checkpoints/task2_best.bin'),
    )
    parser.add_argument(
        '--content_checkpoint',
        default=('./tasks/task2/experiment_record/'
                 'task2-E-content-aware-valid-mask-fgm-5ep-result/checkpoints/task2_best.bin'),
    )
    parser.add_argument(
        '--boundary_checkpoint',
        default=('./tasks/task2/experiment_record/'
                 'task2-H1-boundary-aux-w010-5ep-result/checkpoints/task2_best.bin'),
    )
    parser.add_argument('--baseline_threshold', type=float, default=-0.26)
    parser.add_argument('--target_threshold', type=float, default=-0.75)
    parser.add_argument('--content_threshold', type=float, default=-0.20)
    parser.add_argument('--boundary_threshold', type=float, default=0.53)
    parser.add_argument('--boundary_residual_beta', type=float, default=0.34)
    parser.add_argument('--baseline_weight', type=float, default=0.40)
    parser.add_argument('--target_weight', type=float, default=0.20)
    parser.add_argument('--content_weight', type=float, default=0.40)
    parser.add_argument('--boundary_weight', type=float, default=0.0)
    parser.add_argument('--final_threshold', type=float, default=-0.16)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    parser.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    parser.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    parser.add_argument('--test_file', default='./dataset/cfn-test-B.json')
    parser.add_argument('--task2_content_dim', type=int, default=32)
    parser.add_argument('--task2_content_scale_init', type=float, default=0.1)
    parser.add_argument(
        '--prediction_dir',
        default='./tasks/task2/predictions/task2-BDE-centered-040-020-040',
    )
    parser.add_argument(
        '--instance_sidecar_name', default='task2_instance_predictions.json',
        help=('Internal Task3-pipeline sidecar. Unlike the official output, each span '
              'retains its source instance index so duplicate sentence_id rows are safe.'),
    )
    return parser.parse_args()


def load_model(args, dataset, variant, checkpoint, device):
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = 1
    # Kept for compatibility with all three model constructors.
    with open(args.frame_info_file, encoding='utf8') as handle:
        config.max_cls = len(json.load(handle))
    config.task2_content_dim = args.task2_content_dim
    config.task2_content_scale_init = args.task2_content_scale_init
    config.task2_boundary_loss_weight = 0.1
    model = build_task2_model(config, variant)
    state = torch.load(checkpoint, map_location='cpu')
    message = model.load_state_dict(state, strict=False)
    del state
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(
            f'checkpoint mismatch for {variant}: missing={message.missing_keys}, '
            f'unexpected={message.unexpected_keys}'
        )
    print(f'[load] variant={variant} checkpoint={checkpoint}')
    return model.to(device).eval()


def accumulate_model(args, dataset, device, variant, checkpoint, center, weight,
                     accumulated, metadata, boundary_residual_beta=0.0):
    model = load_model(args, dataset, variant, checkpoint, device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate, device=device),
        drop_last=False,
    )
    example_index = 0
    with torch.no_grad():
        for input_ids, attention_mask, targets, sentence_ids, instance_indices in tqdm(
                loader, desc=f'ensemble-cache-{variant}'):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target=targets,
                labels=None,
                device=device,
                for_test=True,
            )
            logits = output['logits']
            if boundary_residual_beta != 0.0:
                logits = logits + boundary_residual_beta * (
                    output['boundary_start_logits'].unsqueeze(2)
                    + output['boundary_end_logits'].unsqueeze(1)
                )
            logits = logits.detach().cpu().numpy().astype(np.float32)
            lengths = attention_mask.sum(dim=1).detach().cpu().tolist()
            for batch_index, length_value in enumerate(lengths):
                length = int(length_value)
                score = weight * (logits[batch_index, :length, :length] - center)
                if accumulated is None:
                    raise RuntimeError('accumulated score list must be initialized')
                if len(accumulated) <= example_index:
                    accumulated.append(score.copy())
                    metadata.append({
                        'sentence_id': sentence_ids[batch_index],
                        'instance_index': int(instance_indices[batch_index]),
                        'target': list(targets[batch_index]),
                        'sequence_length': length,
                    })
                else:
                    item = metadata[example_index]
                    if (item['sentence_id'] != sentence_ids[batch_index]
                            or item['instance_index'] != int(instance_indices[batch_index])
                            or item['target'] != list(targets[batch_index])
                            or item['sequence_length'] != length):
                        raise RuntimeError(
                            f'model output order/shape mismatch at example {example_index}'
                        )
                    accumulated[example_index] += score
                example_index += 1
    if example_index != len(dataset):
        raise RuntimeError(f'{variant} produced {example_index} rows for dataset size {len(dataset)}')
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def decode_scores(accumulated, metadata, threshold):
    predictions = []
    instance_predictions = []
    for scores, item in tqdm(
            zip(accumulated, metadata), total=len(metadata), desc='safe-decode-ensemble'):
        target_start, target_end = item['target']
        sequence_length = item['sequence_length']
        sentence_id = item['sentence_id']

        # Valid characters left of target: [1, target_start-1].
        if target_start > 1:
            left = scores[1:target_start, 1:target_start] >= threshold
            for start, end in np.argwhere(np.triu(left)):
                raw_start, raw_end = int(start), int(end)
                predictions.append([sentence_id, raw_start, raw_end])
                instance_predictions.append({
                    'instance_index': item['instance_index'],
                    'sentence_id': sentence_id,
                    'start': raw_start,
                    'end': raw_end,
                })  # local index 0 maps to original character 0

        # After two target markers, right characters begin at target_end+3.
        right_start = target_end + 3
        right_end = sequence_length - 1  # exclude [SEP]
        if right_start < right_end:
            right = scores[right_start:right_end, right_start:right_end] >= threshold
            for start, end in np.argwhere(np.triu(right)):
                raw_start = int(start + right_start - 3)
                raw_end = int(end + right_start - 3)
                predictions.append([sentence_id, raw_start, raw_end])
                instance_predictions.append({
                    'instance_index': item['instance_index'],
                    'sentence_id': sentence_id,
                    'start': raw_start,
                    'end': raw_end,
                })
    return predictions, instance_predictions


def write_predictions(predictions, instance_predictions, output_dir, sidecar_name):
    os.makedirs(output_dir, exist_ok=True)
    data_json = '[\n' + ',\n'.join(
        '   ' + json.dumps(item, ensure_ascii=False) for item in predictions
    ) + '\n]'
    for name in ('A_task2_test.json', 'B_task2_test.json'):
        with open(os.path.join(output_dir, name), 'w', encoding='utf8', newline='\n') as handle:
            handle.write(data_json)
    with open(os.path.join(output_dir, sidecar_name), 'w', encoding='utf8') as handle:
        json.dump(instance_predictions, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    weights = [args.baseline_weight, args.target_weight,
               args.content_weight, args.boundary_weight]
    if any(value < 0 for value in weights):
        raise ValueError(f'ensemble weights must be non-negative: {weights}')
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError(f'ensemble weights must sum to 1, got {sum(weights)}')
    os.makedirs(args.prediction_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = Task2TestDataset(args.test_file, tokenizer)

    accumulated = []
    metadata = []
    specifications = [
        ('baseline', args.baseline_checkpoint, args.baseline_threshold, args.baseline_weight),
        ('target_conditioned', args.target_checkpoint, args.target_threshold, args.target_weight),
        ('content_aware', args.content_checkpoint, args.content_threshold, args.content_weight),
        ('boundary_auxiliary', args.boundary_checkpoint,
         args.boundary_threshold, args.boundary_weight),
    ]
    for variant, checkpoint, center, weight in specifications:
        if weight == 0.0:
            continue
        accumulate_model(
            args, dataset, device, variant, checkpoint, center, weight,
            accumulated, metadata,
            args.boundary_residual_beta if variant == 'boundary_auxiliary' else 0.0,
        )

    predictions, instance_predictions = decode_scores(
        accumulated, metadata, args.final_threshold
    )
    write_predictions(
        predictions, instance_predictions, args.prediction_dir, args.instance_sidecar_name
    )
    run_summary = {
        'device': str(device),
        'num_examples': len(dataset),
        'predicted_spans': len(predictions),
        'instance_sidecar': os.path.join(args.prediction_dir, args.instance_sidecar_name),
        'weights': {
            'B': args.baseline_weight,
            'D': args.target_weight,
            'E': args.content_weight,
            'H': args.boundary_weight,
        },
        'centering_thresholds': {
            'B': args.baseline_threshold,
            'D': args.target_threshold,
            'E': args.content_threshold,
            'H': args.boundary_threshold,
        },
        'boundary_residual_beta': args.boundary_residual_beta,
        'final_threshold': args.final_threshold,
        'test_file': args.test_file,
        'prediction_dir': args.prediction_dir,
    }
    with open(os.path.join(args.prediction_dir, 'ensemble_summary.json'),
              'w', encoding='utf8') as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
