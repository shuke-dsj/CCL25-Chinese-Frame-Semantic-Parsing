#!/usr/bin/env python3
"""Smoke test for definition-guided multi-role GlobalPointer (G0)."""
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader, Subset
from transformers import BertConfig, BertTokenizer

from .dataset_frame_guided import FrameGuidedDataset, frame_guided_collate
from .model_frame_guided import FrameGuidedGlobalPointer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    parser.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    parser.add_argument('--data_file', default='./dataset/cfn-dev.json')
    parser.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--task2_role_query_dim', type=int, default=64)
    parser.add_argument('--task2_role_scale_init', type=float, default=0.1)
    parser.add_argument(
        '--baseline_checkpoint',
        default='./tasks/task2/experiment_record/task2-valid-mask-fgm-5ep-result/checkpoints/task2_best.bin',
    )
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = FrameGuidedDataset(
        args.data_file, args.frame_info_file, tokenizer, max_length=args.max_length
    )
    subset = Subset(dataset, range(args.batch_size))
    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=False,
        collate_fn=partial(
            frame_guided_collate, pad_token_id=tokenizer.pad_token_id, device=device
        ),
    )
    batch = next(iter(loader))

    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = 1
    config.task2_role_query_dim = args.task2_role_query_dim
    config.task2_role_scale_init = args.task2_role_scale_init
    model = FrameGuidedGlobalPointer(config)
    state = torch.load(args.baseline_checkpoint, map_location='cpu')
    message = model.load_state_dict(state, strict=False)
    if message.unexpected_keys:
        raise RuntimeError(f'unexpected baseline keys: {message.unexpected_keys}')
    expected_missing_prefixes = (
        'role_scale', 'role_start_query.', 'role_end_query.',
        'role_start_token.', 'role_end_token.',
    )
    unknown_missing = [
        key for key in message.missing_keys
        if not key.startswith(expected_missing_prefixes)
    ]
    if unknown_missing:
        raise RuntimeError(f'unexpected missing keys: {unknown_missing}')
    model = model.to(device).train()
    output = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        target=batch['target'],
        sentence_end=batch['sentence_end'],
        role_spans=batch['role_spans'],
        role_mask=batch['role_mask'],
        labels=batch['labels'],
        device=device,
    )
    if not torch.isfinite(output['loss']):
        raise RuntimeError(f'non-finite loss: {output["loss"]}')
    output['loss'].backward()
    gradient_names = [
        'role_start_query.weight', 'role_end_query.weight',
        'role_start_token.weight', 'role_end_token.weight', 'role_scale',
    ]
    gradient_norms = {}
    named_parameters = dict(model.named_parameters())
    for name in gradient_names:
        gradient = named_parameters[name].grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f'invalid gradient for {name}')
        gradient_norms[name] = float(gradient.norm().detach().cpu())

    result = {
        'device': str(device),
        'input_shape': list(batch['input_ids'].shape),
        'sentence_matrix_shape': list(output['logits'].shape),
        'role_logits_shape': list(output['role_logits'].shape),
        'active_roles': [int(value) for value in batch['role_mask'].sum(dim=1).cpu()],
        'positive_role_spans': int(batch['labels'].sum().item()),
        'loss': float(output['loss'].detach().cpu()),
        'role_scale': float(output['role_scale'].detach().cpu()),
        'missing_from_baseline': len(message.missing_keys),
        'unexpected_from_baseline': len(message.unexpected_keys),
        'gradient_norms': gradient_norms,
    }
    print(result)
    print('[OK] Frame-guided G0 forward/backward and baseline weight compatibility passed.')


if __name__ == '__main__':
    main()
