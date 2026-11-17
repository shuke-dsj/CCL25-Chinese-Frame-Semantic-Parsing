#!/usr/bin/env python3
"""One-batch compatibility/gradient smoke test for Task2 experiment H1."""
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from .dataset import Dataset
from .model import Model as BaselineModel
from .model_boundary_auxiliary import BoundaryAuxiliaryModel
from .train import get_model_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    parser.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    parser.add_argument(
        '--checkpoint',
        default='./tasks/task2/experiment_record/task2-valid-mask-fgm-5ep-result/checkpoints/task2_best.bin',
    )
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--boundary_loss_weight', default=0.1, type=float)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = Dataset(
        './dataset/cfn-train.json', './dataset/frame_info.json', tokenizer
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(get_model_input, device=device), drop_last=False,
    )
    input_ids, attention_mask, target, labels, _, target_cls = next(iter(loader))

    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = 1
    config.max_cls = dataset.max_cls
    config.task2_valid_loss_mask = True
    config.task2_boundary_loss_weight = args.boundary_loss_weight
    baseline = BaselineModel(config)
    boundary = BoundaryAuxiliaryModel(config)
    state = torch.load(args.checkpoint, map_location='cpu')
    baseline_message = baseline.load_state_dict(state, strict=True)
    boundary_message = boundary.load_state_dict(state, strict=False)
    del state
    expected_missing = {
        'boundary_start.weight', 'boundary_start.bias',
        'boundary_end.weight', 'boundary_end.bias',
    }
    if set(boundary_message.missing_keys) != expected_missing:
        raise AssertionError(
            f'unexpected H1 missing keys: {boundary_message.missing_keys}'
        )
    if boundary_message.unexpected_keys:
        raise AssertionError(
            f'unexpected H1 checkpoint keys: {boundary_message.unexpected_keys}'
        )

    baseline = baseline.to(device).eval()
    boundary = boundary.to(device).eval()
    with torch.no_grad():
        baseline_output = baseline(
            input_ids=input_ids, attention_mask=attention_mask, target=target,
            target_cls=target_cls, labels=labels, device=device,
        )
    boundary_output = boundary(
        input_ids=input_ids, attention_mask=attention_mask, target=target,
        target_cls=target_cls, labels=labels, device=device,
    )
    max_logit_difference = (
        baseline_output['logits'] - boundary_output['logits']
    ).abs().max().item()
    expected_total = (
        boundary_output['span_loss']
        + args.boundary_loss_weight * boundary_output['boundary_loss']
    )
    loss_difference = (boundary_output['loss'] - expected_total).abs().item()
    boundary_output['loss'].backward()
    gradient_norms = {
        name: float(parameter.grad.norm().item())
        for name, parameter in boundary.named_parameters()
        if name.startswith('boundary_') and parameter.grad is not None
    }
    result = {
        'device': str(device),
        'logits_shape': list(boundary_output['logits'].shape),
        'span_loss': float(boundary_output['span_loss'].item()),
        'boundary_loss': float(boundary_output['boundary_loss'].item()),
        'weighted_total_loss': float(boundary_output['loss'].item()),
        'loss_identity_difference': loss_difference,
        'max_logit_difference_from_B': max_logit_difference,
        'positive_starts': int(boundary_output['boundary_start_labels'].sum().item()),
        'positive_ends': int(boundary_output['boundary_end_labels'].sum().item()),
        'valid_start_positions': int(boundary_output['boundary_start_valid'].sum().item()),
        'valid_end_positions': int(boundary_output['boundary_end_valid'].sum().item()),
        'missing_from_B': boundary_message.missing_keys,
        'baseline_missing': baseline_message.missing_keys,
        'gradient_norms': gradient_norms,
    }
    print(result)
    assert max_logit_difference < 1e-7
    assert loss_difference < 1e-6
    assert torch.isfinite(boundary_output['loss'])
    assert set(gradient_norms) == expected_missing
    assert all(value > 0 for value in gradient_norms.values())
    print('[OK] H1 boundary auxiliary forward/backward and B compatibility passed.')


if __name__ == '__main__':
    main()
