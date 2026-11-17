#!/usr/bin/env python3
"""One-batch forward/backward smoke test for Target-conditioned GlobalPointer."""

import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from .dataset import Dataset
from .model_factory import build_task2_model
from .train import get_model_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    parser.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    parser.add_argument('--init_checkpoint', default='./chinese_bert_wwm_ext/pytorch_model.bin')
    parser.add_argument('--batch_size', default=2, type=int)
    smoke_args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=smoke_args.vocab_file, do_lower_case=True)
    dataset = Dataset('./dataset/cfn-train.json', './dataset/frame_info.json', tokenizer)
    loader = DataLoader(
        dataset, batch_size=smoke_args.batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(get_model_input, device=device), drop_last=False
    )
    input_ids, attention_mask, target, labels, _, _ = next(iter(loader))

    config = BertConfig.from_json_file(smoke_args.config_file)
    config.num_labels = 1
    config.max_cls = dataset.max_cls
    config.task2_valid_loss_mask = True
    model = build_task2_model(config, 'target_conditioned')
    load_message = model.load_state_dict(
        torch.load(smoke_args.init_checkpoint, map_location='cpu'), strict=False
    )
    model = model.to(device)

    model.train()
    output = model(
        input_ids=input_ids, attention_mask=attention_mask, target=target,
        labels=labels, device=device
    )
    output['loss'].backward()
    projection_grad = model.target_projection.weight.grad
    gate_grad = model.target_gate.weight.grad

    model.eval()
    alternative_target = list(reversed(target)) if len(target) > 1 else [[target[0][0], target[0][0]]]
    with torch.no_grad():
        original_logits = model(
            input_ids=input_ids, attention_mask=attention_mask, target=target,
            labels=None, device=device, for_test=True
        )['logits']
        alternative_logits = model(
            input_ids=input_ids, attention_mask=attention_mask, target=alternative_target,
            labels=None, device=device, for_test=True
        )['logits']
    target_sensitivity = (original_logits - alternative_logits).abs().max().item()

    result = {
        'device': str(device),
        'logits_shape': list(original_logits.shape),
        'loss': float(output['loss'].detach()),
        'target_sensitivity_max_diff': target_sensitivity,
        'target_projection_grad_norm': float(projection_grad.norm()),
        'target_gate_grad_norm': float(gate_grad.norm()),
        'missing_keys': len(load_message.missing_keys),
        'unexpected_keys': len(load_message.unexpected_keys),
        'new_missing_key_examples': [
            key for key in load_message.missing_keys if key.startswith('target_')
        ][:10],
    }
    print(result)
    assert torch.isfinite(output['loss'])
    assert projection_grad is not None and torch.isfinite(projection_grad).all()
    assert gate_grad is not None and torch.isfinite(gate_grad).all()
    assert float(projection_grad.norm()) > 0.0
    assert float(gate_grad.norm()) > 0.0
    assert target_sensitivity > 1e-7, 'Changing the explicit target did not change logits.'
    print('[OK] Task2 Target-conditioned GlobalPointer smoke test passed.')


if __name__ == '__main__':
    main()

