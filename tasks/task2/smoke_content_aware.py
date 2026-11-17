#!/usr/bin/env python3
"""One-batch forward/backward checks for the Task2 content-aware scorer."""

import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from .dataset import Dataset
from .model_factory import build_task2_model
from .train import FGM, get_model_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    parser.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    parser.add_argument('--init_checkpoint', default='./chinese_bert_wwm_ext/pytorch_model.bin')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--task2_content_dim', default=32, type=int)
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
    config.task2_content_dim = smoke_args.task2_content_dim
    config.task2_content_scale_init = 0.1
    model = build_task2_model(config, 'content_aware')
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
    fgm = FGM(model)
    fgm.attack()
    adversarial_output = model(
        input_ids=input_ids, attention_mask=attention_mask, target=target,
        labels=labels, device=device
    )
    adversarial_output['loss'].backward()
    fgm.restore()
    content_grad = model.content_projection.weight.grad
    target_grad = model.content_target_projection.weight.grad
    scale_grad = model.content_scale.grad

    model.eval()
    with torch.no_grad():
        original_scale = float(model.content_scale)
        enabled_logits = model(
            input_ids=input_ids, attention_mask=attention_mask, target=target,
            labels=None, device=device, for_test=True
        )['logits']
        model.content_scale.zero_()
        baseline_logits = model(
            input_ids=input_ids, attention_mask=attention_mask, target=target,
            labels=None, device=device, for_test=True
        )['logits']
        model.content_scale.fill_(original_scale)
        alternative_target = list(reversed(target))
        alternative_logits = model(
            input_ids=input_ids, attention_mask=attention_mask, target=alternative_target,
            labels=None, device=device, for_test=True
        )['logits']

    branch_effect = (enabled_logits - baseline_logits).abs().max().item()
    target_sensitivity = (enabled_logits - alternative_logits).abs().max().item()
    result = {
        'device': str(device),
        'logits_shape': list(enabled_logits.shape),
        'loss': float(output['loss'].detach()),
        'adversarial_loss': float(adversarial_output['loss'].detach()),
        'content_dim': model.content_dim,
        'content_scale': float(model.content_scale.detach()),
        'content_branch_max_effect': branch_effect,
        'target_sensitivity_max_diff': target_sensitivity,
        'content_projection_grad_norm': float(content_grad.norm()),
        'target_projection_grad_norm': float(target_grad.norm()),
        'content_scale_grad': float(scale_grad),
        'missing_keys': len(load_message.missing_keys),
        'unexpected_keys': len(load_message.unexpected_keys),
        'new_missing_key_examples': [
            key for key in load_message.missing_keys if key.startswith('content_')
        ][:10],
    }
    print(result)
    assert torch.isfinite(output['loss'])
    assert torch.isfinite(adversarial_output['loss'])
    assert content_grad is not None and torch.isfinite(content_grad).all()
    assert target_grad is not None and torch.isfinite(target_grad).all()
    assert scale_grad is not None and torch.isfinite(scale_grad)
    assert float(content_grad.norm()) > 0.0
    assert float(target_grad.norm()) > 0.0
    assert branch_effect > 1e-7, 'Non-zero content scale did not change logits.'
    assert target_sensitivity > 1e-7, 'Changing the target did not change content-aware logits.'
    print('[OK] Task2 additive content-aware span scorer smoke test passed.')


if __name__ == '__main__':
    main()
