#!/usr/bin/env python3
"""Smoke test for the cheap local oracle-frame prototype experiment G2."""
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader, Subset
from transformers import BertConfig, BertTokenizer

from .dataset import Dataset
from .model_frame_prototype import FramePrototypeConditionedModel


def collate(batch, device=None):
    max_length = max(len(item[0]) for item in batch)
    pad = lambda values: values + [0] * (max_length - len(values))
    input_ids = torch.tensor([pad(item[0]) for item in batch], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [pad(item[1]) for item in batch], dtype=torch.long, device=device
    )
    labels = torch.zeros(len(batch), max_length, max_length, device=device)
    for batch_index, item in enumerate(batch):
        for start, end in item[3]:
            labels[batch_index, start, end] = 1.0
    return (
        input_ids, attention_mask, [item[2] for item in batch], labels,
        [item[4] for item in batch],
        torch.tensor([item[5] for item in batch], dtype=torch.long, device=device),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    parser.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    parser.add_argument('--data_file', default='./dataset/cfn-dev.json')
    parser.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    parser.add_argument(
        '--prototype_file',
        default='./tasks/task1/assets/frame_prototypes/name-def-fes-mean.pt',
    )
    parser.add_argument(
        '--baseline_checkpoint',
        default='./tasks/task2/experiment_record/task2-valid-mask-fgm-5ep-result/checkpoints/task2_best.bin',
    )
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = Dataset(args.data_file, args.frame_info_file, tokenizer)
    loader = DataLoader(
        Subset(dataset, range(args.batch_size)), batch_size=args.batch_size,
        collate_fn=partial(collate, device=device), shuffle=False,
    )
    input_ids, attention_mask, target, labels, _, target_cls = next(iter(loader))
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = 1
    config.max_cls = dataset.max_cls
    config.task2_valid_loss_mask = True
    config.task2_frame_prototype_file = args.prototype_file
    config.task2_frame_query_dim = 64
    config.task2_frame_scale_init = 0.1
    config.task2_frame_names = list(dataset.label2cls.keys())
    model = FramePrototypeConditionedModel(config)
    message = model.load_state_dict(
        torch.load(args.baseline_checkpoint, map_location='cpu'), strict=False
    )
    if message.unexpected_keys:
        raise RuntimeError(f'unexpected keys: {message.unexpected_keys}')
    model = model.to(device).train()
    output = model(
        input_ids=input_ids, attention_mask=attention_mask, target=target,
        target_cls=target_cls, labels=labels, device=device,
    )
    output['loss'].backward()
    gradients = {}
    for name in ('frame_start_query.weight', 'frame_end_query.weight',
                 'frame_start_token.weight', 'frame_end_token.weight', 'frame_scale'):
        gradient = dict(model.named_parameters())[name].grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f'invalid gradient: {name}')
        gradients[name] = float(gradient.norm().detach().cpu())
    print({
        'device': str(device),
        'logits_shape': list(output['logits'].shape),
        'loss': float(output['loss'].detach().cpu()),
        'frame_scale': float(model.frame_scale.detach().cpu()),
        'missing': len(message.missing_keys),
        'unexpected': len(message.unexpected_keys),
        'gradient_norms': gradients,
    })
    print('[OK] G2 oracle-frame prototype forward/backward passed.')


if __name__ == '__main__':
    main()
