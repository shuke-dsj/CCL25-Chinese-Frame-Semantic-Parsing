#!/usr/bin/env python3
"""Generate task1 test predictions by probability-averaging RoPE and target-aware models."""
import argparse
import codecs
import json
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from tasks.task1.model import Model


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, label_file, tokenizer):
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            labels = json.load(f)
        self.idx2label = [x['frame_name'] for x in labels]
        self.num_labels = len(self.idx2label)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        row = self.all_data[item]
        encoded = self.tokenizer.encode_plus(list(row['text']))
        target = [row['target'][-1]['start'] + 1, row['target'][-1]['end'] + 1]
        return encoded.data['input_ids'], encoded.data['attention_mask'], target, row['sentence_id']


def collate(data, device=None):
    max_len = max(len(x[0]) for x in data)
    pad = lambda seq: seq + [0] * (max_len - len(seq))
    return (
        torch.tensor([pad(x[0]) for x in data], dtype=torch.long, device=device),
        torch.tensor([pad(x[1]) for x in data], dtype=torch.long, device=device),
        [x[2] for x in data],
        [x[3] for x in data],
    )


def load_model(checkpoint, args, num_labels, variant, aware_layers, device):
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = num_labels
    config.model_variant = variant
    config.target_aware_layers = aware_layers
    model = Model(config)
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=False)
    return model.to(device)


def collect_probabilities(model, loader, device, desc):
    model.eval()
    probs, ids = [], []
    with torch.no_grad():
        for input_ids, attention_mask, target, sentence_ids in tqdm(loader, desc=desc):
            out = model(input_ids=input_ids, attention_mask=attention_mask, target=target,
                        labels=None, device=device, for_test=True)
            probs.append(torch.softmax(out['logits'], dim=-1).cpu())
            ids.extend(sentence_ids)
    return torch.cat(probs).numpy(), ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline_checkpoint', required=True)
    p.add_argument('--aware_checkpoint', required=True)
    p.add_argument('--baseline_weight', default=0.75, type=float)
    p.add_argument('--aware_layers', default='last', choices=['last', 'last4'])
    p.add_argument('--batch_size', default=3, type=int)
    p.add_argument('--test_file', default='./dataset/cfn-test-B.json')
    p.add_argument('--label_file', default='./dataset/frame_info.json')
    p.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    p.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    p.add_argument('--output_file', default='./tasks/task1/predictions/ensemble/A_task1_test.json')
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.baseline_weight <= 1:
        raise ValueError('--baseline_weight must be between 0 and 1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = TestDataset(args.test_file, args.label_file, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=partial(collate, device=device), drop_last=False)

    baseline = load_model(args.baseline_checkpoint, args, dataset.num_labels, 'rope', 'last', device)
    base_prob, ids = collect_probabilities(baseline, loader, device, 'baseline')
    del baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    aware = load_model(args.aware_checkpoint, args, dataset.num_labels,
                       'target_aware', args.aware_layers, device)
    aware_prob, ids2 = collect_probabilities(aware, loader, device, 'target-aware')
    if ids != ids2:
        raise RuntimeError('Model outputs are not aligned.')

    weight = args.baseline_weight
    pred = (weight * base_prob + (1 - weight) * aware_prob).argmax(axis=1)
    output = [[ids[i], dataset.idx2label[int(pred[i])]] for i in range(len(ids))]
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w', encoding='utf8', newline='\n') as f:
        json.dump(output, f, indent=1, ensure_ascii=False)
    print(f'wrote {len(output)} predictions to {args.output_file}')
    print(f'weights: baseline={weight:.3f}, target-aware={1-weight:.3f}')


if __name__ == '__main__':
    main()

