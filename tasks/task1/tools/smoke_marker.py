#!/usr/bin/env python3
"""One-batch forward/backward smoke test for task1 target markers."""
import argparse
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from tasks.task1.dataset import Dataset
from tasks.task1.model import Model


def collate(data, device=None):
    max_len = max(len(x[0]) for x in data)
    pad = lambda seq: seq + [0] * (max_len - len(seq))
    input_ids = torch.tensor([pad(x[0]) for x in data], dtype=torch.long, device=device)
    attention_mask = torch.tensor([pad(x[1]) for x in data], dtype=torch.long, device=device)
    target = [x[2] for x in data]
    labels = torch.tensor([x[3] for x in data], dtype=torch.long, device=device)
    sentence_ids = [x[4] for x in data]
    return input_ids, attention_mask, target, labels, sentence_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    p.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    p.add_argument('--init_checkpoint', default='./chinese_bert_wwm_ext/pytorch_model.bin')
    p.add_argument('--batch_size', default=2, type=int)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = Dataset('./dataset/cfn-train.json', './dataset/frame_info.json', tokenizer,
                      target_markers=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=partial(collate, device=device), drop_last=False)

    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = dataset.num_labels
    config.model_variant = 'target_aware'
    config.target_aware_layers = 'last'
    model = Model(config)
    state = torch.load(args.init_checkpoint, map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.train()

    input_ids, attention_mask, target, labels, sentence_ids = next(iter(loader))
    output = model(input_ids=input_ids, attention_mask=attention_mask, target=target,
                   labels=labels, device=device)
    loss = output['loss']
    loss.backward()

    embedding_grad = model.bert.embeddings.word_embeddings.weight.grad
    start_grad = embedding_grad[dataset.target_start_token_id].norm().item()
    end_grad = embedding_grad[dataset.target_end_token_id].norm().item()
    print({
        'device': str(device),
        'batch_shape': list(input_ids.shape),
        'logits_shape': list(output['logits'].shape),
        'loss': float(loss.item()),
        'sentence_ids': sentence_ids,
        'target_positions': target,
        'start_marker_id': dataset.target_start_token_id,
        'end_marker_id': dataset.target_end_token_id,
        'start_marker_grad_norm': start_grad,
        'end_marker_grad_norm': end_grad,
        'missing_keys': len(msg.missing_keys),
        'unexpected_keys': len(msg.unexpected_keys),
    })
    if output['logits'].shape != (args.batch_size, dataset.num_labels):
        raise AssertionError('Unexpected logits shape.')
    if not torch.isfinite(loss):
        raise AssertionError('Loss is not finite.')
    if start_grad <= 0 or end_grad <= 0:
        raise AssertionError('Marker embeddings did not receive gradients.')
    print('[OK] Target-marker one-batch forward/backward smoke test passed.')


if __name__ == '__main__':
    main()

