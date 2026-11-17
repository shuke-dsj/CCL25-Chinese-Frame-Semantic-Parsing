#!/usr/bin/env python3
"""Precompute fixed semantic prototypes from frame_info.json."""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer

from .utils import load_frame_texts


def load_bert_weights(model, checkpoint):
    state = torch.load(checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    converted = {}
    for key, value in state.items():
        if key.startswith('cls.'):
            continue
        if key.startswith('bert.'):
            key = key[len('bert.'):]
        converted[key] = value
    return model.load_state_dict(converted, strict=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    p.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    p.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    p.add_argument('--init_checkpoint', default='./chinese_bert_wwm_ext/pytorch_model.bin')
    p.add_argument('--output_file', default='./tasks/task1/assets/frame_prototypes/name-def-fes-mean.pt')
    p.add_argument('--max_length', default=384, type=int)
    p.add_argument('--max_fes', default=16, type=int)
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--pooling', default='mean', choices=['mean', 'cls'])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames, labels, texts = load_frame_texts(args.frame_info_file, args.max_fes)
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    config = BertConfig.from_json_file(args.config_file)
    model = BertModel(config)
    msg = load_bert_weights(model, args.init_checkpoint)
    model = model.to(device).eval()

    loader = DataLoader(texts, batch_size=args.batch_size, shuffle=False)
    vectors = []
    truncated = 0
    with torch.no_grad():
        for batch_texts in loader:
            raw_lengths = [len(tokenizer.tokenize(text)) + 2 for text in batch_texts]
            truncated += sum(length > args.max_length for length in raw_lengths)
            encoded = tokenizer(list(batch_texts), padding=True, truncation=True,
                                max_length=args.max_length, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded).last_hidden_state
            if args.pooling == 'cls':
                pooled = out[:, 0]
            else:
                mask = encoded['attention_mask'].unsqueeze(-1).to(out.dtype)
                # Excluding special tokens is not essential; mean pooling is kept deterministic.
                pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    prototypes = torch.cat(vectors, dim=0)
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    payload = {
        'prototypes': prototypes,
        'idx2label': labels,
        'texts': texts,
        'metadata': vars(args),
    }
    torch.save(payload, args.output_file)
    print(json.dumps({
        'output_file': args.output_file,
        'shape': list(prototypes.shape),
        'num_frames': len(labels),
        'truncated_descriptions': truncated,
        'missing_keys': len(msg.missing_keys),
        'unexpected_keys': len(msg.unexpected_keys),
        'device': str(device),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


