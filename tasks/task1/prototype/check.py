#!/usr/bin/env python3
import argparse
import json

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prototype_file', default='./tasks/task1/assets/frame_prototypes/name-def-fes-mean.pt')
    p.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    p.add_argument('--train_file', default='./dataset/cfn-train.json')
    p.add_argument('--dev_file', default='./dataset/cfn-dev.json')
    args = p.parse_args()
    payload = torch.load(args.prototype_file, map_location='cpu')
    vectors = payload['prototypes']
    labels = payload['idx2label']
    frames = json.load(open(args.frame_info_file, encoding='utf8'))
    train = json.load(open(args.train_file, encoding='utf8'))
    dev = json.load(open(args.dev_file, encoding='utf8'))
    expected = [x['frame_name'] for x in frames]
    assert labels == expected, 'Prototype label order differs from frame_info.json.'
    assert vectors.shape == (len(expected), 768), vectors.shape
    assert torch.isfinite(vectors).all(), 'Prototype tensor contains NaN/Inf.'
    norms = vectors.norm(dim=-1)
    assert (norms > 0.99).all() and (norms < 1.01).all(), 'Prototypes are not normalized.'
    train_frames = {x['frame'] for x in train}
    unseen_dev = [x for x in dev if x['frame'] not in train_frames]
    similarity = vectors @ vectors.t()
    similarity.fill_diagonal_(-1)
    top_values, top_indices = similarity.max(dim=1)
    print({
        'shape': list(vectors.shape),
        'norm_min': float(norms.min()),
        'norm_max': float(norms.max()),
        'unique_labels': len(set(labels)),
        'train_seen_frame_types': len(train_frames),
        'unseen_frame_types': len(set(labels) - train_frames),
        'dev_unseen_examples': len(unseen_dev),
        'mean_nearest_other_cosine': float(top_values.mean()),
        'max_nearest_other_cosine': float(top_values.max()),
    })
    print('[OK] Frame prototype checks passed.')


if __name__ == '__main__':
    main()

