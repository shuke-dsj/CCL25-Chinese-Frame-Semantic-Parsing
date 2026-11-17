#!/usr/bin/env python3
"""Train task1 with fixed semantic frame-definition prototypes."""
import argparse
import json
import time
from collections import Counter
from functools import partial

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from tasks.task1.dataset import Dataset
from tasks.common.experiment_utils import ExperimentLogger, set_seed
from .model import FramePrototypeModel


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment_name', default='frame-prototype-mvp-v1')
    p.add_argument('--result_dir', default=None)
    p.add_argument('--num_train_epochs', default=10, type=int)
    p.add_argument('--batch_size', default=3, type=int)
    p.add_argument('--encoder_lr', default=2e-5, type=float)
    p.add_argument('--head_lr', default=1e-4, type=float)
    p.add_argument('--warmup_proportion', default=0.1, type=float)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--log_interval', default=100, type=int)
    p.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    p.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    p.add_argument('--init_checkpoint', default='./chinese_bert_wwm_ext/pytorch_model.bin')
    p.add_argument('--prototype_file', default='./tasks/task1/assets/frame_prototypes/name-def-fes-mean.pt')
    p.add_argument('--train_file', default='./dataset/cfn-train.json')
    p.add_argument('--dev_file', default='./dataset/cfn-dev.json')
    p.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    return p.parse_args()


def collate(data, device=None):
    max_len = max(len(x[0]) for x in data)
    pad = lambda seq: seq + [0] * (max_len - len(seq))
    return (
        torch.tensor([pad(x[0]) for x in data], dtype=torch.long, device=device),
        torch.tensor([pad(x[1]) for x in data], dtype=torch.long, device=device),
        [x[2] for x in data],
        torch.tensor([x[3] for x in data], dtype=torch.long, device=device),
        [x[4] for x in data],
    )


def load_pretrained(model, checkpoint):
    state = torch.load(checkpoint, map_location='cpu')
    return model.load_state_dict(state, strict=False)


def bucket(n):
    if n == 0: return 'unseen'
    if n <= 5: return 'rare'
    if n <= 15: return 'few'
    if n <= 50: return 'medium'
    return 'head'


def evaluate(model, loader, train_freq, idx2label):
    model.eval()
    total = correct = top5_correct = 0
    loss_sum = 0.0
    stats = {k: [0, 0, 0] for k in ['unseen', 'rare', 'few', 'medium', 'head']}
    with torch.no_grad():
        for input_ids, attention_mask, target, labels, _ in tqdm(loader, desc='eval'):
            out = model(input_ids, attention_mask, target, labels=labels, for_test=True)
            logits = out['logits']
            loss_sum += torch.nn.functional.cross_entropy(logits, labels).item()
            pred = logits.argmax(-1)
            top5 = logits.topk(5, dim=-1).indices
            for i, gold in enumerate(labels.tolist()):
                name = idx2label[gold]
                b = bucket(train_freq[name])
                ok = int(pred[i].item() == gold)
                ok5 = int(gold in top5[i].tolist())
                stats[b][0] += ok; stats[b][1] += ok5; stats[b][2] += 1
                correct += ok; top5_correct += ok5; total += 1
    unseen_correct, unseen_top5, unseen_count = stats['unseen']
    seen_count = total - unseen_count
    metrics = {
        'val_loss': loss_sum / max(len(loader), 1),
        'val_acc': correct / max(total, 1),
        'val_top5_acc': top5_correct / max(total, 1),
        'val_seen_acc': (correct - unseen_correct) / max(seen_count, 1),
        'val_seen_top5': (top5_correct - unseen_top5) / max(seen_count, 1),
    }
    for name, (c1, c5, n) in stats.items():
        metrics[f'val_{name}_acc'] = c1 / n if n else 0.0
        metrics[f'val_{name}_top5'] = c5 / n if n else 0.0
        metrics[f'val_{name}_count'] = n
    return metrics


def main():
    args = args_parser()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    train_set = Dataset(args.train_file, args.frame_info_file, tokenizer)
    dev_set = Dataset(args.dev_file, args.frame_info_file, tokenizer)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              collate_fn=partial(collate, device=device), drop_last=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            collate_fn=partial(collate, device=device), drop_last=False)

    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = train_set.num_labels
    model = FramePrototypeModel(config, args.prototype_file)
    msg = load_pretrained(model, args.init_checkpoint)
    model = model.to(device)

    with open(args.train_file, encoding='utf8') as f:
        train_rows = json.load(f)
    train_freq = Counter(x['frame'] for x in train_rows)
    seen_ids = torch.tensor(sorted(train_set.label2idx[x] for x in train_freq), dtype=torch.long, device=device)
    global_to_seen = torch.full((train_set.num_labels,), -1, dtype=torch.long, device=device)
    global_to_seen[seen_ids] = torch.arange(len(seen_ids), device=device)

    encoder_params = list(model.bert.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith('bert.')]
    optimizer = AdamW([
        {'params': encoder_params, 'lr': args.encoder_lr, 'weight_decay': 0.01},
        {'params': head_params, 'lr': args.head_lr, 'weight_decay': 0.01},
    ])
    total_steps = len(train_loader) * args.num_train_epochs
    warmup = int(total_steps * args.warmup_proportion)
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        return max((total_steps - step) / max(total_steps - warmup, 1), 0.0)
    scheduler = LambdaLR(optimizer, lr_lambda)

    fields = ['epoch','train_loss','val_loss','val_acc','val_top5_acc','val_seen_acc','val_seen_top5','val_unseen_acc','val_unseen_top5',
              'val_rare_acc','val_few_acc','val_medium_acc','val_head_acc','best_val_acc','best_unseen_acc','best_unseen_top5','epoch_seconds',
              'encoder_lr','head_lr','peak_gpu_memory_mb']
    logger = ExperimentLogger(args, 'task1_prototype', fields)
    best = 0.0
    best_unseen = 0.0
    best_unseen_top5 = 0.0
    print(f'[prototype config] frames={train_set.num_labels} seen={len(seen_ids)} unseen={train_set.num_labels-len(seen_ids)}')
    print(f'[weight load] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')
    print('[missing keys]', msg.missing_keys)
    print('[unexpected keys]', msg.unexpected_keys)

    for epoch in range(1, args.num_train_epochs + 1):
        start_time = time.perf_counter()
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        model.train(); total_loss = 0.0
        bar = tqdm(train_loader, desc=f'epoch_{epoch}')
        for step, (input_ids, attention_mask, target, labels, _) in enumerate(bar, 1):
            out = model(input_ids, attention_mask, target, labels=labels,
                        seen_label_ids=seen_ids, global_to_seen=global_to_seen)
            loss = out['loss']; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            if args.log_interval > 0 and step % args.log_interval == 0:
                bar.set_postfix(loss=total_loss / step)
        metrics = evaluate(model, dev_loader, train_freq, dev_set.idx2label)
        if metrics['val_acc'] > best:
            best = metrics['val_acc']
            torch.save(model.state_dict(), logger.checkpoint_path('task1_prototype'))
        if metrics['val_unseen_acc'] > best_unseen:
            best_unseen = metrics['val_unseen_acc']
            torch.save(model.state_dict(), logger.checkpoint_dir / 'task1_prototype_best_unseen.bin')
        if metrics['val_unseen_top5'] > best_unseen_top5:
            best_unseen_top5 = metrics['val_unseen_top5']
            torch.save(model.state_dict(), logger.checkpoint_dir / 'task1_prototype_best_unseen_top5.bin')
        metrics.update({
            'epoch': epoch, 'train_loss': total_loss / max(len(train_loader), 1), 'best_val_acc': best,
            'best_unseen_acc': best_unseen, 'best_unseen_top5': best_unseen_top5,
            'epoch_seconds': time.perf_counter() - start_time,
            'encoder_lr': optimizer.param_groups[0]['lr'], 'head_lr': optimizer.param_groups[1]['lr'],
            'peak_gpu_memory_mb': torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0.0,
        })
        logger.log(metrics)
        print(metrics)


if __name__ == '__main__':
    main()


