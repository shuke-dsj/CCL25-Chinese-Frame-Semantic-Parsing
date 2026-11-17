#!/usr/bin/env python3
"""G1: oracle-frame Definition-guided Task2/Task3 training."""
import argparse
import math
import time
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from tasks.common.experiment_utils import ExperimentLogger, save_checkpoint, set_seed
from .dataset_frame_guided import FrameGuidedDataset, frame_guided_collate
from .decoding import build_safe_span_mask
from .model_frame_guided import FrameGuidedGlobalPointer


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='embeddings'):
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None and emb_name in name:
                norm = torch.norm(parameter.grad)
                if torch.isfinite(norm) and norm > 0:
                    self.backup[name] = parameter.data.clone()
                    parameter.data.add_(epsilon * parameter.grad / norm)

    def restore(self):
        for name, value in self.backup.items():
            dict(self.model.named_parameters())[name].data.copy_(value)
        self.backup = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default='task2-G1-oracle-frame-definition-guided-5ep')
    parser.add_argument('--result_dir', default=None)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--accumulate_gradients', type=int, default=1)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--head_lr', type=float, default=5e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--disable_fgm', action='store_true')
    parser.add_argument('--fgm_epsilon', type=float, default=1.0)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--task2_role_query_dim', type=int, default=64)
    parser.add_argument('--task2_role_scale_init', type=float, default=0.1)
    parser.add_argument('--config_file', default='./chinese_bert_wwm_ext/config.json')
    parser.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    parser.add_argument('--train_file', default='./dataset/cfn-train.json')
    parser.add_argument('--dev_file', default='./dataset/cfn-dev.json')
    parser.add_argument('--frame_info_file', default='./dataset/frame_info.json')
    parser.add_argument(
        '--warm_start_checkpoint',
        default='./tasks/task2/experiment_record/task2-valid-mask-fgm-5ep-result/checkpoints/task2_best.bin',
    )
    return parser.parse_args()


def token_coverage_counts(prediction, gold):
    true_positive = predicted = target = 0
    for index in range(prediction.shape[0]):
        length = prediction.shape[-1]
        pred_coverage = torch.zeros(length, dtype=torch.bool, device=prediction.device)
        gold_coverage = torch.zeros(length, dtype=torch.bool, device=prediction.device)
        for start, end in torch.nonzero(prediction[index], as_tuple=False).tolist():
            pred_coverage[start:end + 1] = True
        for start, end in torch.nonzero(gold[index], as_tuple=False).tolist():
            gold_coverage[start:end + 1] = True
        true_positive += int((pred_coverage & gold_coverage).sum().item())
        predicted += int(pred_coverage.sum().item())
        target += int(gold_coverage.sum().item())
    return true_positive, predicted, target


def prf(correct, predicted, gold):
    precision = correct / max(predicted, 1e-9)
    recall = correct / max(gold, 1e-9)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def sentence_attention(sentence_end, length):
    positions = torch.arange(length, device=sentence_end.device).unsqueeze(0)
    return (positions <= sentence_end.unsqueeze(1)).long()


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    task_exact = [0, 0, 0]
    task_token = [0, 0, 0]
    role_exact = [0, 0, 0]
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval'):
            output = model(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                target=batch['target'], sentence_end=batch['sentence_end'],
                role_spans=batch['role_spans'], role_mask=batch['role_mask'],
                labels=batch['labels'], device=device,
            )
            total_loss += float(output['loss'].item())
            length = batch['labels'].shape[-1]
            sent_attention = sentence_attention(batch['sentence_end'], length)
            valid_span = build_safe_span_mask(sent_attention, batch['target'])
            role_valid = batch['role_mask'].unsqueeze(-1).unsqueeze(-1) & valid_span.unsqueeze(1)
            role_prediction = (output['role_logits'] >= 0) & role_valid
            role_gold = batch['labels'].bool()
            role_exact[0] += int((role_prediction & role_gold).sum().item())
            role_exact[1] += int(role_prediction.sum().item())
            role_exact[2] += int(role_gold.sum().item())

            task_prediction = role_prediction.any(dim=1)
            task_gold = role_gold.any(dim=1)
            task_exact[0] += int((task_prediction & task_gold).sum().item())
            task_exact[1] += int(task_prediction.sum().item())
            task_exact[2] += int(task_gold.sum().item())
            counts = token_coverage_counts(task_prediction, task_gold)
            for item in range(3):
                task_token[item] += counts[item]

    exact_p, exact_r, exact_f1 = prf(*task_exact)
    token_p, token_r, token_f1 = prf(*task_token)
    role_p, role_r, role_f1 = prf(*role_exact)
    return {
        'val_loss': total_loss / max(len(loader), 1),
        'val_exact_precision': exact_p,
        'val_exact_recall': exact_r,
        'val_exact_f1': exact_f1,
        'val_token_precision': token_p,
        'val_token_recall': token_r,
        'val_token_f1': token_f1,
        'val_role_precision': role_p,
        'val_role_recall': role_r,
        'val_role_f1': role_f1,
    }


def lr_factor(step, total_steps, warmup):
    progress = step / max(total_steps, 1)
    if progress < warmup:
        return progress / max(warmup, 1e-9)
    return max((1.0 - progress) / max(1.0 - warmup, 1e-9), 0.0)


def build_optimizer(model, args):
    groups = []
    for is_head in (False, True):
        for decay in (True, False):
            parameters = []
            for name, parameter in model.named_parameters():
                head_parameter = name.startswith('role_')
                no_decay = ('bias' in name or 'LayerNorm.weight' in name
                            or 'LayerNorm.bias' in name or name == 'role_scale')
                if head_parameter == is_head and (not no_decay) == decay:
                    parameters.append(parameter)
            if parameters:
                groups.append({
                    'params': parameters,
                    'lr': args.head_lr if is_head else args.encoder_lr,
                    'initial_lr': args.head_lr if is_head else args.encoder_lr,
                    'weight_decay': args.weight_decay if decay else 0.0,
                })
    return torch.optim.AdamW(groups)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    train_dataset = FrameGuidedDataset(
        args.train_file, args.frame_info_file, tokenizer, max_length=args.max_length
    )
    dev_dataset = FrameGuidedDataset(
        args.dev_file, args.frame_info_file, tokenizer, max_length=args.max_length,
        extra_fes_by_frame=train_dataset.augmented_fes_by_frame,
    )
    collator = partial(
        frame_guided_collate, pad_token_id=tokenizer.pad_token_id, device=device
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collator, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=collator, drop_last=False,
    )

    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = 1
    config.task2_role_query_dim = args.task2_role_query_dim
    config.task2_role_scale_init = args.task2_role_scale_init
    model = FrameGuidedGlobalPointer(config)
    state = torch.load(args.warm_start_checkpoint, map_location='cpu')
    message = model.load_state_dict(state, strict=False)
    del state
    expected_prefixes = (
        'role_scale', 'role_start_query.', 'role_end_query.',
        'role_start_token.', 'role_end_token.',
    )
    unexpected_missing = [
        name for name in message.missing_keys if not name.startswith(expected_prefixes)
    ]
    if message.unexpected_keys or unexpected_missing:
        raise RuntimeError(
            f'warm-start mismatch: missing={message.missing_keys} '
            f'unexpected={message.unexpected_keys}'
        )
    print(f'[warm start] missing={len(message.missing_keys)} unexpected=0 '
          f'checkpoint={args.warm_start_checkpoint}')
    print(f'[G1 config] device={device} train={len(train_dataset)} dev={len(dev_dataset)} '
          f'batch={args.batch_size} epochs={args.num_train_epochs} FGM={not args.disable_fgm} '
          f'encoder_lr={args.encoder_lr} head_lr={args.head_lr}')
    model = model.to(device)

    optimizer = build_optimizer(model, args)
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulate_gradients)
    total_steps = updates_per_epoch * args.num_train_epochs
    fgm = FGM(model)
    global_step = 0
    best_token_f1 = 0.0
    best_exact_f1 = 0.0
    best_role_f1 = 0.0
    logger = ExperimentLogger(args, 'task2', [
        'epoch', 'train_loss', 'val_loss',
        'val_exact_precision', 'val_exact_recall', 'val_exact_f1',
        'val_token_precision', 'val_token_recall', 'val_token_f1',
        'val_role_precision', 'val_role_recall', 'val_role_f1',
        'best_token_f1', 'best_exact_f1', 'best_role_f1',
        'epoch_seconds', 'samples_per_second', 'encoder_lr', 'head_lr',
        'peak_gpu_memory_mb', 'role_scale',
    ])
    optimizer.zero_grad()
    for epoch in range(1, args.num_train_epochs + 1):
        epoch_start = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f'G1-epoch_{epoch}')):
            output = model(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                target=batch['target'], sentence_end=batch['sentence_end'],
                role_spans=batch['role_spans'], role_mask=batch['role_mask'],
                labels=batch['labels'], device=device,
            )
            loss = output['loss'] / args.accumulate_gradients
            total_loss += float(output['loss'].item())
            loss.backward()
            if not args.disable_fgm:
                fgm.attack(args.fgm_epsilon)
                adversarial_loss = model(
                    input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    target=batch['target'], sentence_end=batch['sentence_end'],
                    role_spans=batch['role_spans'], role_mask=batch['role_mask'],
                    labels=batch['labels'], device=device,
                )['loss'] / args.accumulate_gradients
                adversarial_loss.backward()
                fgm.restore()

            should_step = ((step + 1) % args.accumulate_gradients == 0
                           or step + 1 == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                global_step += 1
                factor = lr_factor(global_step, total_steps, args.warmup_proportion)
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * factor
                optimizer.step()
                optimizer.zero_grad()
            if args.log_interval and (step + 1) % args.log_interval == 0:
                print(f'[G1 train] epoch={epoch} step={step + 1} '
                      f'loss={total_loss / (step + 1):.6f}')

        metrics = evaluate(model, dev_loader, device)
        token_improved = metrics['val_token_f1'] > best_token_f1
        exact_improved = metrics['val_exact_f1'] > best_exact_f1
        role_improved = metrics['val_role_f1'] > best_role_f1
        best_token_f1 = max(best_token_f1, metrics['val_token_f1'])
        best_exact_f1 = max(best_exact_f1, metrics['val_exact_f1'])
        best_role_f1 = max(best_role_f1, metrics['val_role_f1'])
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            **metrics,
            'epoch': epoch,
            'train_loss': total_loss / max(len(train_loader), 1),
            'best_token_f1': best_token_f1,
            'best_exact_f1': best_exact_f1,
            'best_role_f1': best_role_f1,
            'epoch_seconds': epoch_seconds,
            'samples_per_second': len(train_dataset) / max(epoch_seconds, 1e-9),
            'encoder_lr': next(
                group['lr'] for group in optimizer.param_groups
                if group['initial_lr'] == args.encoder_lr
            ),
            'head_lr': next(
                group['lr'] for group in optimizer.param_groups
                if group['initial_lr'] == args.head_lr
            ),
            'peak_gpu_memory_mb': (
                torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == 'cuda' else 0.0
            ),
            'role_scale': float(model.role_scale.detach().cpu()),
        }
        logger.log(row)
        if token_improved:
            save_checkpoint(model.state_dict(), logger.checkpoint_path('task2'))
        if exact_improved:
            save_checkpoint(model.state_dict(), logger.checkpoint_dir / 'task2_best_exact.bin')
        if role_improved:
            save_checkpoint(model.state_dict(), logger.checkpoint_dir / 'task23_best_role.bin')
        print(
            f'[G1 eval] epoch={epoch} token={metrics["val_token_f1"]:.6f} '
            f'exact={metrics["val_exact_f1"]:.6f} role={metrics["val_role_f1"]:.6f} '
            f'role_scale={row["role_scale"]:.6f} seconds={epoch_seconds:.1f}'
        )


if __name__ == '__main__':
    main()
