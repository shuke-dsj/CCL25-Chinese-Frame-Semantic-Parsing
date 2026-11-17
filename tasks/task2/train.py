#!/usr/bin/python3

import os
import sys
import time
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from .dataset import Dataset
from tasks.common.params import args
from .model_factory import build_task2_model
from .decoding import build_safe_span_mask
from tasks.common.experiment_utils import ExperimentLogger, save_checkpoint, set_seed


TASK2_DEFAULT_EPOCHS = 5


def apply_task2_defaults():
    """Apply Task2-only defaults without overriding explicit CLI arguments."""
    epoch_was_explicit = any(
        value == '--num_train_epochs' or value.startswith('--num_train_epochs=')
        for value in sys.argv[1:]
    )
    if not epoch_was_explicit:
        args.num_train_epochs = TASK2_DEFAULT_EPOCHS


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)


def get_model_input(data, device=None):
    """

    :param data: input_ids1, input_ids2, label_starts, label_ends, true_label
    :return:
    """

    def pad(d, max_len, v=0):
        return d + [v] * (max_len - len(d))

    bs = len(data)
    max_len = max([len(x[0]) for x in data])

    input_ids_list = []
    attention_mask_list = []
    target = []
    target_cls = []
    labels = []
    sentence_id = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        labels.append(d[3])
        sentence_id.append(d[4])
        target_cls.append(d[5])

    input_ids = np.array(input_ids_list, dtype=np.int64)
    attention_mask = np.array(attention_mask_list, dtype=np.int64)
    target_cls = np.array(target_cls, dtype=np.int64)

    input_ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.from_numpy(attention_mask).to(device)
    target_cls = torch.from_numpy(target_cls).to(device)

    H_label = torch.zeros(bs, max_len, max_len).to(device)
    for i in range(bs):
        for idx in labels[i]:
            H_label[i][idx[0], idx[1]] = 1

    return input_ids, attention_mask, target, H_label, sentence_id, target_cls


def _token_coverage_counts(prediction, gold):
    """Micro token-union counts in inserted-token coordinates, per example."""
    intersection = predicted_tokens = gold_tokens = 0
    for i in range(prediction.shape[0]):
        length = prediction.shape[1]
        pred_coverage = torch.zeros(length, dtype=torch.bool, device=prediction.device)
        gold_coverage = torch.zeros(length, dtype=torch.bool, device=prediction.device)
        for start, end in torch.nonzero(prediction[i], as_tuple=False).tolist():
            pred_coverage[start:end + 1] = True
        for start, end in torch.nonzero(gold[i], as_tuple=False).tolist():
            gold_coverage[start:end + 1] = True
        intersection += int((pred_coverage & gold_coverage).sum().item())
        predicted_tokens += int(pred_coverage.sum().item())
        gold_tokens += int(gold_coverage.sum().item())
    return intersection, predicted_tokens, gold_tokens


def eval(model, val_loader):
    model.eval()
    exact_tp = exact_predicted = exact_gold = 0
    token_tp = token_predicted = token_gold = 0
    total_loss = 0.0
    total_span_loss = 0.0
    total_boundary_loss = 0.0
    boundary_batches = 0
    start_tp = start_predicted = start_gold = 0
    end_tp = end_predicted = end_gold = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, label, sentence_id, target_cls = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target,
                           target_cls=target_cls, labels=label, device=device, for_test=False)
            total_loss += output['loss'].item()
            total_span_loss += output['span_loss'].item()
            if output.get('boundary_loss') is not None:
                total_boundary_loss += output['boundary_loss'].item()
                boundary_batches += 1
                start_prediction = (
                    output['boundary_start_logits'] >= 0
                ) & output['boundary_start_valid']
                end_prediction = (
                    output['boundary_end_logits'] >= 0
                ) & output['boundary_end_valid']
                start_labels = output['boundary_start_labels']
                end_labels = output['boundary_end_labels']
                start_tp += int((start_prediction & start_labels).sum().item())
                start_predicted += int(start_prediction.sum().item())
                start_gold += int(start_labels.sum().item())
                end_tp += int((end_prediction & end_labels).sum().item())
                end_predicted += int(end_prediction.sum().item())
                end_gold += int(end_labels.sum().item())
            valid = build_safe_span_mask(attention_mask, target)
            prediction = (output['logits'] >= 0) & valid
            gold = label.bool()
            exact_tp += int((prediction & gold).sum().item())
            exact_predicted += int(prediction.sum().item())
            exact_gold += int(gold.sum().item())
            batch_counts = _token_coverage_counts(prediction, gold)
            token_tp += batch_counts[0]
            token_predicted += batch_counts[1]
            token_gold += batch_counts[2]

    exact_precision = exact_tp / max(exact_predicted, 1e-9)
    exact_recall = exact_tp / max(exact_gold, 1e-9)
    exact_f1 = 2 * exact_precision * exact_recall / max(exact_precision + exact_recall, 1e-9)
    token_precision = token_tp / max(token_predicted, 1e-9)
    token_recall = token_tp / max(token_gold, 1e-9)
    token_f1 = 2 * token_precision * token_recall / max(token_precision + token_recall, 1e-9)
    start_precision = start_tp / max(start_predicted, 1e-9)
    start_recall = start_tp / max(start_gold, 1e-9)
    start_f1 = 2 * start_precision * start_recall / max(start_precision + start_recall, 1e-9)
    end_precision = end_tp / max(end_predicted, 1e-9)
    end_recall = end_tp / max(end_gold, 1e-9)
    end_f1 = 2 * end_precision * end_recall / max(end_precision + end_recall, 1e-9)
    return {
        'val_loss': total_loss / max(len(val_loader), 1),
        'val_span_loss': total_span_loss / max(len(val_loader), 1),
        'val_boundary_loss': total_boundary_loss / boundary_batches if boundary_batches else '',
        'val_exact_precision': exact_precision,
        'val_exact_recall': exact_recall,
        'val_exact_f1': exact_f1,
        'val_token_precision': token_precision,
        'val_token_recall': token_recall,
        'val_token_f1': token_f1,
        # Registry/backward-compatible primary metric: official-like token F1.
        'val_precision': token_precision,
        'val_recall': token_recall,
        'val_f1': token_f1,
        'val_start_precision': start_precision if boundary_batches else '',
        'val_start_recall': start_recall if boundary_batches else '',
        'val_start_f1': start_f1 if boundary_batches else '',
        'val_end_precision': end_precision if boundary_batches else '',
        'val_end_recall': end_recall if boundary_batches else '',
        'val_end_f1': end_f1 if boundary_batches else '',
    }


def train(model, train_loader, val_loader):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'content_scale', 'frame_scale']
    auxiliary_lr = args.task2_aux_head_lr if args.task2_aux_head_lr is not None else args.lr
    optimizer_grouped_parameters = []
    for is_auxiliary in (False, True):
        for use_decay in (True, False):
            parameters = [
                p for n, p in param_optimizer
                if n.startswith('boundary_') == is_auxiliary
                and (not any(nd in n for nd in no_decay)) == use_decay
            ]
            if parameters:
                initial_lr = auxiliary_lr if is_auxiliary else args.lr
                optimizer_grouped_parameters.append({
                    'params': parameters,
                    'weight_decay': 0.01 if use_decay else 0.0,
                    'lr': initial_lr,
                    'initial_lr': initial_lr,
                    'is_auxiliary': is_auxiliary,
                })

    if not args.task2_disable_parameter_noise:
        noise_lambda = 0.15
        for name, para in param_optimizer:
            # Scalar residual scales must remain at their explicit initialization.
            if para.numel() <= 1:
                continue
            model.state_dict()[name][:] += (
                (torch.rand(para.size()).to(device) - 0.5) * noise_lambda * torch.std(para)
            )

    total_steps = int(len(train_loader) * args.num_train_epochs / args.accumulate_gradients)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    global_step = 0
    best_token_f1 = 0.0
    best_exact_f1 = 0.0
    fgm = FGM(model)
    logger = ExperimentLogger(
        args, 'task2',
        ['epoch', 'train_loss', 'train_span_loss', 'train_boundary_loss',
         'val_loss', 'val_span_loss', 'val_boundary_loss',
         'val_exact_precision', 'val_exact_recall', 'val_exact_f1',
         'val_token_precision', 'val_token_recall', 'val_token_f1',
         'val_precision', 'val_recall', 'val_f1',
         'val_start_precision', 'val_start_recall', 'val_start_f1',
         'val_end_precision', 'val_end_recall', 'val_end_f1',
         'best_token_f1', 'best_exact_f1', 'epoch_seconds', 'samples_per_second',
         'learning_rate', 'aux_head_lr', 'peak_gpu_memory_mb',
         'content_scale', 'frame_scale', 'boundary_loss_weight']
    )
    for i_epoch in range(1, 1 + args.num_train_epochs):
        epoch_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        total_loss = 0.0
        total_span_loss = 0.0
        total_boundary_loss = 0.0
        boundary_batches = 0
        iter_bar = tqdm(train_loader, total=len(train_loader), desc=f'epoch_{i_epoch} ')
        model.train()
        for step, batch in enumerate(iter_bar):
            input_ids, attention_mask, target, label, sentence_id, target_cls = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target,
                           target_cls=target_cls, labels=label, device=device)
            loss = output['loss']
            total_loss += loss.item()
            total_span_loss += output['span_loss'].item()
            if output.get('boundary_loss') is not None:
                total_boundary_loss += output['boundary_loss'].item()
                boundary_batches += 1
            if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                print(f'loss: {total_loss / (step + 1)}')
            loss.backward()

            if not args.task2_disable_fgm:
                fgm.attack()
                adversarial_loss = model(
                    input_ids=input_ids, attention_mask=attention_mask, target=target,
                    target_cls=target_cls, labels=label, device=device
                )['loss']
                adversarial_loss.backward()
                fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if (step + 1) % args.accumulate_gradients == 0:
                lr_factor = warmup_linear(global_step / total_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['initial_lr'] * lr_factor
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        metrics = eval(model, val_loader)
        train_loss = total_loss / max(len(train_loader), 1)
        token_improved = metrics['val_token_f1'] > best_token_f1
        exact_improved = metrics['val_exact_f1'] > best_exact_f1
        if token_improved:
            print(f"new token-best: {metrics['val_token_f1']} (old {best_token_f1})")
            best_token_f1 = metrics['val_token_f1']
        if exact_improved:
            print(f"new exact-best: {metrics['val_exact_f1']} (old {best_exact_f1})")
            best_exact_f1 = metrics['val_exact_f1']
        print(f"token P/R/F1: {metrics['val_token_precision']:.6f}/"
              f"{metrics['val_token_recall']:.6f}/{metrics['val_token_f1']:.6f}; "
              f"exact F1: {metrics['val_exact_f1']:.6f}")
        epoch_seconds = time.perf_counter() - epoch_start
        model_to_save = model.module if hasattr(model, 'module') else model
        metrics.update({
            'epoch': i_epoch, 'train_loss': train_loss,
            'train_span_loss': total_span_loss / max(len(train_loader), 1),
            'train_boundary_loss': total_boundary_loss / boundary_batches if boundary_batches else '',
            'best_token_f1': best_token_f1, 'best_exact_f1': best_exact_f1,
            'epoch_seconds': epoch_seconds,
            'samples_per_second': len(train_loader.dataset) / max(epoch_seconds, 1e-9),
            'learning_rate': next(
                group['lr'] for group in optimizer.param_groups if not group['is_auxiliary']
            ),
            'aux_head_lr': next(
                (group['lr'] for group in optimizer.param_groups if group['is_auxiliary']), ''
            ),
            'peak_gpu_memory_mb': (torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0.0,
            'content_scale': (
                float(model_to_save.content_scale.detach().cpu())
                if hasattr(model_to_save, 'content_scale') else ''
            ),
            'frame_scale': (
                float(model_to_save.frame_scale.detach().cpu())
                if hasattr(model_to_save, 'frame_scale') else ''
            ),
            'boundary_loss_weight': (
                model_to_save.boundary_loss_weight
                if hasattr(model_to_save, 'boundary_loss_weight') else ''
            ),
        })
        # Persist metrics before large checkpoint I/O so an I/O failure never loses the epoch record.
        logger.log(metrics)
        if token_improved:
            save_checkpoint(model_to_save.state_dict(), logger.checkpoint_path('task2'))
        if exact_improved:
            save_checkpoint(model_to_save.state_dict(), logger.checkpoint_dir / 'task2_best_exact.bin')


def load_pretrained_bert(bert_model, init_checkpoint):
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location='cpu')
        if 'model_bert_best' in init_checkpoint:
            bert_model.load_state_dict(state['model_bert'], strict=False)
        else:
            state = {k.replace('bert.', '').replace('roformer.', ''): v for k, v in state.items() if
                     not k.startswith('cls.')}
            # state['embeddings.token_type_embeddings.weight'] = state['embeddings.token_type_embeddings.weight'][:2, :]
            bert_model.load_state_dict(state, strict=False)


def main():
    global device
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    apply_task2_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    train_dataset = Dataset("./dataset/cfn-train.json",
                            "./dataset/frame_info.json",
                            tokenizer)
    dev_dataset = Dataset("./dataset/cfn-dev.json",
                          "./dataset/frame_info.json",
                          tokenizer)

    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = 1
    config.max_cls = train_dataset.max_cls
    config.task2_valid_loss_mask = args.task2_valid_loss_mask
    config.task2_content_dim = args.task2_content_dim
    config.task2_content_scale_init = args.task2_content_scale_init
    config.task2_frame_prototype_file = args.task2_frame_prototype_file
    config.task2_frame_query_dim = args.task2_frame_query_dim
    config.task2_frame_scale_init = args.task2_frame_scale_init
    config.task2_frame_names = list(train_dataset.label2cls.keys())
    config.task2_boundary_loss_weight = args.task2_boundary_loss_weight
    print(f'[task2 config] experiment={args.experiment_name} valid_loss_mask={args.task2_valid_loss_mask} ' 
          f'fgm={not args.task2_disable_fgm} variant={args.task2_model_variant} '
          f'content_dim={args.task2_content_dim} content_scale_init={args.task2_content_scale_init} '
          f'frame_query_dim={args.task2_frame_query_dim} frame_scale_init={args.task2_frame_scale_init} '
          f'boundary_loss_weight={args.task2_boundary_loss_weight} aux_head_lr={args.task2_aux_head_lr} '
          f'epochs={args.num_train_epochs} seed={args.seed}')
    model = build_task2_model(config, args.task2_model_variant)
    # load_pretrained_bert(model, args.init_checkpoint)
    state = torch.load(args.init_checkpoint, map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
    print(f'[weight load] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} '
          f'checkpoint={args.init_checkpoint}')
    # model.load_state_dict(torch.load('', map_location='cpu'))
    model = model.to(device)
    train_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=train_dataset,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(get_model_input, device=device),
        drop_last=True
    )

    val_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=dev_dataset,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(get_model_input, device=device),
        drop_last=False
    )
    train(model, train_loader, val_loader)




if __name__ == "__main__":
    main()

