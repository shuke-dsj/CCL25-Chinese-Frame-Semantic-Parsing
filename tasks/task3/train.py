#!/usr/bin/python3

import os
import time
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from .dataset import Dataset
from tasks.common.params import args
from .model import Model
from tasks.common.experiment_utils import ExperimentLogger, set_seed


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认为2范数
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
    labels = []
    sentence_id = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        labels.append(d[3])
        sentence_id.append(d[4])

    input_ids = np.array(input_ids_list, dtype=np.int64)
    attention_mask = np.array(attention_mask_list, dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)

    input_ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.from_numpy(attention_mask).to(device)
    labels = torch.from_numpy(labels).to(device)

    return input_ids, attention_mask, target, labels, sentence_id


def eval(model, val_loader):
    model.eval()
    correct = 0.0
    total = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, labels, sentence_id = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target,
                           labels=labels, device=device, for_test=False)
            logits = output["logits"]
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == labels).sum().item()
            total += len(pred)
            total_loss += output['loss'].item()

    acc = correct / max(total, 1.0)
    return {
        'val_loss': total_loss / max(len(val_loader), 1),
        'val_acc': acc,
        'val_precision': acc,
        'val_recall': acc,
        'val_f1': acc,
    }


def train(model, train_loader, val_loader):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    noise_lambda = 0.15
    for name, para in param_optimizer:
        model.state_dict()[name][:] += (torch.rand(para.size()).to(device) - 0.5) * noise_lambda * torch.std(para)

    total_steps = int(len(train_loader) * args.num_train_epochs / args.accumulate_gradients)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    global_step = 0
    best_acc = 0.0
    logger = ExperimentLogger(
        args, 'task3',
        ['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 'best_val_f1', 'epoch_seconds', 'samples_per_second', 'learning_rate', 'peak_gpu_memory_mb']
    )
    for i_epoch in range(1, 1 + args.num_train_epochs):
        epoch_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        total_loss = 0.0
        iter_bar = tqdm(train_loader, total=len(train_loader), desc=f'epoch_{i_epoch} ')
        model.train()
        for step, batch in enumerate(iter_bar):
            input_ids, attention_mask, target, labels, sentence_id = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target,
                           labels=labels, device=device)
            loss = output['loss']
            total_loss += loss.item()
            if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                print(f'loss: {total_loss / (step + 1)}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if (step + 1) % args.accumulate_gradients == 0:
                lr_this_step = args.lr * warmup_linear(global_step / total_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        metrics = eval(model, val_loader)
        train_loss = total_loss / max(len(train_loader), 1)
        if metrics['val_acc'] > best_acc:
            print(f"saved! new best acc {metrics['val_acc']}, ori_acc {best_acc}")
            best_acc = metrics['val_acc']
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), logger.checkpoint_path('task3'))
        else:
            print(f"current acc: {metrics['val_acc']}")
        epoch_seconds = time.perf_counter() - epoch_start
        metrics.update({
            'epoch': i_epoch, 'train_loss': train_loss, 'best_val_f1': best_acc,
            'epoch_seconds': epoch_seconds,
            'samples_per_second': len(train_loader.dataset) / max(epoch_seconds, 1e-9),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'peak_gpu_memory_mb': (torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0.0,
        })
        logger.log(metrics)


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
    config.num_labels = train_dataset.num_labels
    model = Model(config)
    # load_pretrained_bert(model, args.init_checkpoint)
    state = torch.load(args.init_checkpoint, map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
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

