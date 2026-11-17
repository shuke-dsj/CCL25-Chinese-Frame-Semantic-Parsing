#!/usr/bin/python3

import os

import torch
import codecs
import json
from functools import partial
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from tasks.common.params import args
from .model_factory import build_task2_model
from .decoding import build_safe_span_mask, decode_to_character_spans
from tasks.common.experiment_utils import REPO_ROOT, resolve_checkpoint


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, tokenizer, for_test=False):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        self.label2idx = {}
        self.label2cls = {}
        self.num_labels = 0
        for k, line in enumerate(self.ori_labels):
            frame_name = line["frame_name"]
            self.label2cls[frame_name] = k
            if frame_name not in self.label2idx:
                self.label2idx[frame_name] = {}
            for i, fes in enumerate(line["fes"]):
                self.label2idx[frame_name][fes["fe_name"]] = i
            if self.num_labels < len(line["fes"]):
                self.num_labels = len(line["fes"])
        self.max_cls = len(self.label2idx)

        pass

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        d1 = self.all_data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        target = [d1["target"][-1]["start"] + 1, d1["target"][-1]["end"] + 1]
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[
                                                                                                target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]

        return input_ids, attention_mask, target, sentence_id


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
    sentence_id = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        sentence_id.append(d[3])

    input_ids = np.array(input_ids_list, dtype=np.int64)
    attention_mask = np.array(attention_mask_list, dtype=np.int64)

    input_ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.from_numpy(attention_mask).to(device)


    return input_ids, attention_mask, target, sentence_id


def test(model, val_loader):
    model.eval()
    predicts = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, sentence_id = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=None,
                           device=device, for_test=True)
            logits = output["logits"]
            beta = float(args.task2_boundary_residual_beta)
            if beta != 0.0:
                if ('boundary_start_logits' not in output
                        or 'boundary_end_logits' not in output):
                    raise ValueError(
                        '--task2_boundary_residual_beta requires '
                        '--task2_model_variant boundary_auxiliary'
                    )
                logits = logits + beta * (
                    output['boundary_start_logits'].unsqueeze(2)
                    + output['boundary_end_logits'].unsqueeze(1)
                )
            positive = logits >= args.task2_threshold
            if args.task2_safe_decode:
                prediction = positive & build_safe_span_mask(attention_mask, target)
                predicts.extend(decode_to_character_spans(
                    prediction, attention_mask, target, sentence_id
                ))
            else:
                legacy_mask = torch.triu(
                    attention_mask.unsqueeze(2).bool() & attention_mask.unsqueeze(1).bool(), diagonal=0
                )
                prediction = positive & legacy_mask
                for batch_index, start, end in torch.nonzero(prediction, as_tuple=False).tolist():
                    if end < target[batch_index][0]:
                        predicts.append([sentence_id[batch_index], start - 1, end - 1])
                    elif start > target[batch_index][1]:
                        predicts.append([sentence_id[batch_index], start - 3, end - 3])
    print(f'[task2 decode] threshold={args.task2_threshold} safe={args.task2_safe_decode} '
          f'boundary_residual_beta={args.task2_boundary_residual_beta} '
          f'predicted_spans={len(predicts)}')
    # Keep one JSON tuple per line for official submission readability.
    data_json = '[\n' + ',\n'.join('   ' + json.dumps(p, ensure_ascii=False) for p in predicts) + '\n]'
    os.makedirs(args.prediction_dir, exist_ok=True)
    with open(os.path.join(args.prediction_dir, 'B_task2_test.json'), 'w', encoding='utf8', newline='\n') as f:
        f.write(data_json)
    with open(os.path.join(args.prediction_dir, 'A_task2_test.json'), 'w', encoding='utf8', newline='\n') as f:
        f.write(data_json)


def main():
    global device
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.prediction_dir is None:
        args.prediction_dir = str(REPO_ROOT / 'tasks' / 'task2' / 'predictions' / 'default')

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    test_dataset = Dataset(args.test_file,
                            "./dataset/frame_info.json",
                            tokenizer)

    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = 1
    config.max_cls = test_dataset.max_cls
    config.task2_content_dim = args.task2_content_dim
    config.task2_content_scale_init = args.task2_content_scale_init
    config.task2_boundary_loss_weight = args.task2_boundary_loss_weight
    model = build_task2_model(config, args.task2_model_variant)
    # load_pretrained_bert(model, args.init_checkpoint)
    state = torch.load(resolve_checkpoint(args, 'model_task2'), map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
    print(f'[权重加载检查] missing_keys={len(msg.missing_keys)}, unexpected_keys={len(msg.unexpected_keys)}; '
          f'variant={args.task2_model_variant}; checkpoint: {resolve_checkpoint(args, "model_task2")}')
    if msg.missing_keys:
        print('[警告] 以下参数没有加载到（模型将用随机值！）:', msg.missing_keys[:10])
    # model.load_state_dict(torch.load('', map_location='cpu'))
    model = model.to(device)


    test_loader = DataLoader(
        batch_size=args.batch_size,
        dataset=test_dataset,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(get_model_input, device=device),
        drop_last=False
    )
    test(model, test_loader)



if __name__ == "__main__":
    main()

