#!/usr/bin/python3

import os

import torch
import codecs
import json
from functools import partial
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from tasks.common.params import args
from .model import Model
from .frame_utils import apply_frame_mask, load_frame_ontology, load_sentence_frame_predictions
from tasks.common.experiment_utils import REPO_ROOT, resolve_checkpoint


class Dataset(torch.utils.data.Dataset):

    def __init__(self, json_file, label_file, task2_file, tokenizer,
                 task1_file=None, for_test=False):
        aeda_chars = [".", ";", "?", ":", "!", ",", "，", "。"]
        self.for_test = for_test
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)
        with codecs.open(task2_file, 'r', encoding='utf8') as f:
            self.task2_data = json.load(f)
        self.task1_frames = (
            load_sentence_frame_predictions(task1_file) if task1_file else {}
        )
        self.idx2label = []
        for line in self.ori_labels:
            for fes in line["fes"]:
                if fes["fe_name"] not in self.idx2label:
                    self.idx2label.append(fes["fe_name"])
        self.label2idx = {}
        for i in range(len(self.idx2label)):
            self.label2idx[self.idx2label[i]] = i
        self.data_dict = {}
        for line in self.all_data:
            text = line["text"]
            target = [line["target"][-1]["start"] + 1, line["target"][-1]["end"] + 1]
            self.data_dict[line["sentence_id"]] = {"text": text, "target": target}
        self.data = []
        self.skipped_target_overlaps = 0
        for line in self.task2_data:
            text = self.data_dict[line[0]]["text"]
            target = self.data_dict[line[0]]["target"]
            if line[2] + 1 < target[0]:
                label_idx = [line[1] + 1, line[2] + 1]
            elif line[1] + 1 > target[1]:
                label_idx = [line[1] + 3, line[2] + 3]
            else:
                self.skipped_target_overlaps += 1
                continue
            self.data.append({
                'text': text,
                "label_idx": label_idx,
                "sentence_id": line[0],
                "target": target,
                "ori_target": [line[1], line[2]],
                "frame": self.task1_frames.get(int(line[0]), "")
            })
        self.num_labels = len(self.idx2label)
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d1 = self.data[item]
        data = self.tokenizer.encode_plus(list(d1['text']))
        input_ids = data.data['input_ids']
        attention_mask = data.data['attention_mask']
        label_idx = d1["label_idx"]
        target = d1["target"]
        input_ids = input_ids[0: target[0]] + [1] + input_ids[target[0]: target[1] + 1] + [2] + input_ids[
                                                                                                target[1] + 1:]
        attention_mask = attention_mask + [1, 1]
        sentence_id = d1["sentence_id"]
        ori_target = d1["ori_target"]

        return input_ids, attention_mask, label_idx, sentence_id, ori_target, d1["frame"]


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
    ori_target = []
    frames = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        sentence_id.append(d[3])
        ori_target.append(d[4])
        frames.append(d[5])

    input_ids = np.array(input_ids_list, dtype=np.int64)
    attention_mask = np.array(attention_mask_list, dtype=np.int64)

    input_ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.from_numpy(attention_mask).to(device)

    return input_ids, attention_mask, target, sentence_id, ori_target, frames


def test(model, val_loader):
    model.eval()
    idx2label = val_loader.dataset.idx2label
    predicts = []
    frame_to_role_ids = None
    if args.task3_frame_constrained:
        _, ontology_labels, _, _, frame_to_role_ids = load_frame_ontology(
            str(REPO_ROOT / 'dataset' / 'frame_info.json')
        )
        if ontology_labels != idx2label:
            raise RuntimeError('Task3 label order differs from frame_info ontology order.')
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='eval'):
            input_ids, attention_mask, target, sentence_id, ori_target, frames = batch

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=None,
                           device=device, for_test=True)
            logits = output["logits"]
            if args.task3_frame_constrained:
                missing = [sentence_id[i] for i, frame in enumerate(frames) if not frame]
                if missing:
                    raise ValueError(
                        f'Task1 frame prediction missing for {len(missing)} Task3 examples; '
                        f'first sentence IDs: {missing[:5]}'
                    )
                logits, _ = apply_frame_mask(logits, frames, frame_to_role_ids, 'error')
            pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            for i in range(len(pred)):
                predicts.append([sentence_id[i], ori_target[i][0], ori_target[i][1], idx2label[pred[i]]])

            pass
    # 严格按官方格式：每个条目 [id, 起点, 终点, "角色名"] 占一行（原 indent=1 会把条目拆成多行）
    data_json = '[\n' + ',\n'.join('   ' + json.dumps(p, ensure_ascii=False) for p in predicts) + '\n]'
    os.makedirs(args.prediction_dir, exist_ok=True)
    # A榜和B榜的文件名不同：A榜要 A_task3_test.json，B榜要 B_task3_test.json（见官方数据包里的样例）
    for name in ('A_task3_test.json', 'B_task3_test.json'):
        with open(os.path.join(args.prediction_dir, name), 'w', encoding='utf8', newline='\n') as f:
            f.write(data_json)


def main():
    global device
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.prediction_dir is None:
        args.prediction_dir = str(REPO_ROOT / 'tasks' / 'task3' / 'predictions' / 'default')

    task1_file = args.task1_prediction_file
    if args.task3_frame_constrained and task1_file is None:
        task1_file = str(REPO_ROOT / 'tasks' / 'task1' / 'predictions' / 'B_task1_test.json')
    task2_file = args.task2_prediction_file or os.path.join(args.prediction_dir, "B_task2_test.json")

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    test_dataset = Dataset(args.test_file,
                            "./dataset/frame_info.json",
                            task2_file,
                            tokenizer,
                            task1_file=task1_file)
    if test_dataset.skipped_target_overlaps:
        print(f'[Task3 decode] skipped {test_dataset.skipped_target_overlaps} Task2 spans overlapping target')

    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = test_dataset.num_labels
    model = Model(config)
    # load_pretrained_bert(model, args.init_checkpoint)
    state = torch.load(resolve_checkpoint(args, 'model_task3'), map_location='cpu')
    msg = model.load_state_dict(state, strict=False)
    print(f'[权重加载检查] missing_keys={len(msg.missing_keys)}, unexpected_keys={len(msg.unexpected_keys)}; '
          f'checkpoint: {resolve_checkpoint(args, "model_task3")}')
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

