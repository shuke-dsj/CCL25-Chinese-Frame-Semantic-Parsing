# !/usr/bin/python3

import os

import codecs
import json
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
from tasks.common.params import args
from .model import Model
from .dataset import Dataset
from tasks.common.experiment_utils import REPO_ROOT, resolve_checkpoint




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


def test(model, test_loader):
    model.eval()
    all_dataset = []
    idx2label = test_loader.dataset.idx2label
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='eval'):
            input_ids, attention_mask, target, sentence_id = batch

            output = model(input_ids=input_ids, attention_mask=attention_mask, target=target, labels=None,
                           device=device, for_test=True)
            logits = output["logits"]
            pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            for i in range(len(sentence_id)):
                all_dataset.append([sentence_id[i], idx2label[pred[i]]])

    # 严格按官方格式：每个条目 [id, "框架名"] 占一行（原 indent=1 会把条目拆成多行）
    data_json = '[\n' + ',\n'.join('   ' + json.dumps(p, ensure_ascii=False) for p in all_dataset) + '\n]'
    os.makedirs(args.prediction_dir, exist_ok=True)
    # A榜和B榜的文件名不同：A榜要 A_task1_test.json，B榜要 B_task1_test.json（见官方数据包里的样例）
    for name in ('A_task1_test.json', 'B_task1_test.json'):
        with open(os.path.join(args.prediction_dir, name), 'w', encoding='utf8', newline='\n') as f:
            f.write(data_json)


def main():
    global device
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.prediction_dir is None:
        args.prediction_dir = str(REPO_ROOT / 'tasks' / 'task1' / 'predictions' / 'default')
    print(f'[task1 predict config] experiment={args.experiment_name} variant={args.model_variant} '
          f'layers={args.target_aware_layers} target_markers={args.task1_target_markers}')

    tokenizer = BertTokenizer(vocab_file=args.vocab_file,
                              do_lower_case=True)

    test_dataset = Dataset(args.test_file,
                           "./dataset/frame_info.json",
                           tokenizer,
                           for_test=True,
                           target_markers=args.task1_target_markers)

    config = BertConfig.from_json_file(args.config_file)
    # BertConfig.from_pretrained('hfl/chinese-bert-wwm-ext')
    config.num_labels = test_dataset.num_labels
    config.model_variant = args.model_variant
    config.target_aware_layers = args.target_aware_layers
    model = Model(config)
    # load_pretrained_bert(model, args.init_checkpoint)
    # state = torch.load(args.init_checkpoint, map_location='cpu')
    state = torch.load(resolve_checkpoint(args, 'model_task1'), map_location='cpu')

    msg = model.load_state_dict(state, strict=False)
    print(f'[权重加载检查] missing_keys={len(msg.missing_keys)}, unexpected_keys={len(msg.unexpected_keys)}; '
          f'checkpoint: {resolve_checkpoint(args, "model_task1")}')
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

