import codecs
import json

import torch
from transformers import BertTokenizer


TARGET_START_TOKEN = "[unused1]"
TARGET_END_TOKEN = "[unused2]"


class Dataset(torch.utils.data.Dataset):
    """Task1 dataset with optional explicit target boundary markers.

    Without markers this preserves the original baseline behavior exactly.
    With markers, the encoded sequence becomes::

        [CLS] left [unused1] target [unused2] right [SEP]

    ``target`` continues to point to the real target tokens, not the marker
    tokens, so the existing task1 model can be reused without modification.
    """

    def __init__(self, json_file, label_file, tokenizer, for_test=False,
                 target_markers=False):
        self.for_test = for_test
        self.target_markers = target_markers
        self.tokenizer = tokenizer
        with codecs.open(json_file, 'r', encoding='utf8') as f:
            self.all_data = json.load(f)
        with codecs.open(label_file, 'r', encoding='utf8') as f:
            self.ori_labels = json.load(f)

        self.idx2label = []
        self.label2idx = {}
        for i, line in enumerate(self.ori_labels):
            self.idx2label.append(line["frame_name"])
            self.label2idx[line["frame_name"]] = i
        self.num_labels = len(self.idx2label)

        self.target_start_token_id = self.tokenizer.convert_tokens_to_ids(TARGET_START_TOKEN)
        self.target_end_token_id = self.tokenizer.convert_tokens_to_ids(TARGET_END_TOKEN)
        if self.target_markers:
            unk_id = self.tokenizer.unk_token_id
            if self.target_start_token_id == unk_id or self.target_end_token_id == unk_id:
                raise ValueError(
                    f"Target marker token missing from vocab: "
                    f"{TARGET_START_TOKEN}={self.target_start_token_id}, "
                    f"{TARGET_END_TOKEN}={self.target_end_token_id}"
                )
            if self.target_start_token_id == self.target_end_token_id:
                raise ValueError("Target start/end marker token ids must be different.")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        row = self.all_data[item]
        encoded = self.tokenizer.encode_plus(list(row['text']))
        input_ids = encoded.data['input_ids']
        attention_mask = encoded.data['attention_mask']

        # Character offsets become token offsets after the leading [CLS].
        target_start = row["target"][-1]["start"] + 1
        target_end = row["target"][-1]["end"] + 1
        target = [target_start, target_end]

        if self.target_markers:
            input_ids = (
                input_ids[:target_start]
                + [self.target_start_token_id]
                + input_ids[target_start:target_end + 1]
                + [self.target_end_token_id]
                + input_ids[target_end + 1:]
            )
            attention_mask = (
                attention_mask[:target_start]
                + [1]
                + attention_mask[target_start:target_end + 1]
                + [1]
                + attention_mask[target_end + 1:]
            )
            # Both real target boundaries shift right by the start marker.
            target = [target_start + 1, target_end + 1]

        sentence_id = row["sentence_id"]
        if self.for_test:
            return input_ids, attention_mask, target, sentence_id

        label = self.label2idx[row["frame"]]
        return input_ids, attention_mask, target, label, sentence_id


if __name__ == '__main__':
    tokenizer = BertTokenizer(
        vocab_file='./chinese_bert_wwm_ext/vocab.txt',
        do_lower_case=True)
    dataset = Dataset(
        "./dataset/cfn-train.json",
        "./dataset/frame_info.json",
        tokenizer=tokenizer,
        target_markers=True,
    )
    dataset[0]

