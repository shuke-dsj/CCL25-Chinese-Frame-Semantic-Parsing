"""Grouped positive/hard-negative data for D3 definition querying."""

from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
import torch

from tasks.task3.verifier.dataset import (
    MARKERS,
    encode_with_target_argument_markers,
)


def read_json(path):
    with open(path, 'r', encoding='utf8') as handle:
        return json.load(handle)


def build_definition_maps(frame_info, label2idx):
    pair_text = {}
    global_texts = {}
    frame_role_ids = {}
    for frame in frame_info:
        frame_name = frame['frame_name']
        frame_prefix = (
            f"框架：{frame_name}（{frame.get('frame_ename', '')}）。"
            f"{frame.get('frame_def', '')}"
        )
        role_ids = []
        for fe in frame.get('fes', []):
            role = fe['fe_name']
            role_id = int(label2idx[role])
            definition = (
                f"{frame_prefix} 框架元素：{role}（{fe.get('fe_ename', '')}）。"
                f"{fe.get('fe_def', '')}"
            )
            pair_text[(frame_name, role_id)] = definition
            global_texts.setdefault(role_id, []).append(
                f"框架元素：{role}（{fe.get('fe_ename', '')}）。{fe.get('fe_def', '')}"
            )
            role_ids.append(role_id)
        frame_role_ids[frame_name] = role_ids
    global_fallback = {
        role_id: texts[0] for role_id, texts in global_texts.items()
    }
    return pair_text, global_fallback, frame_role_ids


class DefinitionQueryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_file,
        frame_info,
        tokenizer,
        label2idx,
        role_names,
        negative_count=4,
        hard_negative_pool=12,
        prototype_file=None,
        seed=42,
        max_length=512,
        all_legal_candidates=False,
        train_role_frequency=None,
        train_frame_frequency=None,
        max_examples=None,
    ):
        self.rows = read_json(source_file)
        self.frame_info = frame_info
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.role_names = role_names
        self.negative_count = int(negative_count)
        self.hard_negative_pool = int(hard_negative_pool)
        self.seed = int(seed)
        self.max_length = int(max_length)
        self.all_legal_candidates = bool(all_legal_candidates)
        self.pair_text, self.global_text, self.frame_role_ids = build_definition_maps(
            frame_info, label2idx
        )
        self.records = []
        split_role_frequency = Counter(
            span['fe_name']
            for row in self.rows
            for span in row.get('cfn_spans', [])
        )
        split_frame_frequency = Counter(row['frame'] for row in self.rows)
        role_frequency = train_role_frequency or split_role_frequency
        frame_frequency = train_frame_frequency or split_frame_frequency
        for instance_index, row in enumerate(self.rows):
            target = row['target'][-1]
            for span in row.get('cfn_spans', []):
                role = span['fe_name']
                self.records.append({
                    'instance_index': int(instance_index),
                    'sentence_id': int(row['sentence_id']),
                    'text': row['text'],
                    'frame': row['frame'],
                    'target_start': int(target['start']),
                    'target_end': int(target['end']),
                    'argument_start': int(span['start']),
                    'argument_end': int(span['end']),
                    'role_id': int(label2idx[role]),
                    'role': role,
                    'train_role_frequency': int(role_frequency.get(role, 0)),
                    'train_frame_frequency': int(frame_frequency.get(row['frame'], 0)),
                })
        if max_examples is not None:
            self.records = self.records[: int(max_examples)]

        self.prototype_similarity = None
        if prototype_file:
            bank = np.load(prototype_file, allow_pickle=False)
            prototypes = np.asarray(bank['global_role_prototypes'], dtype=np.float32)
            prototypes /= np.maximum(
                np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8
            )
            self.prototype_similarity = prototypes @ prototypes.T

    def __len__(self):
        return len(self.records)

    def _definition(self, frame, role_id):
        value = self.pair_text.get((frame, role_id))
        if value is not None:
            return value
        # Handles the two observed Train licenses absent from frame_info.
        return (
            f"框架：{frame}。"
            + self.global_text.get(
                role_id, f"框架元素：{self.role_names[role_id]}。"
            )
        )

    def _candidates(self, record, index):
        gold = record['role_id']
        legal = list(dict.fromkeys(self.frame_role_ids[record['frame']]))
        if gold not in legal:
            legal.append(gold)
        negatives = [role_id for role_id in legal if role_id != gold]
        if self.all_legal_candidates:
            return [gold] + negatives
        rng = random.Random(self.seed * 1000003 + index)
        if self.prototype_similarity is not None:
            negatives.sort(
                key=lambda role_id: float(self.prototype_similarity[gold, role_id]),
                reverse=True,
            )
            pool = negatives[: max(self.hard_negative_pool, self.negative_count)]
            # Keep the hardest half deterministic and sample the rest for
            # epoch-independent reproducibility with useful diversity.
            hard_n = min((self.negative_count + 1) // 2, len(pool))
            chosen = pool[:hard_n]
            remainder = [role for role in pool[hard_n:] if role not in chosen]
            rng.shuffle(remainder)
            chosen.extend(remainder[: self.negative_count - len(chosen)])
        else:
            rng.shuffle(negatives)
            chosen = negatives[: self.negative_count]
        if len(chosen) < self.negative_count:
            remainder = [role for role in negatives if role not in chosen]
            rng.shuffle(remainder)
            chosen.extend(remainder[: self.negative_count - len(chosen)])
        return [gold] + chosen

    def _encode_pair(self, record, role_id):
        sentence_ids, _, positions = encode_with_target_argument_markers(
            self.tokenizer,
            record['text'],
            record['target_start'], record['target_end'],
            record['argument_start'], record['argument_end'],
            max_length=self.max_length,
        )
        # sentence_ids already contains [CLS] ... [SEP].
        definition = self._definition(record['frame'], role_id)
        definition_ids = self.tokenizer.encode(
            definition, add_special_tokens=False
        )
        available = self.max_length - len(sentence_ids) - 1
        if available <= 0:
            raise ValueError(
                f'No room for definition sid={record["sentence_id"]}; '
                f'sentence_length={len(sentence_ids)} max_length={self.max_length}'
            )
        definition_ids = definition_ids[:available]
        input_ids = sentence_ids + definition_ids + [self.tokenizer.sep_token_id]
        token_type_ids = [0] * len(sentence_ids) + [1] * (len(definition_ids) + 1)
        return {
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'token_type_ids': token_type_ids,
            'marker_positions': positions,
            'role_id': int(role_id),
        }

    def __getitem__(self, index):
        record = self.records[index]
        candidates = self._candidates(record, index)
        return {
            'pairs': [self._encode_pair(record, role_id) for role_id in candidates],
            'candidate_role_ids': candidates,
            'gold_role_id': int(record['role_id']),
            'record': record,
        }


def collate_groups(items, pad_token_id=0):
    pairs = [pair for item in items for pair in item['pairs']]
    max_len = max(len(pair['input_ids']) for pair in pairs)
    pad = lambda values, value: values + [value] * (max_len - len(values))
    marker_positions = {
        name: torch.tensor(
            [pair['marker_positions'][name] for pair in pairs], dtype=torch.long
        )
        for name in MARKERS
    }
    return {
        'input_ids': torch.tensor([
            pad(pair['input_ids'], pad_token_id) for pair in pairs
        ], dtype=torch.long),
        'attention_mask': torch.tensor([
            pad(pair['attention_mask'], 0) for pair in pairs
        ], dtype=torch.long),
        'token_type_ids': torch.tensor([
            pad(pair['token_type_ids'], 0) for pair in pairs
        ], dtype=torch.long),
        'marker_positions': marker_positions,
        'group_sizes': [len(item['pairs']) for item in items],
        'candidate_role_ids': [item['candidate_role_ids'] for item in items],
        'gold_role_ids': [item['gold_role_id'] for item in items],
        'records': [item['record'] for item in items],
    }
