"""Robust Task3 dataset used by frame-masked and efficient-span experiments."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .frame_utils import load_frame_ontology


def shifted_character_position(pos: int, target_start: int, target_end: int) -> int:
    """Map a raw character offset after [CLS] and two target markers."""
    mapped = int(pos) + 1
    if pos >= target_start:
        mapped += 1
    if pos > target_end:
        mapped += 1
    return mapped


def load_json(path):
    with open(path, 'r', encoding='utf8') as file_handle:
        return json.load(file_handle)


def build_augmented_frame_roles(frame_info_file, train_file):
    """Return ontology labels and frame candidates augmented by observed gold.

    The competition data contains two Train annotations whose FE is absent
    from that frame's frame_info inventory.  Masked CE would otherwise assign
    the gold class negative infinity, so observed Train licenses are added and
    explicitly counted.
    """
    frame_info, idx2label, label2idx, frame_to_roles, frame_to_role_ids = (
        load_frame_ontology(frame_info_file)
    )
    augmented = {frame: list(role_ids) for frame, role_ids in frame_to_role_ids.items()}
    added = []
    for row in load_json(train_file):
        frame = row['frame']
        if frame not in augmented:
            raise KeyError(f'Train frame is missing from frame_info: {frame}')
        current = set(augmented[frame])
        for span in row.get('cfn_spans', []):
            role = span['fe_name']
            if role not in label2idx:
                raise KeyError(f'Train FE is missing globally from frame_info: {role}')
            role_id = label2idx[role]
            if role_id not in current:
                augmented[frame].append(role_id)
                current.add(role_id)
                added.append({
                    'sentence_id': int(row['sentence_id']),
                    'frame': frame,
                    'role': role,
                })
    return frame_info, idx2label, label2idx, augmented, added


class Task3RoleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file,
        tokenizer,
        label2idx,
        train_role_frequency=None,
        train_frame_frequency=None,
        max_examples=None,
        max_length=512,
    ):
        self.source_file = str(json_file)
        self.raw_data = load_json(json_file)
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.train_role_frequency = train_role_frequency or Counter()
        self.train_frame_frequency = train_frame_frequency or Counter()
        self.max_length = max_length
        self.target_start_id = tokenizer.convert_tokens_to_ids('[unused1]')
        self.target_end_id = tokenizer.convert_tokens_to_ids('[unused2]')
        if self.target_start_id == tokenizer.unk_token_id or self.target_end_id == tokenizer.unk_token_id:
            raise ValueError('Task3 target markers [unused1]/[unused2] are missing from the vocabulary.')

        self.records = []
        self.target_overlap_count = 0
        for instance_index, row in enumerate(self.raw_data):
            target = row['target'][-1]
            target_start, target_end = int(target['start']), int(target['end'])
            for span_index, span in enumerate(row.get('cfn_spans', [])):
                start, end = int(span['start']), int(span['end'])
                if not (end < target_start or start > target_end):
                    self.target_overlap_count += 1
                role = span['fe_name']
                self.records.append({
                    'instance_index': instance_index,
                    'span_index': span_index,
                    'sentence_id': int(row['sentence_id']),
                    'text': row['text'],
                    'frame': row['frame'],
                    'target_start': target_start,
                    'target_end': target_end,
                    'argument_start': start,
                    'argument_end': end,
                    'label': int(label2idx[role]),
                    'role': role,
                    'train_frequency': int(self.train_role_frequency.get(role, 0)),
                    'frame_train_frequency': int(
                        self.train_frame_frequency.get(row['frame'], 0)
                    ),
                })
        if max_examples is not None:
            self.records = self.records[:max_examples]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        encoded = self.tokenizer.encode_plus(list(row['text']))
        input_ids = encoded.data['input_ids']
        attention_mask = encoded.data['attention_mask']
        raw_target_start, raw_target_end = row['target_start'], row['target_end']
        target_start, target_end = raw_target_start + 1, raw_target_end + 1
        input_ids = (
            input_ids[:target_start]
            + [self.target_start_id]
            + input_ids[target_start:target_end + 1]
            + [self.target_end_id]
            + input_ids[target_end + 1:]
        )
        attention_mask = (
            attention_mask[:target_start]
            + [1]
            + attention_mask[target_start:target_end + 1]
            + [1]
            + attention_mask[target_end + 1:]
        )
        if len(input_ids) > self.max_length:
            raise ValueError(
                f'Task3 sequence exceeds max_length={self.max_length}: '
                f'sid={row["sentence_id"]} length={len(input_ids)}'
            )
        target = [target_start + 1, target_end + 1]
        argument = [
            shifted_character_position(row['argument_start'], raw_target_start, raw_target_end),
            shifted_character_position(row['argument_end'], raw_target_start, raw_target_end),
        ]
        if not (0 <= argument[0] <= argument[1] < len(input_ids)):
            raise ValueError(
                f'Invalid mapped Task3 span sid={row["sentence_id"]}: '
                f'argument={argument}, sequence_length={len(input_ids)}'
            )
        return input_ids, attention_mask, target, argument, row


def make_role_collate(
    frame_to_role_ids, num_labels, pad_token_id, device, frame_to_id=None,
    frame_sample_weights=None, role_sample_weights=None,
    instance_frame_probabilities=None,
):
    def collate(batch):
        max_len = max(len(item[0]) for item in batch)
        pad_ids = lambda values: values + [pad_token_id] * (max_len - len(values))
        pad_mask = lambda values: values + [0] * (max_len - len(values))
        rows = [item[4] for item in batch]
        candidate_mask = torch.zeros(
            (len(batch), num_labels), dtype=torch.bool, device=device
        )
        for index, row in enumerate(rows):
            role_ids = frame_to_role_ids.get(row['frame'])
            if not role_ids:
                raise KeyError(f'Frame has no candidate roles: {row["frame"]}')
            candidate_mask[index, torch.tensor(role_ids, dtype=torch.long, device=device)] = True
        labels = torch.tensor([row['label'] for row in rows], dtype=torch.long, device=device)
        if not candidate_mask.gather(1, labels[:, None]).all():
            bad = [rows[i] for i in range(len(rows)) if not candidate_mask[i, labels[i]]]
            raise ValueError(f'Gold Task3 roles are absent from frame candidates: {bad[:2]}')
        output = {
            'input_ids': torch.tensor(
                [pad_ids(item[0]) for item in batch], dtype=torch.long, device=device
            ),
            'attention_mask': torch.tensor(
                [pad_mask(item[1]) for item in batch], dtype=torch.long, device=device
            ),
            'target': [item[2] for item in batch],
            'argument': [item[3] for item in batch],
            'labels': labels,
            'candidate_mask': candidate_mask,
            'rows': rows,
        }
        if frame_to_id is not None:
            try:
                output['frame_ids'] = torch.tensor(
                    [frame_to_id[row['frame']] for row in rows],
                    dtype=torch.long,
                    device=device,
                )
            except KeyError as error:
                raise KeyError(f'Frame is absent from definition prototype order: {error}')
        if frame_sample_weights is not None:
            output['prototype_sample_weights'] = torch.tensor(
                [float(frame_sample_weights[row['frame']]) for row in rows],
                dtype=torch.float32,
                device=device,
            )
        if role_sample_weights is not None:
            output['role_sample_weights'] = torch.tensor(
                [float(role_sample_weights[row['role']]) for row in rows],
                dtype=torch.float32,
                device=device,
            )
        if instance_frame_probabilities is not None:
            output['frame_probabilities'] = torch.tensor(
                np.stack([
                    instance_frame_probabilities[row['instance_index']]
                    for row in rows
                ]),
                dtype=torch.float32,
                device=device,
            )
        return output
    return collate


def train_role_frequencies(train_file):
    return Counter(
        span['fe_name']
        for row in load_json(train_file)
        for span in row.get('cfn_spans', [])
    )


def train_frame_frequencies(train_file):
    """Count predicate instances per frame in Train (not argument spans)."""
    return Counter(row['frame'] for row in load_json(train_file))
