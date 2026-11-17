"""Top-k cross-Frame candidate groups for Task3 D4."""

from __future__ import annotations

import json

import numpy as np
import torch

from tasks.task3.definition_query.dataset import build_definition_maps
from tasks.task3.verifier.dataset import MARKERS, encode_with_target_argument_markers


def read_json(path):
    with open(path, 'r', encoding='utf8') as handle:
        return json.load(handle)


class CandidateAwareDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_file,
        cache_file,
        frame_info,
        frame_to_role_ids,
        tokenizer,
        role2idx,
        role_names,
        pair_frames_per_role=2,
        max_length=512,
        train_mode=False,
        max_groups=0,
        generator_perturb_probability=0.0,
        generator_perturb_min_margin=0.1,
        generator_perturb_max_margin=0.6,
        perturb_seed=42,
    ):
        self.rows = read_json(source_file)
        bank = np.load(cache_file, allow_pickle=False)
        required = {
            'candidate_role_ids', 'candidate_generator_scores', 'frame_probs',
            'instance_indices', 'sentence_ids', 'starts', 'ends', 'role_names',
            'frame_names', 'gold_candidate_mask', 'boundary_match',
            'gold_tuples', 'gold_boundaries',
        }
        missing = required.difference(bank.files)
        if missing:
            raise ValueError(f'D4 cache missing fields: {sorted(missing)}')
        if bank['role_names'].tolist() != list(role_names):
            raise ValueError('D4 cache role order differs from frame ontology.')
        self.candidate_role_ids = bank['candidate_role_ids'].astype(np.int64)
        self.generator_scores = bank['candidate_generator_scores'].astype(np.float32)
        self.frame_probs = bank['frame_probs'].astype(np.float32)
        self.instance_indices = bank['instance_indices'].astype(np.int64)
        self.sentence_ids = bank['sentence_ids'].astype(np.int64)
        self.starts = bank['starts'].astype(np.int64)
        self.ends = bank['ends'].astype(np.int64)
        self.gold_mask = bank['gold_candidate_mask'].astype(bool)
        self.boundary_match = bank['boundary_match'].astype(bool)
        self.gold_tuples = int(bank['gold_tuples'][0])
        self.gold_boundaries = int(bank['gold_boundaries'][0])
        self.frame_names = bank['frame_names'].tolist()
        self.frame2idx = {name: index for index, name in enumerate(self.frame_names)}
        self.tokenizer = tokenizer
        self.role_names = list(role_names)
        self.pair_frames_per_role = int(pair_frames_per_role)
        self.max_length = int(max_length)
        self.train_mode = bool(train_mode)
        self.generator_perturb_probability = float(generator_perturb_probability)
        self.generator_perturb_min_margin = float(generator_perturb_min_margin)
        self.generator_perturb_max_margin = float(generator_perturb_max_margin)
        self.perturb_seed = int(perturb_seed)
        self.epoch = 0
        if not 0.0 <= self.generator_perturb_probability <= 1.0:
            raise ValueError('generator_perturb_probability must be in [0,1].')
        if not 0.0 <= self.generator_perturb_min_margin <= self.generator_perturb_max_margin:
            raise ValueError('Invalid D4 generator perturbation margin range.')
        if self.pair_frames_per_role < 1:
            raise ValueError('pair_frames_per_role must be positive.')
        n = len(self.candidate_role_ids)
        for name, value in (
            ('generator_scores', self.generator_scores),
            ('frame_probs', self.frame_probs),
            ('instance_indices', self.instance_indices),
            ('starts', self.starts), ('ends', self.ends), ('gold_mask', self.gold_mask),
        ):
            if len(value) != n:
                raise ValueError(f'D4 cache row mismatch for {name}: {len(value)} != {n}')
        self.indices = np.arange(n, dtype=np.int64)
        if self.train_mode:
            self.indices = self.indices[
                self.boundary_match & self.gold_mask.any(axis=1)
            ]
        if max_groups:
            self.indices = self.indices[: int(max_groups)]

        pair_text, global_text, _ = build_definition_maps(frame_info, role2idx)
        self.pair_text = pair_text
        self.global_text = global_text
        role_to_frames = [[] for _ in role_names]
        for frame, role_ids in frame_to_role_ids.items():
            frame_id = self.frame2idx[frame]
            for role_id in role_ids:
                role_to_frames[int(role_id)].append(frame_id)
        self.role_to_frames = [
            np.asarray(sorted(set(values)), dtype=np.int64)
            for values in role_to_frames
        ]

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        """Change deterministic synthetic hard negatives between epochs."""
        self.epoch = int(epoch)

    def _definition(self, frame_id, role_id):
        frame = self.frame_names[int(frame_id)]
        value = self.pair_text.get((frame, int(role_id)))
        if value is not None:
            return value
        return (
            f'框架：{frame}。'
            + self.global_text.get(
                int(role_id), f'框架元素：{self.role_names[int(role_id)]}。'
            )
        )

    def _best_frames(self, cache_index, role_id):
        legal = self.role_to_frames[int(role_id)]
        probabilities = self.frame_probs[cache_index]
        if not len(legal):
            legal = np.asarray([int(probabilities.argmax())], dtype=np.int64)
        order = legal[np.argsort(probabilities[legal])[::-1]]
        chosen = order[: self.pair_frames_per_role].tolist()
        while len(chosen) < self.pair_frames_per_role:
            chosen.append(chosen[-1])
        weights = probabilities[np.asarray(chosen, dtype=np.int64)].astype(np.float32)
        weights = np.maximum(weights, 1e-12)
        weights /= weights.sum()
        return chosen, np.log(weights).astype(np.float32)

    def _encode(self, source, start, end, frame_id, role_id):
        target = source['target'][-1]
        sentence_ids, _, positions = encode_with_target_argument_markers(
            self.tokenizer,
            source['text'],
            int(target['start']), int(target['end']),
            int(start), int(end),
            max_length=self.max_length,
        )
        definition_ids = self.tokenizer.encode(
            self._definition(frame_id, role_id), add_special_tokens=False,
        )
        available = self.max_length - len(sentence_ids) - 1
        if available <= 0:
            raise ValueError(
                f'No room for D4 definition sid={source["sentence_id"]}.'
            )
        definition_ids = definition_ids[:available]
        input_ids = sentence_ids + definition_ids + [self.tokenizer.sep_token_id]
        return {
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'token_type_ids': (
                [0] * len(sentence_ids) + [1] * (len(definition_ids) + 1)
            ),
            'marker_positions': positions,
        }

    def __getitem__(self, item_index):
        cache_index = int(self.indices[item_index])
        instance_index = int(self.instance_indices[cache_index])
        source = self.rows[instance_index]
        if int(source['sentence_id']) != int(self.sentence_ids[cache_index]):
            raise ValueError(f'D4 source/cache SID mismatch at {cache_index}.')
        candidate_ids = self.candidate_role_ids[cache_index]
        pairs = []
        pair_frame_ids = []
        frame_log_weights = []
        for role_id in candidate_ids:
            frames, log_weights = self._best_frames(cache_index, int(role_id))
            pair_frame_ids.append(frames)
            frame_log_weights.append(log_weights.tolist())
            for frame_id in frames:
                pairs.append(self._encode(
                    source, self.starts[cache_index], self.ends[cache_index],
                    frame_id, int(role_id),
                ))
        generator_scores = self.generator_scores[cache_index].copy()
        perturbed = False
        perturbed_candidate_index = -1
        if self.train_mode and self.generator_perturb_probability > 0:
            rng = np.random.RandomState(
                self.perturb_seed * 1000003 + self.epoch * 9176 + cache_index
            )
            gold_indices = np.flatnonzero(self.gold_mask[cache_index])
            original_top = int(generator_scores.argmax())
            if (
                len(gold_indices) == 1
                and original_top == int(gold_indices[0])
                and rng.rand() < self.generator_perturb_probability
            ):
                gold_index = int(gold_indices[0])
                gold_frame = int(pair_frame_ids[gold_index][0])
                negatives = [
                    index for index in range(len(candidate_ids))
                    if index != gold_index
                ]
                cross_frame = [
                    index for index in negatives
                    if int(pair_frame_ids[index][0]) != gold_frame
                ]
                pool = cross_frame or negatives
                perturbed_candidate_index = max(
                    pool, key=lambda index: float(generator_scores[index])
                )
                margin = rng.uniform(
                    self.generator_perturb_min_margin,
                    self.generator_perturb_max_margin,
                )
                generator_scores[perturbed_candidate_index] = (
                    generator_scores[gold_index] + float(margin)
                )
                perturbed = True
        frame_probability = self.frame_probs[cache_index]
        entropy = -float(np.sum(
            frame_probability * np.log(np.maximum(frame_probability, 1e-12))
        )) / max(np.log(len(frame_probability)), 1e-12)
        sorted_generator = np.sort(generator_scores)
        margin = float(sorted_generator[-1] - sorted_generator[-2])
        return {
            'pairs': pairs,
            'candidate_role_ids': candidate_ids.tolist(),
            'generator_scores': generator_scores.tolist(),
            'frame_log_weights': frame_log_weights,
            'pair_frame_ids': pair_frame_ids,
            'group_features': [margin, entropy],
            'gold_mask': self.gold_mask[cache_index].tolist(),
            'boundary_match': bool(self.boundary_match[cache_index]),
            'generator_perturbed': perturbed,
            'perturbed_candidate_index': int(perturbed_candidate_index),
            'cache_index': cache_index,
            'instance_index': instance_index,
            'sentence_id': int(self.sentence_ids[cache_index]),
            'start': int(self.starts[cache_index]),
            'end': int(self.ends[cache_index]),
        }


def collate_candidate_groups(items, pad_token_id=0):
    pairs = [pair for item in items for pair in item['pairs']]
    max_length = max(len(pair['input_ids']) for pair in pairs)
    pad = lambda values, value: values + [value] * (max_length - len(values))
    marker_positions = {
        name: torch.tensor([
            pair['marker_positions'][name] for pair in pairs
        ], dtype=torch.long)
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
        'generator_scores': torch.tensor([
            item['generator_scores'] for item in items
        ], dtype=torch.float32),
        'frame_log_weights': torch.tensor([
            item['frame_log_weights'] for item in items
        ], dtype=torch.float32),
        'group_features': torch.tensor([
            item['group_features'] for item in items
        ], dtype=torch.float32),
        'gold_mask': torch.tensor([
            item['gold_mask'] for item in items
        ], dtype=torch.bool),
        'candidate_role_ids': torch.tensor([
            item['candidate_role_ids'] for item in items
        ], dtype=torch.long),
        'boundary_match': torch.tensor([
            item['boundary_match'] for item in items
        ], dtype=torch.bool),
        'generator_perturbed': torch.tensor([
            item['generator_perturbed'] for item in items
        ], dtype=torch.bool),
        'rows': items,
    }
