"""Frame/FE-definition guided dataset for Task2/Task3 joint span extraction."""
import json

import torch


class FrameGuidedDataset(torch.utils.data.Dataset):
    """Build ``[CLS] marked sentence [SEP] frame definition + FE slots [SEP]``.

    The original Task2 dataset remains untouched. Sentence character positions
    use exactly the same marker/coordinate convention as ``tasks.task2.dataset``.
    Every FE name is retained as a query slot; an overlong frame definition is
    truncated before any FE slot is dropped.
    """

    def __init__(
        self, json_file, frame_info_file, tokenizer, max_length=512,
        extra_fes_by_frame=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        with open(json_file, encoding='utf8') as handle:
            self.rows = json.load(handle)
        with open(frame_info_file, encoding='utf8') as handle:
            self.frame_info = json.load(handle)
        self.frame_map = {item['frame_name']: item for item in self.frame_info}

        # Prefer the stable FE abbreviation when the annotation and ontology
        # use different Chinese aliases (e.g. 图形/fig vs 焦点图形/fig).  Only a
        # truly unknown FE, with neither a matching name nor abbreviation, is
        # appended as an observed extra query slot.
        self.fe_aliases_by_frame = {}
        self.augmented_fes_by_frame = {}
        for frame_name, extra_names in (extra_fes_by_frame or {}).items():
            ontology_names = {
                item['fe_name'] for item in self.frame_map[frame_name]['fes']
            }
            extras = [name for name in extra_names if name not in ontology_names]
            if extras:
                self.augmented_fes_by_frame[frame_name] = list(dict.fromkeys(extras))

        alias_affected_examples = 0
        for row in self.rows:
            frame_name = row['frame']
            ontology_fes = self.frame_map[frame_name]['fes']
            ontology_names = {item['fe_name'] for item in ontology_fes}
            row_has_alias = False
            for span in row.get('cfn_spans', []):
                fe_name = span['fe_name']
                if fe_name in ontology_names:
                    continue
                abbreviation = span.get('fe_abbr')
                abbreviation_matches = [
                    item['fe_name'] for item in ontology_fes
                    if abbreviation and item.get('fe_abbr') == abbreviation
                ]
                if len(abbreviation_matches) == 1:
                    aliases = self.fe_aliases_by_frame.setdefault(frame_name, {})
                    previous = aliases.setdefault(fe_name, abbreviation_matches[0])
                    if previous != abbreviation_matches[0]:
                        raise ValueError(
                            f'ambiguous FE alias {frame_name}/{fe_name}: '
                            f'{previous} vs {abbreviation_matches[0]}'
                        )
                    row_has_alias = True
                else:
                    extras = self.augmented_fes_by_frame.setdefault(frame_name, [])
                    if fe_name not in extras:
                        extras.append(fe_name)
            alias_affected_examples += int(row_has_alias)

        if self.fe_aliases_by_frame:
            alias_pairs = sum(map(len, self.fe_aliases_by_frame.values()))
            print(
                '[frame ontology alias] '
                f'pairs={alias_pairs} affected_examples={alias_affected_examples} '
                f'details={self.fe_aliases_by_frame}'
            )

        if self.augmented_fes_by_frame:
            augmented_pairs = sum(map(len, self.augmented_fes_by_frame.values()))
            affected_examples = sum(
                any(
                    span['fe_name'] in self.augmented_fes_by_frame.get(row['frame'], ())
                    for span in row.get('cfn_spans', [])
                )
                for row in self.rows
            )
            print(
                '[frame ontology augmentation] '
                f'pairs={augmented_pairs} affected_examples={affected_examples} '
                f'details={self.augmented_fes_by_frame}'
            )

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        self.target_start_id = tokenizer.convert_tokens_to_ids('[unused1]')
        self.target_end_id = tokenizer.convert_tokens_to_ids('[unused2]')
        if self.target_start_id == tokenizer.unk_token_id or self.target_end_id == tokenizer.unk_token_id:
            raise ValueError('BERT vocabulary does not contain [unused1]/[unused2] target markers')

    def __len__(self):
        return len(self.rows)

    def encode_chars(self, text):
        return self.tokenizer.convert_tokens_to_ids(list(text))

    def __getitem__(self, index):
        row = self.rows[index]
        frame = self.frame_map[row['frame']]
        target_raw = row['target'][-1]
        target = [int(target_raw['start']) + 1, int(target_raw['end']) + 1]

        text_ids = self.encode_chars(row['text'])
        sentence_ids = (
            [self.cls_id]
            + text_ids[:target_raw['start']]
            + [self.target_start_id]
            + text_ids[target_raw['start']:target_raw['end'] + 1]
            + [self.target_end_id]
            + text_ids[target_raw['end'] + 1:]
            + [self.sep_id]
        )
        sentence_end = len(sentence_ids) - 1  # index of the first [SEP]

        # A small number of frame_info entries repeat the same FE name. Gold
        # annotations expose only ``fe_name`` and cannot distinguish those
        # duplicate ontology rows, so keep one query slot per observable name.
        role_names = list(dict.fromkeys(item['fe_name'] for item in frame['fes']))
        role_names.extend(self.augmented_fes_by_frame.get(row['frame'], ()))
        role_ids = []
        for role_index, role_name in enumerate(role_names):
            if role_index:
                role_ids.extend(self.encode_chars('、'))
            role_ids.extend(self.encode_chars(role_name))

        prefix_text = frame['frame_name'] + '：' + frame.get('frame_def', '') + '；框架元素：'
        prefix_ids = self.encode_chars(prefix_text)
        # Reserve one final [SEP] and every FE name. Definition prose is the
        # only truncation target, because dropping an FE would invalidate labels.
        prefix_budget = self.max_length - len(sentence_ids) - len(role_ids) - 1
        if prefix_budget < 0:
            raise ValueError(
                f'sentence + FE slots exceed max_length at row {index}: '
                f'{len(sentence_ids)} + {len(role_ids)} + 1 > {self.max_length}'
            )
        prefix_ids = prefix_ids[:prefix_budget]

        definition_ids = list(prefix_ids)
        role_spans = []
        for role_index, role_name in enumerate(role_names):
            if role_index:
                definition_ids.extend(self.encode_chars('、'))
            start = len(sentence_ids) + len(definition_ids)
            name_ids = self.encode_chars(role_name)
            definition_ids.extend(name_ids)
            role_spans.append([start, start + len(name_ids) - 1])

        input_ids = sentence_ids + definition_ids + [self.sep_id]
        if len(input_ids) > self.max_length:
            raise RuntimeError(f'constructed sequence exceeds max_length: {len(input_ids)}')
        attention_mask = [1] * len(input_ids)

        role_to_index = {name: item for item, name in enumerate(role_names)}
        labels = []
        for span in row['cfn_spans']:
            role_name = self.fe_aliases_by_frame.get(row['frame'], {}).get(
                span['fe_name'], span['fe_name']
            )
            if role_name not in role_to_index:
                # This should now indicate a programming/data-schema problem,
                # not a harmless frame_info/annotation mismatch.
                raise ValueError(
                    f"FE {span['fe_name']} is not licensed by frame {row['frame']}"
                )
            if span['end'] < target_raw['start']:
                start, end = span['start'] + 1, span['end'] + 1
            elif span['start'] > target_raw['end']:
                start, end = span['start'] + 3, span['end'] + 3
            else:
                continue
            labels.append([role_to_index[role_name], start, end])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': target,
            'sentence_end': sentence_end,
            'role_spans': role_spans,
            'role_names': role_names,
            'labels': labels,
            'sentence_id': row['sentence_id'],
            'frame': row['frame'],
        }


def frame_guided_collate(batch, pad_token_id=0, device=None):
    batch_size = len(batch)
    max_total_length = max(len(item['input_ids']) for item in batch)
    max_sentence_length = max(item['sentence_end'] + 1 for item in batch)
    max_roles = max(len(item['role_spans']) for item in batch)

    input_ids = torch.full(
        (batch_size, max_total_length), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        batch_size, max_total_length, dtype=torch.long, device=device
    )
    role_spans = torch.zeros(
        batch_size, max_roles, 2, dtype=torch.long, device=device
    )
    role_mask = torch.zeros(batch_size, max_roles, dtype=torch.bool, device=device)
    labels = torch.zeros(
        batch_size, max_roles, max_sentence_length, max_sentence_length,
        dtype=torch.float, device=device,
    )

    for batch_index, item in enumerate(batch):
        length = len(item['input_ids'])
        input_ids[batch_index, :length] = torch.tensor(
            item['input_ids'], dtype=torch.long, device=device
        )
        attention_mask[batch_index, :length] = 1
        count = len(item['role_spans'])
        role_spans[batch_index, :count] = torch.tensor(
            item['role_spans'], dtype=torch.long, device=device
        )
        role_mask[batch_index, :count] = True
        for role_index, start, end in item['labels']:
            labels[batch_index, role_index, start, end] = 1.0

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'target': [item['target'] for item in batch],
        'sentence_end': torch.tensor(
            [item['sentence_end'] for item in batch], dtype=torch.long, device=device
        ),
        'role_spans': role_spans,
        'role_mask': role_mask,
        'labels': labels,
        'sentence_ids': [item['sentence_id'] for item in batch],
        'frames': [item['frame'] for item in batch],
        'role_names': [item['role_names'] for item in batch],
    }
