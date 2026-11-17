#!/usr/bin/env python3
"""Validate task1 target-marker insertion before starting an expensive run."""
import argparse
import json
from pathlib import Path

from transformers import BertTokenizer

from tasks.task1.dataset import Dataset, TARGET_END_TOKEN, TARGET_START_TOKEN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab_file', default='./chinese_bert_wwm_ext/vocab.txt')
    p.add_argument('--label_file', default='./dataset/frame_info.json')
    p.add_argument('--max_samples', default=0, type=int,
                   help='0 checks all rows; otherwise check the first N rows per split')
    p.add_argument('--show_samples', default=3, type=int)
    return p.parse_args()


def unpack(item, for_test):
    if for_test:
        input_ids, attention_mask, target, sentence_id = item
        return input_ids, attention_mask, target, None, sentence_id
    return item


def check_split(path, tokenizer, label_file, for_test, max_samples, show_samples):
    plain = Dataset(path, label_file, tokenizer, for_test=for_test, target_markers=False)
    marked = Dataset(path, label_file, tokenizer, for_test=for_test, target_markers=True)
    raw = json.loads(Path(path).read_text(encoding='utf8'))
    n = len(plain) if max_samples <= 0 else min(len(plain), max_samples)
    errors = []
    alignment_warnings = []

    for i in range(n):
        p_ids, p_mask, p_target, p_label, p_sid = unpack(plain[i], for_test)
        m_ids, m_mask, m_target, m_label, m_sid = unpack(marked[i], for_test)
        ps, pe = p_target
        ms, me = m_target

        def require(condition, message):
            if not condition:
                errors.append(f'{path} row={i} sentence_id={p_sid}: {message}')

        require(p_sid == m_sid, 'sentence_id changed')
        require(p_label == m_label, 'label changed')
        require(len(m_ids) == len(p_ids) + 2, 'marked input length is not plain length + 2')
        require(len(m_mask) == len(p_mask) + 2, 'marked mask length is not plain length + 2')
        require((ms, me) == (ps + 1, pe + 1),
                f'target shift incorrect: plain={p_target}, marked={m_target}')
        require(0 <= ms <= me < len(m_ids), f'marked target out of range: {m_target}')

        if 0 <= ms - 1 < len(m_ids):
            require(m_ids[ms - 1] == marked.target_start_token_id,
                    f'missing start marker at {ms - 1}')
        if 0 <= me + 1 < len(m_ids):
            require(m_ids[me + 1] == marked.target_end_token_id,
                    f'missing end marker at {me + 1}')

        require(p_ids[ps:pe + 1] == m_ids[ms:me + 1],
                'real target token ids changed after marker insertion')
        reconstructed = m_ids[:ms - 1] + m_ids[ms:me + 1] + m_ids[me + 2:]
        require(reconstructed == p_ids, 'removing markers does not reconstruct original input_ids')
        require(all(x == 1 for x in m_mask), 'unpadded dataset attention mask contains non-one value')

        # The original baseline assumes one encoded token per input character.
        expected_len = len(raw[i]['text']) + 2  # [CLS] and [SEP]
        if len(p_ids) != expected_len:
            alignment_warnings.append(
                f'{path} row={i} sentence_id={p_sid}: token_len={len(p_ids)}, expected={expected_len}'
            )

        if i < show_samples:
            target_chars = raw[i]['text'][raw[i]['target'][-1]['start']:raw[i]['target'][-1]['end'] + 1]
            window_start = max(0, ms - 3)
            window_end = min(len(m_ids), me + 4)
            window_tokens = tokenizer.convert_ids_to_tokens(m_ids[window_start:window_end])
            target_tokens = tokenizer.convert_ids_to_tokens(m_ids[ms:me + 1])
            print({
                'split': path,
                'row': i,
                'sentence_id': p_sid,
                'target_chars': target_chars,
                'plain_target': p_target,
                'marked_target': m_target,
                'target_tokens': target_tokens,
                'marker_window_tokens': window_tokens,
            })

    print(f'[CHECK] {path}: checked={n}, errors={len(errors)}, alignment_warnings={len(alignment_warnings)}')
    if errors:
        print('\n'.join(errors[:20]))
        raise AssertionError(f'{path}: marker validation failed with {len(errors)} errors')
    if alignment_warnings:
        print('[WARNING] Character/token alignment examples:')
        print('\n'.join(alignment_warnings[:10]))
        raise AssertionError(
            f'{path}: original character-to-token assumption failed for {len(alignment_warnings)} rows'
        )


def main():
    args = parse_args()
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    print({
        'start_token': TARGET_START_TOKEN,
        'start_id': tokenizer.convert_tokens_to_ids(TARGET_START_TOKEN),
        'end_token': TARGET_END_TOKEN,
        'end_id': tokenizer.convert_tokens_to_ids(TARGET_END_TOKEN),
    })
    splits = [
        ('./dataset/cfn-train.json', False),
        ('./dataset/cfn-dev.json', False),
        ('./dataset/cfn-test-A.json', True),
        ('./dataset/cfn-test-B.json', True),
    ]
    for path, for_test in splits:
        check_split(path, tokenizer, args.label_file, for_test,
                    args.max_samples, args.show_samples)
    print('[OK] All task1 target-marker checks passed.')


if __name__ == '__main__':
    main()

