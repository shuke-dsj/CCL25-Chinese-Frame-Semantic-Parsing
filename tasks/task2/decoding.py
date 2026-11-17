"""Shared Task2 span masking and coordinate conversion."""
import torch


def build_safe_span_mask(attention_mask, target):
    batch, length = attention_mask.shape
    output = torch.zeros(batch, length, length, dtype=torch.bool, device=attention_mask.device)
    for i, (target_start, target_end) in enumerate(target):
        seq_len = int(attention_mask[i].sum().item())
        if target_start > 1:
            n = target_start - 1
            output[i, 1:target_start, 1:target_start] = torch.triu(
                torch.ones(n, n, dtype=torch.bool, device=output.device)
            )
        right_start = target_end + 3
        right_end = seq_len - 1
        if right_start < right_end:
            n = right_end - right_start
            output[i, right_start:right_end, right_start:right_end] = torch.triu(
                torch.ones(n, n, dtype=torch.bool, device=output.device)
            )
    return output


def decode_to_character_spans(prediction, attention_mask, target, sentence_ids):
    output = []
    for i in range(prediction.shape[0]):
        seq_len = int(attention_mask[i].sum().item())
        target_start, target_end = target[i]
        for start, end in torch.nonzero(prediction[i], as_tuple=False).tolist():
            if start >= 1 and end < target_start:
                output.append([sentence_ids[i], start - 1, end - 1])
            elif start >= target_end + 3 and end < seq_len - 1:
                output.append([sentence_ids[i], start - 3, end - 3])
    return output

