"""Utilities for constructing textual descriptions of task1 frames."""
import json


def build_frame_text(frame, max_fes=16):
    fe_names = []
    for fe in frame.get('fes', []):
        name = fe.get('fe_name', '').strip()
        if name and name not in fe_names:
            fe_names.append(name)
        if len(fe_names) >= max_fes:
            break
    parts = [f"框架名称：{frame['frame_name']}"]
    if fe_names:
        parts.append("框架元素：" + "、".join(fe_names))
    parts.append(f"框架定义：{frame['frame_def']}")
    return "。".join(parts)


def load_frame_texts(frame_info_file, max_fes=16):
    with open(frame_info_file, encoding='utf8') as f:
        frames = json.load(f)
    texts = [build_frame_text(frame, max_fes=max_fes) for frame in frames]
    labels = [frame['frame_name'] for frame in frames]
    return frames, labels, texts

