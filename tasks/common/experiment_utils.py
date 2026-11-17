"""Shared experiment logging and checkpoint helpers.

The original baseline only printed metrics to stdout.  This module keeps the
training loop changes small while making every experiment reproducible and
comparable: one result directory contains config, a CSV, a text log, and
checkpoints.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def task_key(task_name: str) -> str:
    normalized = task_name.replace('model_', '')
    if normalized.startswith('task1'):
        return 'task1'
    if normalized.startswith('task2'):
        return 'task2'
    if normalized.startswith('task3'):
        return 'task3'
    raise ValueError(f'Cannot infer task directory from task_name={task_name!r}')


def task_result_dir(args, task_name: str) -> Path:
    requested = getattr(args, 'result_dir', None)
    if requested:
        path = Path(requested)
        return path if path.is_absolute() else REPO_ROOT / path
    experiment = getattr(args, 'experiment_name', 'baseline')
    return REPO_ROOT / 'tasks' / task_key(task_name) / 'experiment_record' / f'{experiment}-result'


class ExperimentLogger:
    """Append epoch-level metrics to ``<experiment>-result``.

    ``fieldnames`` is fixed per task so the CSV remains easy to load on a
    machine without pandas.  Each row is flushed immediately because a long
    GPU run should survive an interrupted process.
    """

    def __init__(self, args, task_name: str, fieldnames: Iterable[str]):
        self.result_dir = task_result_dir(args, task_name)
        self.task_name = task_name
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.result_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.result_dir / f"{task_name}_metrics.csv"
        self.log_path = self.result_dir / f"{task_name}.log"
        self.config_path = self.result_dir / f"{task_name}_run_config.json"
        self.fieldnames = list(fieldnames)
        self._write_config(args)
        self._write_header()

    def _write_config(self, args) -> None:
        if self.config_path.exists():
            return
        config = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))}
        config["command"] = " ".join(sys.argv)
        config["created_at"] = datetime.now().isoformat(timespec="seconds")
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log(self, metrics: Mapping[str, object]) -> None:
        row = {name: metrics.get(name, "") for name in self.fieldnames}
        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = " ".join(f"{k}={v}" for k, v in row.items() if v != "")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {summary}\n")

    def checkpoint_path(self, task_name: Optional[str] = None) -> Path:
        return self.checkpoint_dir / f"{task_name or self.task_name}_best.bin"


def resolve_checkpoint(args, task_name: str) -> Path:
    """Resolve checkpoints inside the owning task directory."""
    explicit = getattr(args, "checkpoint_path", None)
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else REPO_ROOT / path
    base = task_name.replace('model_', '')
    result_dir = task_result_dir(args, task_name)
    task_dir = REPO_ROOT / 'tasks' / task_key(task_name)
    candidates = [
        result_dir / 'checkpoints' / f'{base}_best.bin',
        result_dir / 'checkpoints' / f'model_{base}_best.bin',
        task_dir / 'experiment_record' / 'legacy-saves' / 'checkpoints' / f'model_{base}_best.bin',
        task_dir / 'experiment_record' / 'legacy-saves' / 'checkpoints' / f'{base}_best.bin',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"找不到 {base} 的 checkpoint，已尝试：\n  " + "\n  ".join(str(c) for c in candidates)
        + "\n可用 --checkpoint_path 直接指定路径。")


def binary_prf(correct: float, predicted: float, gold: float) -> Dict[str, float]:
    precision = correct / max(predicted, 1e-9)
    recall = correct / max(gold, 1e-9)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def set_seed(seed: int) -> None:
    """Set Python/NumPy/PyTorch seeds for comparable experiments."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def save_checkpoint(state, path) -> Path:
    """Atomically save a checkpoint through a Python-opened file object.

    Older PyTorch Windows builds may fail inside ``PyTorchFileWriter`` when a
    long/non-ASCII path is passed directly, incorrectly reporting that the
    parent directory does not exist. Python's file APIs handle the same path
    correctly, so we open the temporary file ourselves and pass the binary
    file object to ``torch.save``.
    """
    import os
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    last_error = None
    for _ in range(2):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                temporary.unlink()
            with temporary.open('wb') as file_handle:
                torch.save(state, file_handle)
                file_handle.flush()
                os.fsync(file_handle.fileno())
            os.replace(str(temporary), str(path))
            return path
        except (OSError, RuntimeError) as error:
            last_error = error
            path.parent.mkdir(parents=True, exist_ok=True)
    raise RuntimeError(f'Failed to save checkpoint to {path}: {last_error}')

