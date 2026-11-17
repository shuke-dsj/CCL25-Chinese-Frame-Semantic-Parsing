#!/usr/bin/env python3
"""Validate the post-migration task-local repository layout."""
from pathlib import Path
import py_compile

ROOT=Path(__file__).resolve().parents[2]
required=[
 'tasks/common/params.py','tasks/common/experiment_utils.py',
 'tasks/task1/dataset.py','tasks/task1/model.py','tasks/task1/assets/frame_prototypes/name-def-fes-mean.pt',
 'tasks/task2/dataset.py','tasks/task2/model.py','tasks/task2/train.py','tasks/task2/predict.py',
 'tasks/task2/experiment_record/legacy-saves/checkpoints/model_task2_best.bin',
 'tasks/task3/dataset.py','tasks/task3/model.py','tasks/task3/train.py','tasks/task3/predict.py',
 'tasks/task3/experiment_record/legacy-saves/checkpoints/model_task3_best.bin',
]
forbidden=['train_task1.py','train_task2.py','train_task3.py','predict_task1.py','predict_task2.py','predict_task3.py','params.py','experiment_utils.py','frame_prototypes','predictions','saves']
missing=[x for x in required if not (ROOT/x).exists()]
remaining=[x for x in forbidden if (ROOT/x).exists()]
for p in (ROOT/'tasks').rglob('*.py'):
    py_compile.compile(str(p),doraise=True)
print({'repo_root':str(ROOT),'missing_required':missing,'legacy_root_entries_remaining':remaining})
if missing: raise SystemExit('Missing required post-migration paths.')
if remaining: raise SystemExit('Legacy root entries still remain; migration is incomplete.')
print('[OK] Task-local layout and Python syntax checks passed.')

