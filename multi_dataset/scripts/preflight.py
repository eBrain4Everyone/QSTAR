#!/usr/bin/env python3
from pathlib import Path
import importlib
import json
import os
import sys

project=Path(os.environ.get('PROJECT_ROOT','/scratch/sr7849/QC_June28'))
package=Path(os.environ.get('PACKAGE_ROOT',str(project/'ICASSP_multiseed_new_datasets')))
h5=Path(os.environ.get('KETGPT_SOURCE_H5',str(project/'tables34_multiseed_runs/run_20260816_151916/pennylane_data/table4/seed_1_candidate_160/ketgpt/ketgpt.h5')))

ok=True
print('QSTAR new-dataset preflight')
print('PROJECT_ROOT =',project)
print('PACKAGE_ROOT =',package)
for m in ['torch','torchvision','numpy','pandas','sklearn','psutil','pennylane']:
    try:
        mod=importlib.import_module(m)
        print(f'OK   import {m:12s} {getattr(mod,"__version__","")}')
    except Exception as e:
        print(f'FAIL import {m}: {e}')
        ok=False
for rel in ['run_qstar_baselines_new.py','run_qstar_ketgpt_new.py','run_qstar_adaptive_new.py','scripts/submit_dataset.sh','scripts/summarize_dataset.py']:
    p=package/rel
    print(('OK  ' if p.exists() else 'FAIL'),p)
    ok &= p.exists()
print(('OK  ' if h5.exists() else 'WARN'), 'KetGPT HDF5', h5, (f'{h5.stat().st_size} bytes' if h5.exists() else 'missing; PennyLane will download into unique task paths'))
for ds in ['cifar10','kmnist','svhn']:
    c=package/'cache'/ds
    valid=(c/'manifest.json').exists() and (c/'train.pt').exists() and (c/'test.pt').exists()
    print(('READY' if valid else 'MISS '), f'feature cache {ds}: {c}')
print('\nPREFLIGHT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 2)
