#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('run_root')
args=p.parse_args()
root=Path(args.run_root)

expected={
 'table1/classical':5,
 'table1/standard_qtl':40,
 'table1/ketgpt':5,
 'table2/adaptive180':5,
 'table2/adaptive160':5,
}
print('QSTAR new-dataset run status')
print('run_root:',root)
for rel,n in expected.items():
    files=[x for x in (root/rel).glob('seed_*.csv') if not x.name.endswith('_candidates.csv')]
    print(f'{rel:28s} {len(files):2d}/{n}')

traj=list(root.glob('checkpoints/**/trajectory.json'))
full=0; accelerated=0; reasons={}
for t in traj:
    try:
        r=json.loads(t.read_text()); reason=r.get('stop_reason','unknown')
        reasons[reason]=reasons.get(reason,0)+1
        full += int(bool(r.get('completed_full_run')))
        accelerated += int(reason in {'flashjt_validated','early_stopping'})
    except Exception: pass
print(f'trajectories: {len(traj)} | full-10: {full} | accelerated stops: {accelerated}')
print('stop reasons:', reasons)
status=root/'summary/publication_status.json'
if status.exists():
    print('\npublication status:')
    print(status.read_text())
else:
    print('\nNo summary/publication_status.json yet.')
