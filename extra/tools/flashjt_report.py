#!/usr/bin/env python3
from pathlib import Path
import argparse, json
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('run_root')
p.add_argument('--out',default=None)
args=p.parse_args()
root=Path(args.run_root)
rows=[]
for vpath in sorted(root.glob('checkpoints/**/flashjt_verification_epoch6.json')):
    try:
        v=json.loads(vpath.read_text())
        fpath=vpath.with_name('flashjt_forecast_epoch3.json')
        f=json.loads(fpath.read_text()) if fpath.exists() else {}
        traj=vpath.with_name('trajectory.json')
        t=json.loads(traj.read_text()) if traj.exists() else {}
        meta=t.get('metadata',f.get('metadata',{}))
        rows.append({
            'path':str(vpath.relative_to(root)),
            'seed':t.get('seed',f.get('seed')),
            'dataset':meta.get('dataset'),
            'table':meta.get('table'),
            'model_family':meta.get('model_family'),
            'config_id':meta.get('config_id'),
            'candidate_id':meta.get('candidate_id'),
            'history_count_at_forecast':f.get('history_count'),
            'allow_stop':v.get('allow_stop'),
            'reason':v.get('reason'),
            'max_relative_error':v.get('max_relative_error'),
            'mean_relative_error':v.get('mean_relative_error'),
            'checked_values':v.get('checked_values'),
            'final_stop_reason':t.get('stop_reason'),
            'completed_full_run':t.get('completed_full_run'),
        })
    except Exception as e:
        rows.append({'path':str(vpath.relative_to(root)),'reason':f'parse_error:{e}'})
df=pd.DataFrame(rows)
out=Path(args.out) if args.out else root/'summary/flashjt_audit.csv'
out.parent.mkdir(parents=True,exist_ok=True)
df.to_csv(out,index=False)
print(out)
if len(df):
    print(df.to_string(index=False))
else:
    print('No FlashJT verification files found yet.')
