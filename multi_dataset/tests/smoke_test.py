#!/usr/bin/env python3
from pathlib import Path
import json
import math
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qstar_new.common import TrainingController, FlashJT, atomic_json_dump


def synthetic_history(root: Path, count=24):
    metrics = ["train_loss","accuracy","precision_macro","recall_macro","f1_macro","auc_ovr_macro"]
    for i in range(count):
        epochs=[]
        for e in range(1,11):
            acc=0.45+0.30*(1-math.exp(-e/3.0))+i*1e-4
            m={
                "train_loss":1.5*math.exp(-e/3.0)+0.2,
                "accuracy":acc,
                "precision_macro":acc-0.01,
                "recall_macro":acc-0.02,
                "f1_macro":acc-0.015,
                "auc_ovr_macro":min(0.99,acc+0.15),
            }
            epochs.append({"epoch":e,"metrics":m})
        p=root/f"checkpoints/h{i}/trajectory.json"
        atomic_json_dump({
            "metadata":{"dataset":"cifar10","table":"table1","model_family":"standard_qtl","config_id":f"q{i%8}","candidate_id":0,"num_qubits":8,"circuit_depth":2,"shots":100,"lr":1e-3,"batch_size":16},
            "seed":i,"epochs":epochs,"completed_full_run":True,"stop_reason":"completed"
        },p)


def main():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        synthetic_history(root)
        fj=FlashJT(min_history_runs=20)
        assert fj.fit_from_run_root(root)==24
        rec=json.loads((root/'checkpoints/h0/trajectory.json').read_text())
        early={x['epoch']:x['metrics'] for x in rec['epochs'][:3]}
        pred=fj.predict(early, rec['metadata'])
        assert set(pred)==set(range(4,11))
        perfect={x['epoch']:x['metrics'] for x in rec['epochs'][3:]}
        fj.history_count=24
        dec=fj.verify(550, perfect, {e:perfect[e] for e in (4,5,6)})
        assert dec.allow_stop
        assert not fj.verify(42, perfect, {e:perfect[e] for e in (4,5,6)}).allow_stop

        # Checkpoint/resume path with a cheap classical toy model.
        x=torch.randn(96,4); y=torch.randint(0,2,(96,))
        train=DataLoader(TensorDataset(x[:64],y[:64]),batch_size=16,shuffle=True)
        val=DataLoader(TensorDataset(x[64:80],y[64:80]),batch_size=16)
        model=torch.nn.Linear(4,2)
        controller=TrainingController(
            run_root=root,
            checkpoint_dir=root/'checkpoints/toy',
            metadata={"dataset":"toy","table":"table1","model_family":"classical","config_id":"toy","candidate_id":0,"num_qubits":0,"circuit_depth":0,"shots":0,"lr":1e-3,"batch_size":16},
            seed=1,epochs=3,force_full=True,enable_flashjt=False,enable_early_stopping=False,
        )
        result=controller.train(model,train,val,torch.device('cpu'))
        assert result['epochs_executed']==3
        for name in ['checkpoint_latest.pt','checkpoint_best.pt','checkpoint_final.pt','checkpoint_epoch003.pt','trajectory.json']:
            assert (root/'checkpoints/toy'/name).exists(), name

        print('SMOKE TEST PASS')
        print(json.dumps({'flashjt_history':24,'checkpoint_epoch':result['epochs_executed'],'flashjt_gate':dec.to_dict()},indent=2))

if __name__=='__main__': main()
