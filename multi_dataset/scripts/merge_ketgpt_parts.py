#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--seeds", default="1:42:48:550:2026")
    args = ap.parse_args()
    root = Path(args.run_root)
    seeds = [int(x) for x in args.seeds.split(":")]

    for seed in seeds:
        frames = []
        cframes = []
        for cid in (160, 180):
            p = root / f"table1/ketgpt_parts/candidate_{cid}/seed_{seed}.csv"
            cp = root / f"table1/ketgpt_parts/candidate_{cid}/seed_{seed}_candidates.csv"
            if not p.exists():
                raise FileNotFoundError(p)
            df = pd.read_csv(p)
            hit = df[(df["seed"].astype(int) == seed) & (df["ketgpt_id"].astype(int) == cid)]
            if len(hit) != 1:
                raise RuntimeError(f"Expected exactly one row for seed={seed}, candidate={cid}; got {len(hit)} in {p}")
            frames.append(hit)
            if cp.exists():
                cframes.append(pd.read_csv(cp))

        merged = pd.concat(frames, ignore_index=True).sort_values("ketgpt_id")
        atomic_csv(merged, root / f"table1/ketgpt/seed_{seed}.csv")
        if cframes:
            cm = pd.concat(cframes, ignore_index=True).drop_duplicates(subset=["ketgpt_id"]).sort_values("ketgpt_id")
            atomic_csv(cm, root / f"table1/ketgpt/seed_{seed}_candidates.csv")
        print(f"MERGED seed {seed}: candidates {merged.ketgpt_id.astype(int).tolist()}")


if __name__ == "__main__":
    main()
