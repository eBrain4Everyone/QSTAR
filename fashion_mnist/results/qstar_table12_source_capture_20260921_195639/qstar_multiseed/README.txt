QSTAR FIVE-SEED RUNNER

Purpose
-------
Re-run the Fashion-MNIST experiments behind paper Tables I and II with five
seeds, and automatically produce mean +/- sample-standard-deviation tables.

Default seeds
-------------
1, 42, 48, 550, 2026

Install
-------
Copy the entire qstar_multiseed directory into:

  /scratch/sr7849/QC_June28/

Run
---
  cd /scratch/sr7849/QC_June28
  bash qstar_multiseed/submit_qstar_5seeds.sh

Custom seeds (exactly five; colon separated)
------------------------------------------------
  SEEDS_CSV=2:3:5:7:11 bash qstar_multiseed/submit_qstar_5seeds.sh

Outputs
-------
Every invocation creates a new timestamped directory under multiseed_runs/.
The final paper-ready outputs are placed under that run's summary/ directory:

  table1_paper_mean_std.csv
  table1_paper_mean_std.tex
  table2_paper_mean_std.csv
  table2_paper_mean_std.tex

The numeric summaries and all individual seed rows are also preserved.

Job design
----------
Table I classical:     5 array tasks
Table I Standard QTL: 40 array tasks (5 seeds x 8 configurations)
Table I KetGPT:        5 array tasks (both IDs 22 and 180 per seed)
Table II adaptive:     5 array tasks
Summary:               runs automatically after every array succeeds

No experiment is run on a login node. The submission script only calls sbatch.
