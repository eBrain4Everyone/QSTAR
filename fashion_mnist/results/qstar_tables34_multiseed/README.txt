QSTAR TABLES III AND IV: FIVE-SEED RUNNER
==========================================

Purpose
-------
This runner produces mean +/- sample standard deviation results for:

  Table III: Ants-vs-Bees fixed heads
    - Classical Linear
    - Classical MLP
    - KetGPT candidate #180

  Table IV: Fashion-MNIST KetGPT resource ablation
    - Candidate #22
    - Candidate #160
    - Candidate #180

Default seeds: 1, 42, 48, 550, 2026.

Design
------
Table III uses five Slurm array tasks, one per seed. Each task runs the original
Step5_Full_Adaptive/step5_full_adaptive_ketgpt_antsbees.py program and extracts
the three fixed-head rows.

Table IV uses fifteen Slurm array tasks, one per (seed, candidate) pair. The
wrapper selects the requested KetGPT ID explicitly and then uses the original
KetGPT_Qubit_Ablation/run_ketgpt_qubit_ablation.py training/evaluation code.
This avoids candidate-order RNG coupling and gives each candidate the same five
independent seeds.

Each task receives a private PennyLane dataset directory, avoiding concurrent
HDF5 cache locking failures.

Install and submit
------------------
Copy the archive into /scratch/sr7849/QC_June28, then run:

  cd /scratch/sr7849/QC_June28
  tar -xzf qstar_tables34_multiseed_runner.tar.gz
  bash qstar_tables34_multiseed/submit_tables34_5seeds.sh

This command only submits Slurm jobs; it does not train on the login node.

Monitor
-------
The submission command prints all job IDs. It also writes them to:

  tables34_multiseed_runs/<run_timestamp>/run_manifest.txt

Use the printed squeue command, or inspect completed tasks with sacct.

Expected workload
-----------------
  - 5 Table-III Ants-vs-Bees tasks
  - 15 Table-IV Fashion-MNIST tasks
  - 1 dependent summary task

Outputs
-------
After successful completion:

  tables34_multiseed_runs/<run_timestamp>/summary/
  |-- table3_all_seed_results.csv
  |-- table3_mean_std_numeric.csv
  |-- table3_paper_mean_std.csv
  |-- table3_paper_mean_std.tex
  |-- table4_all_seed_results.csv
  |-- table4_mean_std_numeric.csv
  |-- table4_paper_mean_std.csv
  `-- table4_paper_mean_std.tex

Every paper row is validated to have n=5. Standard deviations use pandas'
sample standard deviation (ddof=1).

Different seeds
---------------
Exactly five seeds are required:

  SEEDS_CSV=2:3:5:7:11 \
  bash qstar_tables34_multiseed/submit_tables34_5seeds.sh

