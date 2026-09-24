QSTAR TABLE EXPORTS
===================

FILES
-----

seed_progress.csv
    One row per Table-I method/seed.
    Shows epoch, seed role, result availability and current state.

table1_all_aggregates.csv
    All Table-I methods.
    Finished rows use all five seeds.
    Incomplete rows aggregate only currently available FINAL results.

table1_ready_entries.csv
    Only fully finished five-seed Table-I rows.
    These are the rows eligible for paper-table use.

table1_incomplete_preview.csv
    Incomplete Table-I experiments.
    Metrics are provisional and are for monitoring only.

table2_component_progress.csv
    Progress of confidence_linear, matched_mlp and ketgpt_quantum
    components for adaptive candidate 160 and 180.

table2_component_aggregates.csv
    Diagnostic component aggregates.
    These are NOT automatically Table-II routing rows.

table2_all_aggregates.csv
    Threshold/routing aggregates discovered from seed-level Table-II
    output files, if such files exist.

table2_ready_entries.csv
    Fully finished five-seed Table-II routing rows only.

table2_incomplete_preview.csv
    Provisional Table-II routing aggregates.

raw_discovered_result_rows.csv
    Deduplicated seed-level rows found in the experiment CSVs.

discovered_csv_schemas.csv
    File/column inventory used to audit result discovery.

status_dictionary.csv
    Meaning of the training/readiness labels.


TRAINING PROTOCOL LABELS
------------------------

FULL_TRAINING
    All five final seed results exist.
    Historical seeds are full histories and both assisted seeds reached epoch 10.

FLASHJT_ASSISTED
    All five final results exist.
    Historical seeds are full histories.
    Assisted seeds terminated through the validated FlashJT stopping point.

HYBRID_FULL_FLASHJT
    All five final results exist.
    One assisted seed stopped through FlashJT while another continued to epoch 10.

INCOMPLETE
    Fewer than five final result rows exist.
    Any aggregate is provisional and MUST NOT be quoted as a final five-seed table entry.

EARLY_FINAL_REVIEW / HYBRID_EARLY_FINAL_REVIEW
    A final assisted run appears to end at an unusual epoch (for example 7-9).
    Inspect provenance before using it in the paper.


IMPORTANT
---------

An epoch >= 6 by itself is NOT treated as a completed FlashJT run.

The script uses final result rows as the primary completion signal.
The trajectory epoch is then used to document whether the final result came
from full training or an assisted/early endpoint.

For Table II, component-training metrics and final routing metrics are kept
separate deliberately. The script will not fabricate a Table-II routing row
from component metrics.
