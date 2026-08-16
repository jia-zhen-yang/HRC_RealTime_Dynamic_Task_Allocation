VNS convergence replay — corrected version
==========================================

Why the old version failed
--------------------------
The previous script defaulted to VNS_dynamic_solver.py. Your extracted folder
is a subfolder of task_monitor, while the actual project modules are normally
stored one folder above it. The corrected script:

1. searches the current folder and its parent folders automatically;
2. tries VNS_dynamic_solver first and falls back to VNS_rescheduler;
3. supports --project-dir for explicitly locating the original project;
4. loads schedule metadata from the original project folder;
5. preserves the original 80-point convergence plot format.

Required files in the original project folder
---------------------------------------------
The parent task_monitor folder should contain:
- VNS_rescheduler.py (or VNS_dynamic_solver.py)
- read_schedule.py
- precedence_matrix.py
- schedule.csv or the schedule file used by read_schedule.py
- Standard Time Calculation/robot_task_time.csv

Recommended PowerShell command
------------------------------
Run the command inside vns_convergence_replay_bundle:

python replay_vns_convergence.py `
  --events dynamic_events_20260621_214951.csv `
  --gantt gantt_data_20260621_214951.txt `
  --output-dir convergence_20260621_214951 `
  --max-iter 80 `
  --seed 43 `
  --shaking-threshold 50 `
  --project-dir .. `
  --solver-module VNS_rescheduler

One-line form:

python replay_vns_convergence.py --events dynamic_events_20260621_214951.csv --gantt gantt_data_20260621_214951.txt --output-dir convergence_20260621_214951 --max-iter 80 --seed 43 --shaking-threshold 50 --project-dir .. --solver-module VNS_rescheduler

Automatic solver selection
--------------------------
You may omit --solver-module VNS_rescheduler. The corrected default is auto:
- first try VNS_dynamic_solver
- if absent, use VNS_rescheduler

Expected startup messages
-------------------------
[module] solver: VNS_rescheduler
[module] loader: ...\read_schedule.py
[module] precedence: ...\precedence_matrix.py

Expected outputs
----------------
- convergence_failure_1.png
- convergence_failure_2.png
- convergence_failure_3.png
- convergence_failure_4.png
- convergence_all_failures.png
- convergence_history.csv
- convergence_summary.csv
- reconstructed_failure_states.csv

Reproducibility note
--------------------
The original experiment did not save the VNS random-generator state or its
per-iteration history. This is a reproducible new run from the reconstructed
failure states, not recovery of the exact historical curve.
