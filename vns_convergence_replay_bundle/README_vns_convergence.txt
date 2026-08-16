VNS convergence replay for experiment 20260621_214951
====================================================

Files
-----
- replay_vns_convergence.py
- dynamic_events_20260621_214951.csv
- gantt_data_20260621_214951.txt

Recommended placement
---------------------
Copy replay_vns_convergence.py and the two raw-data files into the original
scheduling project folder containing:

- VNS_dynamic_solver.py
- read_schedule.py
- precedence_matrix.py
- schedule.csv (at the path used by read_schedule.py)
- Standard Time Calculation/robot_task_time.csv

Run
---
python replay_vns_convergence.py \
  --events dynamic_events_20260621_214951.csv \
  --gantt gantt_data_20260621_214951.txt \
  --output-dir convergence_20260621_214951 \
  --max-iter 80 \
  --seed 43 \
  --shaking-threshold 50

Outputs
-------
- convergence_failure_1.png
- convergence_failure_2.png
- convergence_failure_3.png
- convergence_failure_4.png
- convergence_all_failures.png
- convergence_history.csv
- convergence_summary.csv
- reconstructed_failure_states.csv

Plot format
-----------
The four individual figures intentionally retain the convergence plot block in
VNS_rescheduler.py:

plt.figure(figsize=(10, 3))
plt.plot(history)
plt.title("Convergence (best makespan over iterations)")
plt.xlabel("Iteration")
plt.ylabel("Best makespan")
plt.tight_layout()

Therefore 80 recorded history values are displayed at iteration indices 0-79.

Reproducibility note
--------------------
The exported experiment files do not contain the original random-number state
or per-iteration history. This program reruns a new reproducible VNS search from
each recorded failure state. It does not claim to recover the exact historical
curve that occurred during the experiment.
