# 離線模擬

from read_schedule import load_schedule_csv
from precedence_matrix import build_precedence_matrix
from VNS_rescheduler import solver
import matplotlib.pyplot as plt
from tqdm import tqdm

n_iteration = 200
schedule = load_schedule_csv()
P = build_precedence_matrix(schedule)
makespan_history = []
all_solution_space = list()
search_time_sum = 0


for i in tqdm(range(n_iteration)):
    new_schedule, base_cost, new_cost, search_time, solution_space, history, shake = solver(
            schedule,
            P,
            start_kit_id=1,
            max_iter=15,
            shaking_threshold = 10,
        )
    search_time_sum += search_time
    makespan_history.append(new_cost)
    for s in solution_space:
        if s not in all_solution_space:
            all_solution_space.append(s)

    #print(f"{i} Dynamic makespan : {new_cost}")
    #print(f"Number of unique solutions: {len(solution_space)}")

print(f"best makespan across all trials:{min(makespan_history)}")
print(f"average makespan across all trials:{sum(makespan_history) / len(makespan_history)}")
print(f"solution spaces across all trials:{len(all_solution_space)}")
print(f"Average search time: {search_time_sum / n_iteration} seconds")

plt.figure(figsize=(15,4))
plt.plot(makespan_history)
plt.title(f"VNS Test for {n_iteration} Times")
plt.xlabel("Iteration")
plt.ylabel("Best makespan")
plt.tight_layout()
plt.show()
