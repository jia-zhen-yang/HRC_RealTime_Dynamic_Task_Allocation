import csv

def build_precedence_matrix(schedule, case_1 = False):
    """
    Build task precedence matrix based on kitting rules.

    P[i][j] = 1  → 任務 i 必須在任務 j 之前完成
    P[i][j] = 0  → 沒有先後限制
    """
    n = len(schedule)
    P = [[0 for _ in range(n)] for _ in range(n)]

    # assign kit id by replace
    kit_id = 0
    task_kit = []

    for task in schedule:
        if task["command"] == "replace":
            kit_id += 1
        task_kit.append(kit_id)

    # ---------- kit-level precedence ----------
    for i in range(n):
        for j in range(n):
            # 如果 task i 在較早的 kit，task j 在較晚的 kit
            if task_kit[i] < task_kit[j]:
                P[i][j] = 1

    # ------- replace → pick within same kit (replace_k  ≺  pick_k_*)-------- (CASE 1)
    if case_1:
        for i in range(n):
            if schedule[i]["command"] != "replace":
                continue

            for j in range(n):
                # 對每一個 kit，將replace 指定為該 kit 所有 pick & place 的唯一前置任務
                if task_kit[i] == task_kit[j] and schedule[j]["command"] != "replace":
                    P[i][j] = 1
    

    return P

def print_precedence_matrix(P, task_ids):
    # column header
    header = " "*4 + " ".join(f"{id:>3}" for id in task_ids) # 右對齊、寬度 3
    print(header)

    # each row 
    for i, row in enumerate(P):
        row_str = " ".join(f"{val:>3}" for val in row)
        print(f"{task_ids[i]:>3} {row_str}")


if __name__ == "__main__":
    from read_schedule import load_schedule_csv

    schedule = load_schedule_csv()
    P = build_precedence_matrix(schedule)

    task_ids = [task["ID"] for task in schedule]

    print("Precedence Matrix :\n")
    print_precedence_matrix(P, task_ids)
