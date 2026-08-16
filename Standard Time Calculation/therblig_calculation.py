import pandas as pd
import math
from pathlib import Path
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 獲得執行程式的路徑

# =========================
# 參數設定
# =========================
THERBLIG_FILE = Path(BASE_DIR) / "therblig_calculation.csv"
COORD_FILE = Path(BASE_DIR) / "coord_for_human.csv"
PROCESS_TIME_FILE = Path(BASE_DIR) / "therblig_process_time.csv"

TABLE_TMU_SCALE = 10 # 表格單位是0.1TMU
TMU_TO_SEC = 0.036  # 1 TMU = 0.036 sec

# =========================
# 讀取資料
# =========================
df_calc = pd.read_csv(THERBLIG_FILE)
df_coord = pd.read_csv(COORD_FILE)
df_time = pd.read_csv(PROCESS_TIME_FILE)

# 座標表 / 時間表建立索引
coord_map = df_coord.set_index("Name")
time_map = df_time.set_index("Therblig")

# =========================
# helper
# =========================
def calc_distance(from_name: str, to_name: str) -> float:
    """計算兩點之間的 3D 歐式距離，單位：cm"""
    p1 = coord_map.loc[from_name, ["x_coord", "y_coord", "z_coord"]].astype(float).values
    p2 = coord_map.loc[to_name, ["x_coord", "y_coord", "z_coord"]].astype(float).values
    return math.dist(p1, p2)


def get_available_distance_levels(prefix: str, move_type: str):
    """
    從時間表中抓出某類動素可用的距離級距
    例如 prefix='R', move_type='B' -> [2,4,6,...,80]
    """
    levels = []
    for therblig in time_map.index:
        if therblig.startswith(prefix) and therblig.endswith(move_type):
            middle = therblig[len(prefix):-len(move_type)]
            if middle.isdigit():
                levels.append(int(middle))
    return sorted(levels)


def round_up_to_mtm_level(distance: float, levels: list[int]) -> int:
    """
    MTM 查表時，距離採『往上對應到最近級距』
    例如距離 20.1 -> 對應 22
    """
    for lv in levels:
        if distance <= lv:
            return lv
    return levels[-1]  # 若超過最大級距，先使用表內最大值


def build_therblig_key(name: str, distance: float, move_type: str) -> str:
    """把 R/M 動素組成查表鍵值，例如 R20B、M45B"""
    levels = get_available_distance_levels(name, move_type)
    if not levels:
        raise ValueError(f"找不到 {name} + {move_type} 的距離級距")
    level = round_up_to_mtm_level(distance, levels)
    return f"{name}{level}{move_type}"


def lookup_time_tmu(therblig_key: str, hand: str) -> float:
    """依 therblig key 與手別查 TMU"""
    if therblig_key not in time_map.index:
        raise KeyError(f"時間表中找不到 {therblig_key}")
    if hand not in time_map.columns:
        raise KeyError(f"時間表中找不到欄位 {hand}")
    return float(time_map.loc[therblig_key, hand]/ TABLE_TMU_SCALE)


def split_task_blocks(df: pd.DataFrame):
    """
    依 END 切出每個任務 block
    """
    blocks = []
    start = 0
    for i, row in df.iterrows(): # 每一列
        if str(row["Name"]).strip() == "END":
            block = df.iloc[start:i+1].reset_index(drop=True) # 把切出來的 block 重新編號索引
            blocks.append(block) # block 本身也是一個 DataFrame
            start = i + 1
    return blocks


def get_block_label(block: pd.DataFrame) -> str:
    """
    產生每個任務 block 的名稱
    例如：
    pick and place B + Layout A -> pick and place B (Layout A)
    """
    main_task = block.loc[0, "Task"]

    variant = None
    if len(block) > 1 and pd.notna(block.loc[1, "Task"]): # 這個 block 至少有兩列且第二列的 "Task" 不是空值
        second_task = str(block.loc[1, "Task"]).strip()
        if second_task != "" and second_task != main_task:
            variant = second_task

    if variant:
        return f"{main_task} ({variant})"
    return str(main_task)


def calculate_block_time(block: pd.DataFrame, hand: str):
    """
    計算單一任務 block 的總時間（TMU）
    hand = 'RH' or 'LH'
    回傳:
      total_tmu, details(list)
    """
    total_tmu = 0.0
    details = []

    for _, row in block.iterrows():
        name = row["Name"]

        if pd.isna(name) or str(name).strip() == "END":
            continue

        from_pt = row["From"]
        to_pt = row["To"]
        move_type = row["Type"] if pd.notna(row["Type"]) else ""

        # AGENT 依 hand 替換
        if from_pt == "AGENT":
            from_pt = hand
        if to_pt == "AGENT":
            to_pt = hand

        # R / M 要算距離後查表
        if name in ["R", "M"]:
            distance = calc_distance(from_pt, to_pt)
            therblig_key = build_therblig_key(name, distance, str(move_type))
            tmu = lookup_time_tmu(therblig_key, hand)

            details.append({
                "Name": name,
                "From": from_pt,
                "To": to_pt,
                "Type": move_type,
                "Distance(cm)": round(distance, 3),
                "TherbligKey": therblig_key,
                "TMU": tmu
            })

        # 其他動素直接查表
        else:
            therblig_key = str(name)
            tmu = lookup_time_tmu(therblig_key, hand)

            details.append({
                "Name": name,
                "From": from_pt,
                "To": to_pt,
                "Type": move_type,
                "Distance(cm)": None,
                "TherbligKey": therblig_key,
                "TMU": tmu
            })

        total_tmu += tmu

    return total_tmu, details


# =========================
# 主程式
# =========================
blocks = split_task_blocks(df_calc)

results = []

for block in blocks:
    label = get_block_label(block)
    main_task = str(block.loc[0, "Task"]).strip()

    if main_task.startswith("pick and place"):
        total_tmu, details = calculate_block_time(block, hand="RH")
        results.append({
            "Task": label,
            "Hand": "RH",
            "Total_TMU": total_tmu,
            "Total_sec": total_tmu * TMU_TO_SEC,
            "Details": details
        })

    elif main_task == "replace":
        for hand in ["LH", "RH"]:
            total_tmu, details = calculate_block_time(block, hand=hand)
            results.append({
                "Task": label,
                "Hand": hand,
                "Total_TMU": total_tmu,
                "Total_sec": total_tmu * TMU_TO_SEC,
                "Details": details
            })

print("===== 任務總時間 =====")
for r in results:
    print(f"{r['Task']} | {r['Hand']} | {r['Total_TMU']:.1f} TMU | {r['Total_sec']:.3f} sec")

print("\n===== 各任務明細 =====")
for r in results:
    print(f"\n--- {r['Task']} | {r['Hand']} ---")
    detail_df = pd.DataFrame(r["Details"])
    print(detail_df.to_string(index=False))