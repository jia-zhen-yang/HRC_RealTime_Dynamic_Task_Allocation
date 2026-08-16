import re
import csv
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
INPUT_TXT = Path(BASE_DIR) / "robot_task_time.txt"
OUTPUT_CSV = Path(BASE_DIR) / "robot_task_time.csv"

# 原始資料的 stack 最大編號
# 目前文字檔 stack 是 1~6
MAX_STACK = 6

# 各物件實際堆疊數量
OBJECT_COUNTS = {
    "A": 3,
    "B": 4,
    "C": 5,
    "D": 6,
}

# 各物件需要保留的 layout
LAYOUT_FILTERS = {
    "A": {"LA", "LB"},
    "B": {"LA", "LC"},
    "C": {"LB", "LC"},
    "D": {"LA", "LB", "LC"},
}


def get_object_name(task_name):
    """
    從作業名稱中取得物件名稱
    例如：pick and place A -> A
    """
    return task_name.strip().split()[-1]


def convert_stack(raw_stack, object_name):
    """
    將原始 stack 編號轉成實際物件由上到下的編號

    例：
    A 有 3 個，原始 stack 4, 5, 6
    轉換後為 1, 2, 3
    """
    object_count = OBJECT_COUNTS[object_name]

    first_valid_stack = MAX_STACK - object_count + 1
    converted_stack = raw_stack - first_valid_stack + 1

    if converted_stack < 1 or converted_stack > object_count:
        return None

    return converted_stack


def parse_line(line):
    """
    解析單行文字
    支援有 variant 或沒有 variant 的格式
    """
    pattern = re.compile(
        r"^\s*\d+\.\s*"
        r"(?P<task_name>.*?)\s*\|"
        r".*?\blayout=(?P<layout>[A-Za-z0-9_]+)"
        r".*?\bstack=(?P<stack>\d+):\s*"
        r"(?P<time_sec>\d+(?:\.\d+)?)\s*秒"
    )

    match = pattern.search(line)
    if not match:
        return None

    task_name = match.group("task_name").strip()
    layout = match.group("layout").strip()
    raw_stack = int(match.group("stack"))
    time_sec = float(match.group("time_sec"))

    return task_name, layout, raw_stack, time_sec


def txt_to_csv(input_txt, output_csv):
    rows = []

    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)

            if parsed is None:
                continue

            task_name, layout, raw_stack, time_sec = parsed
            object_name = get_object_name(task_name)

            # 若物件不在參數設定中，跳過
            if object_name not in OBJECT_COUNTS:
                continue

            # layout 篩選
            if layout not in LAYOUT_FILTERS.get(object_name, set()):
                continue

            # stack 編號轉換
            converted_stack = convert_stack(raw_stack, object_name)

            # 若原始 stack 不屬於該物件實際存在的堆疊範圍，跳過
            if converted_stack is None:
                continue

            rows.append({
                "task name": task_name,
                "layout": layout,
                "stack": converted_stack,
                "time(sec)": time_sec,
            })

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task name", "layout", "stack", "time(sec)"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"轉換完成，共輸出 {len(rows)} 筆資料")
    print(f"CSV 檔案已建立：{output_csv}")


if __name__ == "__main__":
    txt_to_csv(INPUT_TXT, OUTPUT_CSV)