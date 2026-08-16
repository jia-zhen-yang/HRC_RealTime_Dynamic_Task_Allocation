import csv
from pathlib import Path
import os
from typing import Union

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 獲得執行程式的路徑
SCHEDULE_PATH = Path(BASE_DIR) / "schedule_test.csv"
REQUIRED_COLS = ["ID", "command", "object", "zone", "agent", "status", "layout", "human_standard_time", "robot_standard_time", "actual_time"]
    
def load_schedule_csv(csv_path: Union[str, Path] = SCHEDULE_PATH):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到排程檔案：{csv_path}")

    schedule = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        #for row in reader:
            #print(row)

        # 檢查欄位
        missing_cols = [c for c in REQUIRED_COLS if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise ValueError(f"缺少欄位：{missing_cols}，目前欄位：{reader.fieldnames}")

        # 讀取
        for i, row in enumerate(reader, start=2):  # start=2 因為第1列是header
            # 去掉前後空白
            row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}

            # ID 必須是數字
            try:
                row["ID"] = int(row["ID"])
            except Exception:
                print(row["ID"])
                raise ValueError(f"ID 列不是數字")

            # standard_time 必須是數字
            try:
                row["human_standard_time"] = float(row["human_standard_time"])
            except Exception:
                raise ValueError(f"human_standard_time 列")
            try:
                row["robot_standard_time"] = float(row["robot_standard_time"])
            except Exception:
                raise ValueError(f"robot_standard_time 列不是數字")

            # actual_time 空白 -> None；有值 -> float
            if row["actual_time"] == "" or row["actual_time"] is None:
                row["actual_time"] = None
            else:
                try:
                    row["actual_time"] = float(row["actual_time"])
                except Exception:
                    raise ValueError(f"actual_time 列不是數字")

            schedule.append(row)
    
         

    return schedule

if __name__ == "__main__":
    schedule = load_schedule_csv(SCHEDULE_PATH)
    print("Loaded schedule:", schedule)