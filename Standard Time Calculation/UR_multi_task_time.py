import csv
import math
import socket
import time
from pathlib import Path
import os

# =========================
# 使用者設定
# =========================
# ROBOT_IP = "192.168.50.204"  # 930 真實機械手臂 ip
ROBOT_IP = "192.168.0.231"      # 虛擬機 ip
ROBOT_PORT = 30002              # UR Secondary Interface
CALLBACK_PORT = 50001           # Python 端接收任務狀態
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'robot_task_waypoint.csv')

# 固定姿態，沿用你原本程式
RX = 0.0
RY = 3.14159
RZ = 0.0

# 運動參數 moveL
ACC = 2.0
VEL = 0.4

# 運動參數 moveJ
ACC_DEG = 250.0
VEL_DEG = 200.0
ACC_RAD = math.radians(ACC_DEG)
VEL_RAD = math.radians(VEL_DEG)

# BOT 點位：每個任務開始前都會先回到 BOT
BOT = (0.0, -0.23, 0.15)

# 是否依 action 欄位變化控制夾爪 / 數位輸出
# R -> M 代表抓取，M -> R 代表放開
ENABLE_DIGITAL_OUT = True
DIGITAL_OUT_PIN = 4
GRIPPER_SETTLE_TIME = 0.5
BLEND_RADIUS = 0.01


def get_local_ip(target_ip: str) -> str:
    """
    自動取得本機 IP
#   讓 UR 控制器可以回傳 START / DONE
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target_ip, 1))
        return s.getsockname()[0]
    finally:
        s.close()


def safe_msg(text: str) -> str:
    """避免訊息中出現換行；保留 | 作為分隔符"""
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def task_display_name(task: dict) -> str:
    """產生輸出用任務名稱。"""
    variant = task.get("variant") or ""
    layout = task.get("layout") or ""
    stack = task.get("stack") or ""

    parts = [task["task"]]
    if variant:
        parts.append(f"variant={variant}")
    if layout:
        parts.append(f"layout={layout}")
    if stack:
        parts.append(f"stack={stack}")
    return " | ".join(parts)


def load_tasks_from_csv(csv_path: str):
    """
    讀取 robot_task_waypoint.csv

    規則：
    - 空白列略過
    - task 欄位以 # Task: 開頭的列視為註解列，略過
    - 連續相同 task / variant / layout / stack 的列組成一個任務
    - 每列的 x, y, z 是該任務需要依序到達的點位
    """
    tasks = []
    current = None

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_task = (row.get("task") or "").strip()
            if not raw_task or raw_task.startswith("#"):
                continue

            try:
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
            except (TypeError, ValueError):
                continue

            variant = (row.get("variant") or "").strip()
            layout = (row.get("layout") or "").strip()
            stack_raw = (row.get("stack") or "").strip()
            stack = ""
            if stack_raw:
                try:
                    stack = str(int(float(stack_raw)))
                except ValueError:
                    stack = stack_raw

            key = (raw_task, variant, layout, stack)
            if current is None or current["key"] != key:
                current = {
                    "key": key,
                    "task": raw_task,
                    "variant": variant,
                    "layout": layout,
                    "stack": stack,
                    "points": [],
                }
                tasks.append(current)

            current["points"].append({
                "x": x,
                "y": y,
                "z": z,
                "action": (row.get("action") or "").strip(),
                "from": (row.get("from") or "").strip(),
                "to": (row.get("to") or "").strip(),
                "resolved_to": (row.get("resolved_to") or "").strip(),
            })

    return tasks


def pose(x: float, y: float, z: float) -> str:
    return f"p[{x:.6f}, {y:.6f}, {z:.6f}, {RX}, {RY}, {RZ}]"


def same_xy(a, b, eps=1e-6) -> bool:
    return abs(a[0] - b[0]) < eps and abs(a[1] - b[1]) < eps


def build_move_line(prev_xyz, next_xyz) -> str:
    """
    運動規則：
    - 每一段移動都使用 movel
    """
    x, y, z = next_xyz
    return f"  movel({pose(x, y, z)}, a={ACC}, v={VEL})"


def send_line_urscript(message: str, socket_name="status_sock", indent="    ") -> str:
    """URScript：送出一行文字，以 \n 結尾。"""
    message = safe_msg(message)
    return (
        f'{indent}socket_send_string("{message}", "{socket_name}")\n'
        f'{indent}socket_send_byte(10, "{socket_name}")'
    )


def build_urscript(pc_ip: str, callback_port: int, tasks) -> str:
    lines = []
    lines.append("def run_all_tasks():")
    lines.append(f'  status_ok = socket_open("{pc_ip}", {callback_port}, "status_sock")')
    lines.append("")

    for idx, task in enumerate(tasks, start=1):
        name = task_display_name(task)
        lines.append(f"  # ================= Task {idx}: {name} =================")
        lines.append("  # 每個任務開始前先回 BOT，此段不計入該任務時間")
        lines.append(f"  movel({pose(*BOT)}, a={ACC}, v={VEL})")
        lines.append("")
        lines.append("  if status_ok:")
        lines.append(send_line_urscript(f"TASK_START|{idx}|{name}"))
        lines.append("  end")
        lines.append("")

        prev_xyz = BOT
        prev_action = None
        for point in task["points"]:
            next_xyz = (point["x"], point["y"], point["z"])
            action = point.get("action", "")
            label = point.get("resolved_to") or point.get("to") or "waypoint"
            lines.append(f"  # {label}, action={action}")

            if ENABLE_DIGITAL_OUT and prev_action and action and prev_action != action:
                if prev_action == "R" and action == "M":
                    lines.append("  # R -> M")
                    lines.append(f"  set_digital_out({DIGITAL_OUT_PIN}, True)")
                    lines.append(f"  sleep({GRIPPER_SETTLE_TIME})")
                elif prev_action == "M" and action == "R":
                    lines.append("  # M -> R")
                    lines.append(f"  set_digital_out({DIGITAL_OUT_PIN}, False)")
                    lines.append(f"  sleep({GRIPPER_SETTLE_TIME})")

            lines.append(build_move_line(prev_xyz, next_xyz))

            prev_xyz = next_xyz
            if action:
                prev_action = action

        lines.append("")
        lines.append("  if status_ok:")
        lines.append(send_line_urscript(f"TASK_DONE|{idx}|{name}"))
        lines.append("  end")
        lines.append("")

    lines.append("  if status_ok:")
    lines.append(send_line_urscript("ALL_DONE"))
    lines.append('    socket_close("status_sock")')
    lines.append("  end")
    lines.append("end")
    lines.append("")
    lines.append("run_all_tasks()")
    lines.append("")
    return "\n".join(lines)


def main():
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 檔案: {csv_path.resolve()}")

    tasks = load_tasks_from_csv(str(csv_path))
    if not tasks:
        raise RuntimeError("CSV 中沒有讀到任何任務點位")

    pc_ip = get_local_ip(ROBOT_IP)
    print(f"本機 IP: {pc_ip}")
    print(f"讀取任務數量: {len(tasks)}")
    print(f"等待 UR 回傳任務狀態，監聽 Port: {CALLBACK_PORT}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("0.0.0.0", CALLBACK_PORT))
        server_sock.listen(1)

        urscript = build_urscript(pc_ip, CALLBACK_PORT, tasks)

        # 可選：輸出產生的 URScript，方便除錯
        Path("generated_multi_task.urscript").write_text(urscript, encoding="utf-8")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as robot_sock:
            robot_sock.connect((ROBOT_IP, ROBOT_PORT))
            robot_sock.sendall(urscript.encode("utf-8"))

        print("URScript 已送出，等待機器人執行...")

        conn, addr = server_sock.accept()
        task_start_time = {}
        task_elapsed = {}
        task_names = {}
        buffer = ""

        with conn:
            print(f"已連線自 UR 控制器: {addr}")

            while True:
                data = conn.recv(1024)
                if not data:
                    break

                buffer += data.decode("utf-8", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    msg = line.strip()
                    if not msg:
                        continue

                    parts = msg.split("|", 2)
                    event = parts[0]

                    if event == "TASK_START" and len(parts) == 3:
                        task_id = int(parts[1])
                        name = parts[2]
                        task_names[task_id] = name
                        task_start_time[task_id] = time.perf_counter()
                        print(f"開始任務 {task_id}: {name}")

                    elif event == "TASK_DONE" and len(parts) == 3:
                        task_id = int(parts[1])
                        name = parts[2]
                        if task_id in task_start_time:
                            elapsed = time.perf_counter() - task_start_time[task_id]
                            task_elapsed[task_id] = elapsed
                            print(f"完成任務 {task_id}: {name}，時間: {elapsed:.3f} 秒")
                        else:
                            print(f"收到 TASK_DONE，但沒有 TASK_START: {task_id} {name}")

                    elif event == "ALL_DONE":
                        print("\n================ 任務完成時間總表 ================")
                        for task_id in sorted(task_elapsed):
                            print(f"{task_id:02d}. {task_names[task_id]}: {task_elapsed[task_id]:.3f} 秒\n")
                        return

                    else:
                        print(f"收到未知訊息: {msg}")

    print("程式結束")


if __name__ == "__main__":
    main()
