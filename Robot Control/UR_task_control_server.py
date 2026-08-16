import socket
import csv
import math
import os

# =========================
# 參數設定
# =========================
# ROBOT_IP = "192.168.50.204"  # 930 真實機械手臂 ip
ROBOT_IP = "192.168.0.231"      # 虛擬機 ip
ROBOT_PORT = 30002
CALLBACK_PORT = 50001
SERVER_HOST = "0.0.0.0"   # 作為 server，監聽所有網卡
SERVER_PORT = 5050

RX = 0.0
RY = 3.14159
RZ = 0.0

ACC = 2.0
VEL = 0.4

ACC_RAD = math.radians(250.0)
VEL_RAD = math.radians(200.0)

BOT = (0.0, -0.23, 0.15)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'robot_task_waypoint.csv')

# =========================
# 自動取得 IP
# =========================
def get_local_ip(target_ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((target_ip, 1))
    ip = s.getsockname()[0]
    s.close()
    return ip

PC_IP = get_local_ip(ROBOT_IP)

# =========================
# CSV 讀取
# =========================
def load_tasks(csv_path):
    tasks = []
    current = None

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = (row.get("task") or "").strip()
            if not task or task.startswith("#"):
                continue

            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            layout = (row.get("layout") or "").strip()
            stack = str(int(float(row.get("stack") or 0)))

            key = (task, layout, stack)

            if current is None or current["key"] != key:
                current = {
                    "key": key,
                    "task": task,
                    "layout": layout,
                    "stack": stack,
                    "points": []
                }
                tasks.append(current)

            current["points"].append({
                "x": x, "y": y, "z": z,
                "action": (row.get("action") or "").strip()
            })

    return tasks

TASK_DB = load_tasks(CSV_PATH)
# for t in TASK_DB:
#     print(t)

# =========================
# 查表找對應任務
# =========================
def find_task(command, obj, layout, stack):
    global TASK_DB

    task_name = f"{command} {obj.split('_')[0]}"

    for t in TASK_DB:
        if t["task"] == task_name and t["layout"] == f"L{layout}" and t["stack"] == stack:
            return t

    return None


# =========================
# 將完整 pick and place 任務拆成兩段
# =========================
def split_pick_place_points(task):
    points = task["points"]

    split_idx = None
    for i in range(1, len(points)):
        if points[i - 1]["action"] == "M" and points[i]["action"] == "R":
            split_idx = i
            break

    if split_idx is None:
        raise RuntimeError("找不到 M -> R 的切分點，無法拆成 pick/place")

    pick_points = points[:split_idx - 1]
    place_points = points[split_idx - 1:]

    return pick_points, place_points


# =========================
# URScript 產生
# =========================
def pose(x, y, z):
    return f"p[{x},{y},{z},{RX},{RY},{RZ}]"


def build_script(points, task_id, segment_name, send_task_done=False, start_from_bot=True):
    lines = []
    lines.append("def run_task():")
    lines.append(f'  status_ok = socket_open("{PC_IP}", {CALLBACK_PORT}, "status_sock")')

    lines.append("  if status_ok:")
    lines.append(f'    socket_send_string("TASK_START|{task_id}", "status_sock")')
    lines.append('    socket_send_byte(10, "status_sock")')
    lines.append("  end")

    if start_from_bot:
        lines.append(f"  movel({pose(*BOT)}, a={ACC}, v={VEL})")

    prev_action = None

    for p in points:
        x, y, z = p["x"], p["y"], p["z"]
        action = p["action"]

        if prev_action and action and prev_action != action:
            if prev_action == "R" and action == "M":
                lines.append("  set_digital_out(4, True)")
                lines.append("  sleep(0.5)")
            elif prev_action == "M" and action == "R":
                lines.append("  set_digital_out(4, False)")
                lines.append("  sleep(0.5)")

        # 全部用 用 movel
        lines.append(f"  movel({pose(x,y,z)}, a={ACC}, v={VEL})")

        prev_action = action

    lines.append("  if status_ok:")
    lines.append(f'    socket_send_string("SEGMENT_DONE|{task_id}|{segment_name}", "status_sock")')
    lines.append('    socket_send_byte(10, "status_sock")')

    if send_task_done:
        lines.append(f'    socket_send_string("TASK_DONE|{task_id}", "status_sock")')
        lines.append('    socket_send_byte(10, "status_sock")')

    lines.append('    socket_close("status_sock")')
    lines.append("  end")

    lines.append("end")
    lines.append("run_task()")

    return "\n".join(lines)


# =========================
# callback server
# =========================
def run_robot_script_and_wait(script, task_id, expected_done_type, expected_segment=None):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cb:
        cb.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        cb.bind(("0.0.0.0", CALLBACK_PORT))
        cb.listen(1)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as robot:
            robot.connect((ROBOT_IP, ROBOT_PORT))
            robot.sendall((script + "\n").encode("utf-8"))

        print("Sent to robot, waiting callback...")

        c, addr = cb.accept()
        buffer = ""

        with c:
            print(f"UR robot callback connected from {addr}")

            while True:
                data = c.recv(1024)
                if not data:
                    break

                buffer += data.decode("utf-8", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    msg = line.strip()

                    if not msg:
                        continue

                    print("[ROBOT CALLBACK]", msg)

                    parts_msg = msg.split("|")

                    if parts_msg[0] == "TASK_START":
                        print(f"Robot task start: {msg}")

                    elif parts_msg[0] == expected_done_type:
                        done_task_id = int(parts_msg[1])

                        if done_task_id != task_id:
                            continue

                        if expected_done_type == "SEGMENT_DONE":
                            segment = parts_msg[2]
                            if segment == expected_segment:
                                print(f"Robot segment done: ID={task_id}, segment={segment}")
                                return True

                        elif expected_done_type == "TASK_DONE":
                            print(f"Robot task done: ID={task_id}")
                            return True

    return False


# =========================
# 等待 server 傳 kit_ready
# =========================
def parse_monitor_message(data):
    parts = {}
    for x in data.strip().split(";"):
        if not x:
            continue
        k, v = x.split("=", 1)
        parts[k] = v
    return parts

def wait_for_kit_ready(conn, task_id):
    conn.sendall(f"PICK_DONE|{task_id}\n".encode("utf-8"))
    print(f"Sent PICK_DONE to monitor: ID={task_id}")

    while True:
        data = conn.recv(1024).decode()
        if not data:
            return False

        # print("[RECV KIT STATUS]", data.strip())

        parts = parse_monitor_message(data)

        recv_task_id = int(parts.get("ID", task_id))
        if recv_task_id != task_id:
            continue

        kit_ready = parts.get("kit_ready", "False") in ("True", "true", "1")

        if kit_ready:
            print(f"Kit ready received: ID={task_id}")
            return True

        # print(f"Waiting kit ready: ID={task_id}")

# =========================
# 測試機械手臂連線
# =========================
HOME_JOINT_DEG = [-57.96, -66.24, -103.53, -100.68, 91.98, 31.99]
HOME_JOINT_RAD = [math.radians(a) for a in HOME_JOINT_DEG]
def test_robot_connection():
    """
    測試是否能連線到 UR 機械手臂
    成功後讓先回到原始姿態，接著關閉夾爪，再打開夾爪
    """
    home_joints = "[" + ",".join(f"{q:.6f}" for q in HOME_JOINT_RAD) + "]"

    test_script = "\n".join([
        "def test_connection():",
        f"  movej({home_joints}, a={ACC_RAD}, v={VEL_RAD})",
        "  sleep(0.5)",

        f"  movel({pose(*BOT)}, a={ACC}, v={VEL})",
        "  sleep(0.5)",

        "  set_digital_out(4, True)",
        "  sleep(1.0)",
        "  set_digital_out(4, False)",
        "  sleep(1.0)",
        "end",
        "test_connection()"
    ])

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as robot:
            robot.settimeout(5.0)
            robot.connect((ROBOT_IP, ROBOT_PORT))
            robot.sendall((test_script + "\n").encode("utf-8"))

        print("Robot connection test success: moved to home pose, gripper closed then opened.")
        return True

    except Exception as e:
        print(f"Robot connection test failed: {e}")
        return False
    

def handle_monitor_connection(conn):
    """
    處理監控主程式 client 傳來的 robot task
    同一條 TCP 連線會雙向傳送：
    - monitor -> UR server：robot task / kit_ready
    - UR server -> monitor：PICK_DONE / DONE
    """
    buffer = ""

    while True:
        data = conn.recv(1024).decode("utf-8", errors="ignore")
        if not data:
            print("Monitor client disconnected.")
            break

        buffer += data

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            msg = line.strip()

            if not msg:
                continue

            print("[RECV FROM MONITOR]", msg)

            # ===== 解析 =====
            parts = {}
            for x in msg.split(";"):
                if not x:
                    continue
                k, v = x.split("=", 1)
                parts[k] = v

            task_id = int(parts["ID"])
            command = parts["command"]
            obj = parts["object"]
            zone = parts["zone"]
            layout = parts["layout"]
            stack = parts["stack"]

            # ===== 查表 =====
            task = find_task(command, obj, layout, stack)
            if task is None:
                print("Task not found!")
                continue

            pick_points, place_points = split_pick_place_points(task)

            pick_script = build_script(
                pick_points,
                task_id,
                segment_name="PICK",
                send_task_done=False,
                start_from_bot=True
            )

            pick_ok = run_robot_script_and_wait(
                pick_script,
                task_id,
                expected_done_type="SEGMENT_DONE",
                expected_segment="PICK"
            )

            if not pick_ok:
                print(f"Pick segment failed or disconnected: ID={task_id}")
                continue

            # ===== 通知 monitor pick 完成，等待 kit_ready =====
            kit_ready = wait_for_kit_ready(conn, task_id)

            if not kit_ready:
                print(f"Kit ready wait failed: ID={task_id}")
                continue

            # ===== 第二段：place =====
            place_script = build_script(
                place_points,
                task_id,
                segment_name="PLACE",
                send_task_done=True,
                start_from_bot=False
            )

            place_ok = run_robot_script_and_wait(
                place_script,
                task_id,
                expected_done_type="TASK_DONE"
            )

            if not place_ok:
                print(f"Place segment failed or disconnected: ID={task_id}")
                continue

            # ===== 整個 pick + place 完成後，才通知監控主程式 DONE =====
            conn.sendall(f"DONE|{task_id}\n".encode("utf-8"))
            print(f"robot full task done: ID={task_id}")


# =========================
# 主程式
# =========================
def main():
    # 啟動主流程前，先測試機械手臂連線與 pin 4 夾爪控制
    if not test_robot_connection():
        print("Stop program because robot connection test failed.")
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((SERVER_HOST, SERVER_PORT))
        server.listen(1)

        print(f"[TCP SERVER] Listening on {SERVER_HOST}:{SERVER_PORT} ...")
        print("Waiting for monitor client...")

        conn, addr = server.accept()
        with conn:
            print(f"[TCP SERVER] Monitor client connected from {addr}")
            handle_monitor_connection(conn)


if __name__ == "__main__":
    main()
