import socket
import time
import math

# =========================
# 使用者設定
# =========================
# ROBOT_IP = "192.168.50.204" # 930真實機械手臂ip
ROBOT_IP = "192.168.0.231" # 虛擬機 ip
ROBOT_PORT = 30002          # UR Secondary Interface
CALLBACK_PORT = 50001       # Python 端接收 START / DONE 訊號

# 固定姿態（請依你的 TCP 實際姿態修改）
RX = 0.0
RY = 3.14159
RZ = 0.0

# 運動參數 (moveL)
ACC = 5.0    # 加速度 m/s²
VEL = 1.0   # 速度 m/s

# 運動參數 (moveJ)
ACC_DEG = 286.0   # 286度/s^2
VEL_DEG = 180.0    # 180度/s
ACC_RAD = math.radians(ACC_DEG)   # 轉成 rad/s^2
VEL_RAD = math.radians(VEL_DEG)   # 轉成 rad/s

# =========================
# 自動取得本機 IP
# 讓 UR 控制器可以回傳 START / DONE
# =========================
def get_local_ip(target_ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target_ip, 1))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

PC_IP = get_local_ip(ROBOT_IP)

# =========================
# 產生 URScript
# =========================
def build_urscript(pc_ip, callback_port):
    script = f"""
    def run_path():
    status_ok = socket_open("{pc_ip}", {callback_port}, "status_sock")

    # 如果 socket 連線成功，機械手臂就送出START\n
    if status_ok:
        socket_send_string("START", "status_sock")
        socket_send_byte(10, "status_sock")
    end

    # 1
    movej(p[0.0, -0.3, 0.15, {RX}, {RY}, {RZ}], a={ACC_RAD}, v={VEL_RAD})

    # 2
    movej(p[0.325, -0.025, 0.15, {RX}, {RY}, {RZ}], a={ACC_RAD}, v={VEL_RAD})

    # 3 (stack 1)
    # movel(p[0.325, -0.025, 0.094, {RX}, {RY}, {RZ}], a={ACC}, v={VEL})

    # 3 (stack 2)
    # movel(p[0.325, -0.025, 0.080, {RX}, {RY}, {RZ}], a={ACC}, v={VEL})

    # 3 (stack 3)
    # movel(p[0.325, -0.025, 0.066, {RX}, {RY}, {RZ}], a={ACC}, v={VEL})

    # 3 (stack 4)
    # movel(p[0.325, -0.025, 0.052, {RX}, {RY}, {RZ}], a={ACC}, v={VEL})

    # 3 (stack 5)
    # movel(p[0.325, -0.025, 0.038, {RX}, {RY}, {RZ}], a={ACC}, v={VEL})

    # 3 (stack 6)
    movel(p[0.325, -0.025, 0.022, {RX}, {RY}, {RZ}], a={ACC}, v={VEL})

    # pin 4 開 停一下
    set_digital_out(4, True)
    sleep(0.5)

    # 4
    movej(p[0.325, -0.025, 0.15, {RX}, {RY}, {RZ}], a={ACC_RAD}, v={VEL_RAD})

    # 5
    movej(p[-0.05, -0.29, 0.15, {RX}, {RY}, {RZ}], a={ACC_RAD}, v={VEL_RAD})

    # 6
    movel(p[-0.05, -0.29, 0.040, {RX}, {RY}, {RZ}], a={ACC}, v={VEL})

    # pin 4 關
    set_digital_out(4, False)

    # 7
    movel(p[-0.05, -0.29, 0.15, {RX}, {RY}, {RZ}], a={ACC}, v={VEL})

    # 8
    movej(p[0.0, -0.3, 0.15, {RX}, {RY}, {RZ}], a={ACC_RAD}, v={VEL_RAD})

    if status_ok:
        # 機械手臂送出：DONE\n
        socket_send_string("DONE", "status_sock")
        socket_send_byte(10, "status_sock")
        socket_close("status_sock")
    end
    end

    run_path()
    """
    return script

# =========================
# 主程式
# =========================
def main():
    print(f"本機 IP: {PC_IP}")
    print(f"等待 UR 回傳 START / DONE，監聽 Port: {CALLBACK_PORT}")

    # 先建立 server，等 URScript 回傳狀態
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("0.0.0.0", CALLBACK_PORT))
        server_sock.listen(1)

        urscript = build_urscript(PC_IP, CALLBACK_PORT)

        # 送 URScript 到機器人
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as robot_sock:
            robot_sock.connect((ROBOT_IP, ROBOT_PORT))
            robot_sock.sendall(urscript.encode("utf-8"))

        print("URScript 已送出，等待機器人執行...")

        conn, addr = server_sock.accept()
        with conn:
            print(f"已連線自 UR 控制器: {addr}")
            buffer = ""
            start_time = None

            while True:
                data = conn.recv(1024)
                if not data:
                    break

                buffer += data.decode("utf-8", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    msg = line.strip()

                    if msg == "START":
                        start_time = time.perf_counter()
                        print("開始計時")
                    elif msg == "DONE":
                        if start_time is not None:
                            elapsed = time.perf_counter() - start_time
                            print(f"執行完成，總時間: {elapsed:.3f} 秒")
                        else:
                            print("收到 DONE，但未收到 START")
                        return

    print("程式結束")

if __name__ == "__main__":
    main()