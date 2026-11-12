"""π0.5 WS 链路端到端冒烟（zy2d）：拉服务器 → 握手 → 假观测推理一次。

在 zy2d 上运行：/data1/liuyi/pi05_openpi_venv/bin/python /data1/liuyi/pi05_ws_smoke/smoke.py
"""
import os
import subprocess
import sys
import time

import numpy as np

XPL = "/data1/liuyi/XPolicyLab"
OPENPI_SRC = "/data2/kangning/fastwam_eval/RoboDojo/XPolicyLab/policy/Pi_05/openpi/src"
MODEL_DIR = "/data2/kangning/models/robotwin_pi05/RoboTwin-lerobot_v30-aloha_agilex-joint-0"
PORT = 6010

env = dict(os.environ)
env["PYTHONPATH"] = f"{XPL}:{OPENPI_SRC}"
# 共享卡：限 jax 预留比例（0.15 ≈ 11GB on 72G 卡，够 π0.5 推理），
# 并绑到当前最空的卡（跑之前用 nvidia-smi 确认）。
env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get("PI05_MEM_FRACTION", "0.15")
env["CUDA_VISIBLE_DEVICES"] = os.environ.get("PI05_GPU", "2")
env["XDG_CACHE_HOME"] = "/tmp/pi05_cache"

cmd = [
    sys.executable, "-u", f"{XPL}/setup_policy_server.py",
    "--config_path", f"{XPL}/policy/Pi_05/deploy.yml",
    "--overrides",
    "policy_name=Pi_05",
    f"model_path={MODEL_DIR}",
    "checkpoint_num=29999",
    "train_config_name=pi05_base_aloha_full_sim_arx-x5_seed_0",
    "repo_id=RoboTwin_lerobot_v30",
    "action_type=joint",
    "env_cfg_type=arx_x5",
    f"port={PORT}",
    "host=127.0.0.1",
]
print("SERVER CMD:", " ".join(cmd), flush=True)
logf = open("/data1/liuyi/pi05_ws_smoke/server.log", "w")
proc = subprocess.Popen(cmd, cwd=XPL, env=env, stdout=logf,
                        stderr=subprocess.STDOUT, start_new_session=True)
print("SERVER PID:", proc.pid, flush=True)

deadline = time.monotonic() + 900
ready = False
while time.monotonic() < deadline:
    if proc.poll() is not None:
        print("SERVER DIED code=", proc.returncode, flush=True)
        os.system("tail -30 /data1/liuyi/pi05_ws_smoke/server.log")
        sys.exit(1)
    try:
        sys.path.insert(0, XPL)
        from XPolicyLab.client_server.ws.model_client import WsModelClient

        client = WsModelClient(url=f"ws://127.0.0.1:{PORT}", evaluation_id="smoke",
                               trial_id="smoke", action_case_id="smoke_case",
                               connect_timeout_s=3, max_connect_attempts=1)
        client.call("reset")
        client.close()
        ready = True
        print("SERVER READY", flush=True)
        break
    except Exception as exc:
        print("waiting:", str(exc)[:80], flush=True)
        time.sleep(5)
if not ready:
    print("TIMEOUT", flush=True)
    sys.exit(1)

# 假观测推理一次（XPolicyLab 标准观测格式，arx_x5 joint）
obs = {
    "instruction": "smoke test",
    "vision": {c: {"color": np.zeros((224, 224, 3), dtype=np.uint8)}
               for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
    "state": {
        "left_arm_joint_state": np.zeros(6, dtype=np.float32),
        "left_ee_joint_state": np.zeros(1, dtype=np.float32),
        "left_ee_pose": np.array([0.4, 0, 0.8, 0, 0, 0, 1], dtype=np.float32),
        "right_arm_joint_state": np.zeros(6, dtype=np.float32),
        "right_ee_joint_state": np.zeros(1, dtype=np.float32),
        "right_ee_pose": np.array([0.4, 0, 0.8, 0, 0, 0, 1], dtype=np.float32),
    },
}
client = WsModelClient(url=f"ws://127.0.0.1:{PORT}", evaluation_id="smoke",
                       trial_id="smoke", action_case_id="smoke_case")
client.call("reset")
t0 = time.time()
client.call("update_obs", obs)
actions = client.call("get_action")
print(f"INFERENCE OK: {len(actions)} actions in {time.time() - t0:.1f}s", flush=True)
a0 = actions[0] if isinstance(actions, list) else actions
print("ACTION0 type:", type(a0).__name__,
      "keys:", sorted(a0.keys()) if isinstance(a0, dict) else np.asarray(a0).shape,
      flush=True)
client.close()
proc.terminate()
try:
    proc.wait(timeout=15)
except Exception:
    proc.kill()
print("SMOKE PASS", flush=True)
