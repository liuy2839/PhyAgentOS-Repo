"""逐相机 episode 录像 —— 接口② 的 videos 字段直接可用的前提。

与原 forge 流程的 `_core/video.py`（三路相机拼一张、落一个文件）不同：RSI
反思节点要按相机独立取关键帧，所以这里**每路相机各写一个 mp4**，路径进
结果 JSON 的 `videos`（键为相机短名：image/head → head）。

编码器沿用实测结论：服务器无 ffmpeg、无 imageio，cv2 的 `mp4v` 可写、
`avc1` 打不开 → 固定 mp4v，失败退 MJPG/avi。录像失败**绝不能影响回合**
—— 所有异常吞掉、只记一行日志、对应路径缺省。
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np


def camera_short_name(channel: str) -> str:
    """image/head → head。其它形态原样返回（不猜）。"""
    return channel.split("/", 1)[1] if "/" in channel else channel


class PerCameraRecorder:
    """按相机攒帧、回合结束时每路各落一个视频文件。"""

    def __init__(
        self,
        record_dir: str | None,
        fps: float = 30.0,
        log: Callable[[str], None] = print,
    ) -> None:
        self.record_dir = record_dir
        self.fps = float(fps)
        self._log = log
        self._writers: dict[str, tuple[Any, str]] = {}   # 短名 → (VideoWriter, path)
        self.enabled = False
        if record_dir:
            try:
                os.makedirs(record_dir, exist_ok=True)
                import cv2  # noqa: F401

                self.enabled = True
            except Exception as exc:  # noqa: BLE001
                self._log(f"录像不可用，已关闭：{type(exc).__name__}: {exc}")

    def add(self, frames: dict[str, np.ndarray]) -> None:
        """每帧调用。writer 首帧懒建（需要 h,w）。单路失败只丢该路。"""
        if not self.enabled:
            return
        for channel, frame in frames.items():
            name = camera_short_name(channel)
            try:
                import cv2

                arr = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
                if arr.ndim != 3 or arr.shape[2] != 3:
                    raise ValueError(f"{channel} 不是 HxWx3")
                entry = self._writers.get(name)
                if entry is None:
                    h, w = arr.shape[:2]
                    path = os.path.join(self.record_dir, f"{name}.mp4")
                    writer = cv2.VideoWriter(
                        path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
                    )
                    if not writer.isOpened():
                        # mp4v 打不开（缺编码器）→ 退 MJPG/avi
                        path = os.path.join(self.record_dir, f"{name}.avi")
                        writer = cv2.VideoWriter(
                            path, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h)
                        )
                        if not writer.isOpened():
                            writer.release()
                            raise RuntimeError("mp4v 与 MJPG 都打不开")
                    self._writers[name] = (writer, path)
                    entry = (writer, path)
                entry[0].write(arr[:, :, ::-1])          # RGB → BGR
            except Exception as exc:  # noqa: BLE001
                self._log(f"录帧失败（{name}，忽略）：{type(exc).__name__}: {exc}")

    def save(self) -> dict[str, str]:
        """收掉所有 writer，返回 {相机短名: 路径}。只收有帧的。"""
        out: dict[str, str] = {}
        for name, (writer, path) in self._writers.items():
            try:
                writer.release()
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    out[name] = path
                else:
                    self._log(f"视频没写出来（{name}）：{path}")
            except Exception as exc:  # noqa: BLE001
                self._log(f"收视频失败（{name}，忽略）：{type(exc).__name__}: {exc}")
        self._writers = {}
        return out

    def close(self) -> None:
        """异常路径兜底：writer 不收会留下打不开的文件句柄。"""
        for name, (writer, _path) in self._writers.items():
            try:
                writer.release()
            except Exception:  # noqa: BLE001
                pass
        self._writers = {}
