"""
infer_service/task.py - 推理任务模型
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict


class TaskStatus:
    CREATED = 'CREATED'
    RUNNING = 'RUNNING'
    STOPPED = 'STOPPED'
    ERROR = 'ERROR'


@dataclass
class StreamSpec:
    """视频流配置"""
    url: str                       # rtsp://... / 摄像头ID / 文件路径
    protocol: str = 'rtsp'         # rtsp / rtmp / http-flv / camera / file
    username: str | None = None
    password: str | None = None
    fps: float | None = None       # 源流帧率（video.streams[].fps）；None=回退模型 train_fps

    @property
    def source(self) -> str:
        """带鉴权的实际拉流地址"""
        if self.username and self.password and '://' in self.url:
            scheme, rest = self.url.split('://', 1)
            return f'{scheme}://{self.username}:{self.password}@{rest}'
        return self.url


@dataclass
class TaskParams:
    """运行期可调参数（变更需重启推理服务生效）

    infer_fps / push_fps 缺省为 None = 由 service.yaml 决定：
      - push_fps ← task_defaults.push_fps
      - infer_fps ← video.streams[].infer_fps，未配置时回落 task_defaults.infer_fps
    避免代码里再写一份与配置漂移的硬编码默认值。
    """
    infer_fps: float | None = None
    push_fps: int | None = None
    reconnect_max_attempts: int = 8


@dataclass
class Task:
    id: str
    session: str
    stream: StreamSpec
    checkpoint: str
    params: TaskParams = field(default_factory=TaskParams)
    status: str = TaskStatus.CREATED
    # 契约扩展（差异 #3/#4/#5）：工位 / 批次 / ROI / 帧率校验上下文
    station_no: str | None = None
    batch_no: str | None = None
    roi: list | None = None                          # 多边形顶点或矩形 [x1,y1,x2,y2]
    infer_fps: float | None = None                   # 用于模型包帧率校验（#8）
    config_label_ids: set | None = None              # 配置 label_map 的 id 集合（#2 校验）
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    stopped_at: float | None = None
    error: str | None = None

    def to_dict(self):
        d = asdict(self)
        d.pop('config_label_ids', None)   # set 不可序列化，剔除
        d['uptime_seconds'] = (
            round((self.stopped_at or time.time()) - self.started_at, 1)
            if self.started_at else 0.0)
        return d
