"""
infer_service/engine/pipeline.py - 单路推理管线（三线程编排）

线程模型（关键：任一环节变慢都不会拖垮其他环节）:
    拉流线程  → frame_queue(全速显示帧,有界) → 推送线程(全速叠加最新结果→JPEG→WS)
      StreamReader
    拉流线程  → infer_queue(模型帧,有界) → 推理线程(按 infer_fps 抽帧) → 更新 self._latest_result
      StreamReader                          StreamDetector

- 显示与模型解耦：显示帧走全速(源帧率)不抽帧→预览流畅；模型仅吃 infer_fps 帧→与训练一致
- 推送线程叠加"最新模型结果"到全速帧：动作标签/时长由模型实时更新；骨架取最近一次结果(≤80ms 偏差可忽略)
- 队列有界且满时丢弃最旧帧 → 推理始终跟随实时流，不堆积历史帧
- JPEG 编码与网络发送在推送线程 → 不影响 MediaPipe/模型推理
- 每路任务独立实例，状态互不干扰
"""

from __future__ import annotations
import logging
from pathlib import Path
import hashlib
import threading
import json
import time
from collections import Counter, deque
from datetime import datetime
from queue import Empty, Queue

import numpy as np
import cv2

from infer_service.engine.annotate import draw_overlay, encode_jpeg, build_label_colors
from infer_service.engine.eventer import ActionSegmenter
from infer_service.engine.continuity import ContinuousEvidenceTimer
from common.frame_timing import TimedFrame
from infer_service.engine.stream import StreamReader, StreamStatus
from infer_service.rules import Rules
from infer_service.task import TaskStatus


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


logger = logging.getLogger('infer_service')


# 推理线程连续异常上限：超过后任务置 ERROR 并停止推理（避免"假运行"）
MAX_CONSECUTIVE_INFER_ERRORS = 10

# 视频质量检测阈值（docs/09 事件表 video_quality；仅启用开关可配，阈值取代码默认）
VQ_BLACK_MEAN_THRESHOLD = 12.0    # 灰度均值 < 此值 → 黑屏（纯黑/纯灰）
VQ_OCCLUSION_LAP_VAR = 25.0       # Laplacian 方差 < 此值 → 低纹理/遮挡（画面失去细节）
VQ_FREEZE_FRAMES = 50             # 连续 N 帧像素哈希一致 → 冻结（@~25fps ≈ 2s）
VQ_CORRUPTION_SAT_RATIO = 0.5     # 极端像素(纯黑/纯白)比例 > 此值 → 花屏/噪点


class InferencePipeline:
    """单路任务推理管线"""

    def __init__(self, task, model_pool, service_cfg, rules_cfg=None,
                 station_no: str = None, batch_no: str = None,
                 pre_rules=None, post_rules=None, enabled_actions=None,
                 object_model_dir: str = None, object_model_cfg=None,
                 object_sample_rate: int = 2,
                 object_report_interval: float = 0.5,
                 count_stop_on: str = 'WATER',
                 object_target_class: str = 'hose',
                 object_roi_points: list = None,
                 action_label_map: dict = None,
                 on_auto_end=None):
        self.task = task
        self.model_pool = model_pool
        self.cfg = service_cfg
        self.td = getattr(service_cfg, 'task_defaults', None)

        # 动作颜色映射：统一取自 service.yaml action_model.label_map.color
        self._label_colors = build_label_colors(action_label_map)

        self.frame_queue = Queue(maxsize=8)    # 全速显示帧（源帧率）
        self.infer_queue = Queue(maxsize=8)    # 模型输入帧（infer_fps）
        self._result_lock = threading.Lock()
        self._latest_result = None             # 最新模型结果（推送线程叠加用）

        # 规则来源：已构建的 Rules 实例 / service.yaml 契约命名 / 旧 service.yaml 兜底
        if isinstance(rules_cfg, Rules):
            self.rules = rules_cfg
        elif pre_rules is not None or post_rules is not None:
            # 后处理参数统一来自 post_rules（task_defaults 同义项已合并）
            self.rules = Rules(
                pre_rules=pre_rules, post_rules=post_rules,
                enabled_actions=enabled_actions,
                merge_tolerance=getattr(post_rules, 'merge_tolerance', 10))
        else:
            self.rules = Rules(rules_cfg or getattr(service_cfg, 'rules', None),
                               self.td)

        self.station_no = station_no or getattr(task, 'session', None)
        self.batch_no = batch_no

        self._loaded = None
        self._model_pool_acquired = False
        self._detector = None
        self._segmenter = None
        po = getattr(service_cfg, 'post_rules', None)
        self._cleaning_timer = ContinuousEvidenceTimer(
            getattr(po, 'cleaning_seconds', 3.), getattr(po, 'max_observation_gap_seconds', .12))
        self._reader = None
        self._infer_thread = None
        self._push_thread = None
        self._stop_event = threading.Event()
        self._detector_lock = threading.Lock()   # 保护检测器初始化/替换

        # 推理侧主动结束批次（docs/09 §二 / 07 §七）：watchdog 检测无动作超时/断流，
        # 命中后经 on_auto_end 回调（由 manager 注入，确保清理 station 映射）结束批次
        self._on_auto_end = on_auto_end
        self._no_action_secs = 0.0           # 累计"无清洗动作"时长（断流期间不累计）
        self._offline_start_ts = None        # 断流(OFFLINE)起始时刻；恢复为 None
        self._watchdog_thread = None
        self._scan_timeout = float(getattr(self.td, 'scan_action_timeout', 300) or 300)
        self._offline_timeout = float(getattr(self.td, 'stream_offline_timeout', 300) or 300)
        # 业务重连超时（task_defaults.biz_lost_timeout，docs/06 §七 / 09 §二）：
        # 批次运行中事件 WS 无订阅者（业务断连）持续超此值 → 清空重放缓存（数据丢弃）+ BIZ_LOST 结束
        self._biz_lost_timeout = float(getattr(self.td, 'biz_lost_timeout', 60) or 0)
        self._biz_lost_start_ts = None       # 事件通道无订阅者的起始时刻；重连即清零

        # 视频质量检测（docs/09 事件表 video_quality）：默认关闭，由 task_defaults.video_quality_enabled 控制
        self._vq_enabled = bool(getattr(self.td, 'video_quality_enabled', False))
        self._vq_last_hash = None
        self._vq_freeze_count = 0
        self._vq_last_types = []

        self._sinks_lock = threading.Lock()
        self._frame_sinks = set()
        self._event_sinks = set()

        self._alerts = deque(maxlen=50)
        # 事件环形缓存（docs/06 §6.2.10 / docs/09 §五）：业务判定事件按序缓存，
        # 供 WS 断线重连后**全量重放**（按序发送，事件自带 batch_no，业务按批次分发幂等消化）；
        # 开关与容量可配（ws.event_cache_enabled/size），关闭时为 None——重放不可用
        _ws = getattr(service_cfg, 'ws', None)
        if _ws is None or _ws.event_cache_enabled:
            self._event_cache = deque(maxlen=max(1, int(getattr(_ws, 'event_cache_size', 512))))
        else:
            self._event_cache = None
        self._gap_pending = False
        self._stream_status = None
        self._stream_extra = {}
        self._fps_window = deque(maxlen=120)
        self._last_elapsed_ms = 0.0
        self._infer_frames = 0

        # ---- 推理诊断日志（联调/排障：动作变化即时输出 + 周期汇总，INFO 级节流）----
        self._log = logging.getLogger('infer_service')
        self._diag_interval = 5.0           # 周期汇总间隔（秒）
        self._diag_last_ts = time.time()
        self._diag_skipped = 0              # 本周期内滑窗未填满被跳过的帧数
        self._diag_rule_hits = Counter()    # 本周期内前置规则命中计数（L1/L2/L3）
        self._diag_last_code = None         # 最近一次动作编码（变化时输出）
        self._diag_first_valid = False      # 是否已产出首个有效模型判定

        # 目标检测（差异 #9）：仅在数量检测窗口内检测 hose 原始数量；
        # 窗口关闭触发动作可配（object_model.count_stop_on，默认 WATER=首个水洗动作出现）
        self.object_model_dir = object_model_dir
        self.object_model_cfg = object_model_cfg
        self.object_sample_rate = max(1, int(object_sample_rate))
        self.object_report_interval = float(object_report_interval)
        self.count_stop_on = count_stop_on
        self.object_target_class = object_target_class
        # 目标检测独立 ROI（与动作检测 video.streams[].roi_points 互不影响）
        self.object_roi_bbox = self._compute_roi_bbox(object_roi_points)
        self._object_detector = None
        self._object_frame_counter = 0
        self._count_window_active = bool(object_model_dir and object_model_cfg)
        self._latest_object_dets = []
        self._last_hose_count = -1
        self._last_count_ts = 0.0

    # ================= 生命周期 =================

    def _initialize_action_runtime(self):
        local_cfg = getattr(self.cfg, 'local_action', None)
        if local_cfg and local_cfg.enabled:
            from infer_service.engine.raw_hand_tool import RawHandToolDetector
            manifest = Path(local_cfg.manifest)
            if not manifest.is_absolute():
                manifest = Path(__file__).resolve().parents[2]/manifest
            self._detector = RawHandToolDetector(manifest, local_cfg.device,
                                float(self._cleaning_timer.max_gap))
            self._loaded = self._detector.loaded
            self.rules.bind_label_map(self._loaded.label_map)
            self._label_colors = build_label_colors(self._loaded.label_map)
        else:
            from infer_service.engine.detector import StreamDetector
            self._loaded = self.model_pool.get(
                self.task.checkpoint,
                infer_fps=getattr(self.task, 'infer_fps', None),
                config_label_ids=getattr(self.task, 'config_label_ids', None))
            self.model_pool.acquire(self.task.checkpoint)
            self._model_pool_acquired = True
            # 动作开关：解析 enabled_actions 名称 → id（依赖模型 label_map）
            self.rules.bind_label_map(self._loaded.label_map)
            # ROI（差异 #11）：多边形/矩形
            if getattr(self, 'roi', None):
                self.rules.set_roi(self.roi)

            hand_provider = None
            hand_cfg = getattr(self.cfg, 'hand_evidence', None)
            if hand_cfg and hand_cfg.enabled:
                from common.glove_evidence import GloveHandEvidence
                manifest = Path(hand_cfg.manifest)
                if not manifest.is_absolute():
                    manifest = Path(__file__).resolve().parents[2]/manifest
                # Explicitly configured model failure aborts startup rather than silently claiming fusion.
                hand_provider = GloveHandEvidence(manifest, hand_cfg.device,
                                hand_cfg.score_threshold, hand_cfg.roi_normalized)
            self._detector = StreamDetector(
                self._loaded, self.rules, self._loaded.mode,
                enable_duration_feat=(self.td.enable_duration_feat
                                     if self.td else True),
                enable_visibility_mask=(self.td.enable_visibility_mask
                                        if self.td else True),
                stability_motion_threshold=(self.td.aug_stability_threshold
                                            if self.td else 0.3),
                device=self.model_pool.device, hand_provider=hand_provider,
                max_observation_gap_seconds=self._cleaning_timer.max_gap)

        # 模型输入帧率 = 训练帧率 train_fps / sample_rate（=推理 infer_fps），段时长换算按此口径
        fps = float(self._loaded.train_fps) / max(1, self._loaded.sample_rate)
        self._segmenter = ActionSegmenter(
            station_no=self.station_no, batch_no=self.batch_no, fps=fps,
            merge_tolerance=self.rules.merge_tolerance,
            min_action_duration=self.rules.min_action_duration,
            sample_rate=self._loaded.sample_rate,
            merge_tolerance_frames=self.rules.merge_tolerance_frames,
            max_observation_gap_seconds=float(self._cleaning_timer.max_gap))
        self._segmenter.set_batch(self.station_no, self.batch_no)


    def start_offline(self):
        """Use the same inference loop for ordered local-file observations.

        The caller supplies timestamped raw frames and local sinks. No stream,
        watchdog or external transport is launched by this method.
        """
        if self.task.status == TaskStatus.RUNNING:
            return
        self._initialize_action_runtime()
        self._stop_event.clear()
        self.task.status = TaskStatus.RUNNING
        self.task.started_at = time.time()
        self._infer_thread = threading.Thread(target=self._infer_loop,
            name=f'infer-{self.task.id}', daemon=True)
        self._infer_thread.start()
        self._emit_event(dict(type='started', task_id=self.task.id,
                             backend=self._loaded.mode, local_file_mode=True, ts=_now()))

    def process_offline(self, packet, timeout_seconds=60):
        """Submit to the actual worker loop and await this exact source PTS."""
        if not isinstance(packet, TimedFrame) or packet.source_time is None:
            raise ValueError('offline pipeline requires TimedFrame with source PTS')
        if self.task.status != TaskStatus.RUNNING or not self.running:
            raise RuntimeError('offline pipeline is not running')
        with self._result_lock:
            old = self._latest_result
        if old is not None and old.epoch == packet.epoch and old.source_pts == packet.source_pts:
            raise ValueError('duplicate offline source PTS')
        self.infer_queue.put(packet, timeout=timeout_seconds)
        deadline = time.monotonic()+timeout_seconds
        while time.monotonic() < deadline:
            with self._result_lock:
                result = self._latest_result
            if result is not None and result.source_pts == packet.source_pts and result.epoch == packet.epoch:
                return result
            if self.task.status == TaskStatus.ERROR or not self._infer_thread.is_alive():
                raise RuntimeError(self.task.error or 'pipeline worker stopped')
            time.sleep(.002)
        raise TimeoutError('pipeline did not produce the requested source PTS; reconcile before restart')

    def start(self):
        if self.task.status == TaskStatus.RUNNING:
            return

        self._initialize_action_runtime()
        fps = float(self._loaded.train_fps)/max(1, self._loaded.sample_rate)

        # 目标检测（差异 #9）：加载 ONNX 模型包（模型类参数/类目来自 model_info.json）
        if self.object_model_dir and self.object_model_cfg:
            # engine 校验：当前仅实现 onnxruntime；配置其他引擎时明确告警并跳过加载，
            # 避免"配了引擎却静默按 ONNX 加载"的误解
            engine = str(getattr(self.object_model_cfg, 'engine', 'onnxruntime')
                         or 'onnxruntime').lower()
            engine_ok = (engine == 'onnxruntime')
            try:
                from infer_service.engine.object_detector import (
                    YoloONNXDetector, ObjectModelError)
                if not engine_ok:
                    raise ObjectModelError(
                        f'不支持的 engine: {engine}（当前仅实现 onnxruntime）')
                om_cfg = self.object_model_cfg
                self._object_detector = YoloONNXDetector(
                    self.object_model_dir,
                    conf_threshold=getattr(om_cfg, 'conf_threshold', 0.5),
                    nms_threshold=getattr(om_cfg, 'nms_threshold', 0.45),
                    class_names=self._object_class_names(om_cfg),
                    input_size=tuple(getattr(om_cfg, 'input_size', [640, 640])),
                    device=self.model_pool.device,
                    target_class=self.object_target_class)
                self._count_window_active = True
                self._emit_event({'type': 'object_model_loaded',
                                  'station_no': self.station_no,
                                  'batch_no': self.batch_no,
                                  'dir': self.object_model_dir, 'ts': _now()})
            except (ObjectModelError, Exception) as e:   # noqa: BLE001
                self.add_alert(f'目标检测模型加载失败: {e}', 'ERROR')
                self._object_detector = None
                self._count_window_active = False

        # 显示全速(源帧率)，模型按 infer_fps(=源/sample_rate) 抽帧 → 与训练一致
        # 优先级：任务参数(Java下发) > task_defaults.infer_fps > 训练帧率 train_fps
        # 显示限速优先取 video.streams[].fps（摄像头真实帧率），缺失时回退模型 train_fps
        src_fps = float(getattr(self.task.stream, 'fps', None)
                        or self._loaded.train_fps)
        model_fps = fps
        requested_fps = self.task.params.infer_fps or (self.td.infer_fps if self.td else fps)
        if abs(float(requested_fps)-fps) > .01:
            self._log.warning('Configured inference FPS %.3f differs from checkpoint %.3f; using checkpoint cadence', requested_fps, fps)
        self._reader = StreamReader(
            self.task.stream.source, self.frame_queue, self.infer_queue,
            display_fps=src_fps,
            infer_fps=model_fps,
            reconnect_max_attempts=self.task.params.reconnect_max_attempts,
            reconnect_initial_delay=(self.td.reconnect_initial_delay
                                    if self.td else 2.0),
            on_status=self._on_stream_status,
            on_gap=self._on_gap,
            name=f'stream-{self.task.id}')

        # 模型/管线自检日志（排障：帧率口径、标签契约、推理设备一眼可查）
        _lm_ids = [v.get('id') if isinstance(v, dict) else v
                   for v in (self._loaded.label_map or {}).values()]
        self._log.info(
            '[%s/%s] 管线启动: mode=%s seq_len=%d aug_v=%s sample_rate=%d '
            'labels=%s | 显示帧率=%.1ffps 推理帧率=%.1ffps device=%s '
            'mp_gpu=%s '
            '(推理帧率须=train_fps/sample_rate，与训练一致)',
            self.station_no, self.batch_no,
            self._loaded.mode, self._loaded.seq_len, self._loaded.aug_version,
            self._loaded.sample_rate, _lm_ids,
            src_fps, model_fps, self.model_pool.device,
            'ON' if getattr(self._detector.extractor, '_gpu_enabled', False) else 'OFF')

        self._stop_event.clear()
        self._reader.start()
        self._infer_thread = threading.Thread(
            target=self._infer_loop, name=f'infer-{self.task.id}', daemon=True)
        self._push_thread = threading.Thread(
            target=self._push_loop, name=f'push-{self.task.id}', daemon=True)
        self._infer_thread.start()
        self._push_thread.start()

        # 推理侧主动结束批次监护线程（无动作超时 / 断流超时检测）
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name=f'watchdog-{self.task.id}', daemon=True)
        self._watchdog_thread.start()

        self.task.status = TaskStatus.RUNNING
        self.task.started_at = time.time()
        self._emit_event({'type': 'started', 'session': self.task.session,
                          'station_no': self.station_no, 'batch_no': self.batch_no,
                          'task_id': self.task.id, 'ts': _now()})
        # 批次开始（业务侧据此开启 record_batch / 订阅 WS 事件）
        self._emit_event({
            'type': 'batch_started',
            'station_no': self.station_no,
            'batch_no': self.batch_no,
            'task_id': self.task.id,
            'ts': _now(),
        })

    def stop(self, reason: str = None):
        if self.task.status == TaskStatus.STOPPED:
            return

        self._log.info('[%s] 任务停止: task_id=%s reason=%s',
                       self.station_no, self.task.id, reason)
        self._stop_event.set()
        if self._reader:
            self._reader.stop()
        for t in (self._infer_thread, self._push_thread):
            if t and t.is_alive():
                t.join(timeout=3.0)

        if self._segmenter:
            for ev in self._segmenter.flush():
                self._emit_event(ev)
        if self._loaded and self._model_pool_acquired:
            self.model_pool.release(self._loaded.path)
            self._model_pool_acquired = False

        self.task.status = TaskStatus.STOPPED
        self.task.stopped_at = time.time()
        # 批次结束（业务侧据此落 batch 状态 / 触发数量/合格判定）
        if self._segmenter:
            self._log.info('[%s] 批次结束: batch_no=%s reason=%s 动作段数=%d',
                           self.station_no, self.batch_no, reason or 'NORMAL',
                           len(self._segmenter.segments))
            self._emit_event(self._segmenter.end_batch(reason or 'NORMAL'))
        self._emit_event({'type': 'stopped', 'session': self.task.session,
                          'station_no': self.station_no, 'batch_no': self.batch_no,
                          'task_id': self.task.id, 'reason': reason, 'ts': _now()})

    # ================= 推理侧主动结束批次（docs/09 §二 / 07 §七）=================

    def _auto_end(self, reason: str):
        """命中超时/断流后结束批次：经 on_auto_end 回调（manager）确保清理 station 映射。

        兜底：未注入回调时直接停自身（不清理 manager 映射，仅保底避免任务假运行）。
        """
        if not self.running:
            return
        cb = self._on_auto_end
        if cb is not None:
            try:
                cb(reason)
                return
            except Exception as e:  # noqa: BLE001
                self.add_alert(f'主动结束回调异常({reason}): {e}', 'ERROR')
        self.stop(reason=reason)

    def _watchdog_loop(self):
        """周期自检：无清洗动作超时 → SCAN_TIMEOUT；断流超时 → INTERRUPTED；
        业务断连超时（事件通道无订阅者）→ 清空重放缓存 + BIZ_LOST。

        - 仅在视频流非 OFFLINE 时累计"无动作时长"，断流期间不计时（断流是数据缺失，
          非工人停滞）；断流 OFFLINE 起始时刻由 _on_stream_status 记录，恢复即清零；
        - 业务断连仅在有活动批次时计时（无批次时业务来不来都不影响数据归属），
          事件通道重连（出现订阅者）即重置；超时先清空事件重放缓存（数据丢弃）再结束；
        - 命中后调 _auto_end（经 manager.stop_station 清理映射并推送 batch_ended）。
        """
        tick = 2.0
        last = time.time()
        while self.running:
            time.sleep(tick)
            if not self.running:
                break
            now = time.time()
            dt = now - last
            last = now

            # 1) 无清洗动作超时（断流期间不累计）
            if self._stream_status != StreamStatus.OFFLINE:
                self._no_action_secs += dt
            if (self._scan_timeout > 0
                    and self._no_action_secs >= self._scan_timeout):
                self.add_alert(
                    f'批次无清洗动作超时({int(self._no_action_secs)}s)，推理侧主动结束',
                    'WARNING')
                self._auto_end('SCAN_TIMEOUT')
                return

            # 2) 断流超时：OFFLINE 持续超阈值
            if (self._offline_timeout > 0
                    and self._offline_start_ts is not None
                    and now - self._offline_start_ts >= self._offline_timeout):
                self.add_alert(
                    f'视频流断流超时({int(now - self._offline_start_ts)}s)，推理侧主动结束',
                    'WARNING')
                self._auto_end('INTERRUPTED')
                return

            # 3) 业务断连超时：批次运行中事件 WS 无订阅者持续超阈值（biz_lost_timeout，0=关闭）
            with self._sinks_lock:
                has_event_sink = bool(self._event_sinks)
            if not self.batch_no or self._biz_lost_timeout <= 0 or has_event_sink:
                self._biz_lost_start_ts = None
            else:
                if self._biz_lost_start_ts is None:
                    self._biz_lost_start_ts = now
                elif now - self._biz_lost_start_ts >= self._biz_lost_timeout:
                    self.add_alert(
                        f'业务事件通道断连超时({int(now - self._biz_lost_start_ts)}s)，'
                        '清空事件重放缓存(数据丢弃)并结束批次',
                        'WARNING')
                    self._drop_event_cache()
                    self._auto_end('BIZ_LOST')
                    return

    @property
    def running(self) -> bool:
        return not self._stop_event.is_set()

    # ================= 回调 =================

    def _on_stream_status(self, status, extra=None):
        prev = self._stream_status
        self._stream_status = status
        self._stream_extra = extra or {}
        # 断流起始计时：进入 OFFLINE 记录起点；恢复（非 OFFLINE）清零
        if status == StreamStatus.OFFLINE and prev != StreamStatus.OFFLINE:
            self._offline_start_ts = time.time()
        elif status != StreamStatus.OFFLINE:
            self._offline_start_ts = None
        self._emit_event({
            'type': 'stream_status',
            'session': self.task.session,
            'status': status,
            'extra': extra or {},
            'ts': _now(),
        })

    def _on_gap(self, start, end):
        seconds = max(0.0, end - start)
        if seconds <= 0:
            return
        self._gap_pending = True
        # Stateful decoders/segmenter/timer are reset only by their owning inference thread.
        self._emit_event({
            'type': 'stream_gap',
            'session': self.task.session,
            'seconds': round(seconds, 2),
            'ts': _now(),
        })

    def add_alert(self, message: str, level: str = 'WARNING'):
        alert = {'type': level, 'message': message, 'ts': _now()}
        self._alerts.append(alert)
        self._emit_event({'type': 'alert', 'session': self.task.session,
                          'level': level, 'message': message, 'ts': _now()})
        return alert

    # ================= 推理线程 =================

    def _infer_loop(self):
        consecutive_errors = 0
        while self.running:
            try:
                frame = self.infer_queue.get(timeout=1.0)
            except Empty:
                continue

            packet = frame if isinstance(frame, TimedFrame) else TimedFrame(frame, captured_monotonic=time.monotonic())
            frame = packet.image

            # 断流恢复：清空滑窗（避免跨断流的时序污染）；
            # 动作段保留起点（on_stream_resume）——断流时长已由 notify_gap 计入 _gap_seconds 扣除，
            # 此处不能用 reset()：否则进行中段的起点被推后，动作 start_time/时长被截断。
            if self._gap_pending:
                self._gap_pending = False
                with self._detector_lock:
                    if self._detector:
                        self._detector.reset()
                if self._segmenter:
                    for event in self._segmenter.on_stream_resume():
                        self._emit_event(event)
                self._cleaning_timer.reset()

            try:
                with self._detector_lock:
                    if self._detector is None:
                        continue
                    result = self._detector.process(packet)
            except Exception as e:   # noqa: BLE001
                self._cleaning_timer.reset()
                if self._segmenter:
                    for event in self._segmenter.flush():
                        self._emit_event(event)
                # MediaPipe / 模型前向 / 归一化等异常不得静默杀死推理线程，
                # 否则任务状态仍为 RUNNING 却不再产出任何动作段（假运行）
                consecutive_errors += 1
                self.add_alert(f'推理异常(连续{consecutive_errors}次): {e}', 'ERROR')
                if consecutive_errors >= MAX_CONSECUTIVE_INFER_ERRORS:
                    self.task.status = TaskStatus.ERROR
                    self.task.error = f'推理连续失败: {e}'
                    self.add_alert('推理连续失败，任务置 ERROR 并终止推理线程', 'ERROR')
                    break
                time.sleep(0.1)
                continue

            consecutive_errors = 0
            self._infer_frames += 1
            self._fps_window.append(time.time())
            self._last_elapsed_ms = result.elapsed_ms

            duration_event = self._cleaning_timer.update(
                result.score_enabled and result.action_code in ('WATER', 'GAS', 'CLEANING_OPERATION') and result.observation_state == 'observed',
                result.source_pts, result.source_time_base, result.epoch,
                observation_state=result.observation_state)
            if duration_event:
                duration_event.update(station_no=self.station_no, batch_no=self.batch_no, ts=_now())
                duration_event['event_id'] = hashlib.sha256(json.dumps([
                    self.station_no, self.batch_no, result.epoch,
                    duration_event['evidence_scope'], duration_event['start_source_seconds']
                ], ensure_ascii=False).encode('utf-8')).hexdigest()
                self._emit_event(duration_event)

            if result.skipped or not result.score_enabled:
                if self._segmenter:
                    for event in self._segmenter.flush():
                        self._emit_event(event)
                # 滑窗重填期（任务启动 / 断流恢复后）模型无有效判定：
                # 不参与动作段统计，避免产生虚假的短 IDLE 段；仍更新叠加显示与数量检测
                self._diag_skipped += 1
                with self._result_lock:
                    self._latest_result = result
                self._maybe_object_detect(frame)
                self._diag_tick()
                continue

            self._diag_tick(result)

            if self._segmenter:
                events = self._segmenter.update(
                    result.action_id, result.action_code, result.confidence,
                    timestamp_s=float(packet.source_time) if packet.source_time is not None else packet.captured_monotonic,
                    observation_state=result.observation_state, time_basis=packet.time_basis, epoch=packet.epoch)
                for ev in events:
                    if ev.get('type') == 'action_segment':
                        self._log.info(
                            '[%s] 动作段提交: action_type=%s duration=%.1fs '
                            'conf=%.2f frames=%d',
                            self.station_no, ev.get('action_type'),
                            ev.get('duration'), ev.get('confidence'), ev.get('frames'))
                    self._emit_event(ev)

            # 出现清洗动作（非 IDLE）→ 重置无动作超时计时
            if result.action_code not in (None, 'IDLE', 'UNKNOWN') and result.observation_state == 'observed':
                self._no_action_secs = 0.0

            # 数量检测窗口关闭：触发动作经 merge_tolerance 确认为当前稳定段后才关窗
            # （防 vote_window 内偶发误判提前关窗——早关窗会漏检其后的摆放更正，
            #   晚关窗无代价：窗口取 max_hose_count。触发动作可配 object_model.count_stop_on，
            #   默认 WATER=首个水洗动作，见 docs/06 §6.2.7）
            if (self.count_stop_on and self._count_window_active
                    and self._segmenter is not None
                    and self._segmenter.current_name == self.count_stop_on):
                self._count_window_active = False
            # 目标检测（数量检测窗口内、有订阅时）
            self._maybe_object_detect(frame)

            with self._result_lock:
                self._latest_result = result

    # ---------------- 推理诊断日志 ----------------

    def _diag_tick(self, result=None):
        """诊断日志：动作变化即时输出；每 _diag_interval 秒汇总一次关键指标。

        汇总内容用于快速定位"有帧无动作段"类问题：
          - 跳过(滑窗未满)：持续偏高说明推理帧率不足或频繁断流
          - 规则拦截 L1/L3：说明被前置规则判为 IDLE，模型根本没参与
          - 当前动作+置信度：模型判定分布（恒 IDLE 则看置信度与规则计数）
        """
        now = time.time()
        if result is not None:
            if result.rule_hit:
                self._diag_rule_hits[result.rule_hit] += 1
            code = result.action_code
            if code != self._diag_last_code:
                self._log.info('[%s] 动作判定: %s → %s (conf=%.2f, 滑窗=%d帧)',
                               self.station_no,
                               self._diag_last_code or '(启动)', code,
                               result.confidence, result.buffered)
                self._diag_last_code = code
            if not self._diag_first_valid and self._detector is not None:
                self._diag_first_valid = True
                self._log.info('[%s] 滑窗已填满(seq_len=%d)，开始产出模型判定',
                               self.station_no, self._detector.seq_len)
        if now - self._diag_last_ts < self._diag_interval:
            return
        self._diag_last_ts = now
        self._log.info(
            '[%s] 推理诊断: 模型帧=%d 本期跳过(滑窗未满)=%d '
            '规则拦截=%s 当前动作=%s fps=%.0f 推理耗时=%.0fms',
            self.station_no, self._infer_frames, self._diag_skipped,
            dict(self._diag_rule_hits) or '无',
            f'{self._diag_last_code}', self.current_fps(),
            self._last_elapsed_ms)
        self._diag_skipped = 0
        self._diag_rule_hits.clear()

    # ---------------- 目标检测（差异 #9）----------------

    @staticmethod
    def _compute_roi_bbox(points):
        """多边形顶点 → 外接矩形 (x0,y0,x1,y1)；不足 3 点返回 None（全画面）"""
        if not points or len(points) < 3:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    def _object_crop(self, frame):
        """按目标检测独立 ROI 裁剪；返回 (crop, 偏移x, 偏移y)。无 ROI 原样返回。"""
        bbox = self.object_roi_bbox
        if not bbox:
            return frame, 0, 0
        x0, y0, x1, y1 = bbox
        h, w = frame.shape[:2]
        x0 = max(0, min(int(x0), w - 1))
        x1 = max(x0 + 1, min(int(x1), w))
        y0 = max(0, min(int(y0), h - 1))
        y1 = max(y0 + 1, min(int(y1), h))
        return frame[y0:y1, x0:x1], x0, y0

    def _maybe_object_detect(self, frame):
        det = self._object_detector
        if det is None or not self._count_window_active:
            return
        # 无订阅（业务未监听事件/帧）则不推理，省 CPU
        with self._sinks_lock:
            has_sink = bool(self._event_sinks) or bool(self._frame_sinks)
        if not has_sink:
            return

        self._object_frame_counter += 1
        if self._object_frame_counter % self.object_sample_rate != 0:
            return

        # 目标检测独立 ROI 裁剪（仅胶管区域），框坐标映射回原图
        crop, ox, oy = self._object_crop(frame)
        if crop is None or crop.size == 0:
            return
        try:
            dets = det.detect(crop)
        except Exception as e:   # noqa: BLE001
            self.add_alert(f'目标检测推理失败: {e}', 'ERROR')
            return

        # 检测框由裁剪局部坐标 → 原图像素坐标
        for d in dets:
            b = d['box']
            d['box'] = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]

        self._latest_object_dets = dets
        hose_count = len(dets)   # 检测器已仅输出目标类别（胶管）
        conf = (float(np.mean([d['score'] for d in dets]))
                if dets else 0.0)
        self._maybe_emit_count_report(hose_count, conf)

    def _maybe_emit_count_report(self, hose_count: int, conf: float):
        now = time.time()
        if (hose_count != self._last_hose_count or
                (now - self._last_count_ts) >= self.object_report_interval):
            self._last_hose_count = hose_count
            self._last_count_ts = now
            self._emit_event({
                'type': 'count_report',
                'station_no': self.station_no,
                'batch_no': self.batch_no,
                'hose_count': hose_count,
                'confidence': round(conf, 4),
                'class_counts': self._class_counts(self._latest_object_dets),
                'ts': _now(),
            })

    @staticmethod
    def _class_counts(dets):
        counts = {}
        for d in dets:
            counts[d['class_name']] = counts.get(d['class_name'], 0) + 1
        return counts

    # ================= 推送线程 =================

    def _check_video_quality(self, frame):
        """视频质量检测（docs/09 事件表 video_quality）：轻量 CV，不进模型、不依赖订阅。

        摄像头通常自带异常告警（黑屏/遮挡等可直接声音提醒），优先使用摄像头能力，
        本方法仅作辅助兜底；默认关闭（task_defaults.video_quality_enabled=false）。
        命中（黑屏/冻结/花屏/遮挡）经 WS 事件 video_quality 推送（瞬时告警，不进重放缓存）；
        异常类型集合变化时发一次（进入/退出/切换均触发）；恢复时 level=OK、types=[]。
        """
        if not self._vq_enabled or frame is None:
            return
        try:
            small = cv2.resize(frame, (64, 64))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        except Exception:   # noqa: BLE001
            return

        types = []
        mean = float(gray.mean())
        if mean < VQ_BLACK_MEAN_THRESHOLD:
            types.append('black')
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < VQ_OCCLUSION_LAP_VAR:
            types.append('occlusion')

        # 冻结：连续多帧像素哈希一致（解码卡死时画面完全相同）
        h = small.tobytes()
        if self._vq_last_hash is not None and h == self._vq_last_hash:
            self._vq_freeze_count += 1
        else:
            self._vq_freeze_count = 0
        self._vq_last_hash = h
        if self._vq_freeze_count >= VQ_FREEZE_FRAMES:
            types.append('freeze')

        # 花屏：极端像素（纯黑/纯白）占比过高 → 解码异常/噪点
        sat = float(np.mean((gray == 0) | (gray == 255)))
        if sat > VQ_CORRUPTION_SAT_RATIO:
            types.append('corruption')

        # 状态机：异常类型集合变化才发事件（防抖 + 进入/退出/切换）
        cur = set(types)
        if cur != set(self._vq_last_types):
            self._vq_last_types = sorted(cur)
            level = 'OK' if not cur else (
                'ERROR' if ('black' in cur or 'occlusion' in cur) else 'WARN')
            self._emit_event({
                'type': 'video_quality',
                'station_no': self.station_no,
                'batch_no': self.batch_no,
                'level': level,
                'types': sorted(cur),
                'details': {
                    'mean': round(mean, 2),
                    'lap_var': round(lap_var, 2),
                    'freeze_frames': int(self._vq_freeze_count),
                    'sat_ratio': round(sat, 3),
                },
                'ts': _now(),
            })

    def _push_loop(self):
        # 推送帧率：任务参数(Java下发) > task_defaults.push_fps
        push_fps = float(getattr(self.task.params, 'push_fps', None)
                         or (self.td.push_fps if self.td else 25) or 25)
        min_interval = 1.0 / max(1.0, push_fps)
        last_push = 0.0
        last_heartbeat = 0.0

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except Empty:
                frame = None

            now = time.time()

            if frame is not None:
                # 视频质量检测（每显示帧，独立于推送，不进模型）
                self._check_video_quality(frame)
                # 无订阅时不编码，省 CPU（无订阅不编码不上送）
                with self._sinks_lock:
                    has_sink = bool(self._frame_sinks)
                if has_sink and (now - last_push) >= min_interval:
                    last_push = now
                    with self._result_lock:
                        result = self._latest_result
                    self._push_frame(frame, result,
                                    boxes=self._latest_object_dets)

            hb_interval = float(getattr(self.td, 'heartbeat_interval', 1.0)
                                if self.td else 1.0)
            if (now - last_heartbeat) >= hb_interval:
                last_heartbeat = now
                self._emit_heartbeat()

    def _push_frame(self, frame, result, boxes=None):
        # 显示帧可能滞后于模型帧，result 可能为 None（首帧/模型尚未产出）
        action_name = result.action_name if result is not None else 'IDLE'
        action_code = result.action_code if result is not None else 'IDLE'
        confidence = result.confidence if result is not None else 0.0
        buffered = result.buffered if result is not None else 0
        results = result.results if result is not None else None
        rule_hit = result.rule_hit if result is not None else None
        # 骨架取最近一次模型结果（≤80ms 偏差，姿态变化缓慢可忽略）；无结果则不画
        draw_landmarks = bool(getattr(self.td, 'draw_landmarks', True)
                              if self.td else True) and results is not None
        try:
            # 先缩放后叠加：高分辨率源帧（如 4K）上按固定像素绘制的面板/文字，
            # 经编码缩到 ws.frame_width(640) 后仅剩骨架可辨、文字不可读。
            # MediaPipe 骨架为归一化坐标，缩放后绘制不受影响；目标检测框是
            # 原图像素坐标，须按缩放比换算。
            tw = getattr(self.cfg.ws, 'frame_width', None)
            if tw and 0 < tw < frame.shape[1]:
                h0, w0 = frame.shape[:2]
                th0 = max(1, int(h0 * tw / w0))
                frame = cv2.resize(frame, (tw, th0), interpolation=cv2.INTER_AREA)
                s = tw / w0
                if boxes:
                    boxes = [dict(d, box=[v * s for v in d['box']]) for d in boxes]
            annotated = draw_overlay(
                frame,
                action_name=action_name,
                action_code=action_code,
                confidence=confidence,
                duration=(self._segmenter.current_duration()
                          if self._segmenter else 0.0),
                buffered=buffered,
                seq_len=(self._detector.seq_len if self._detector else 30),
                alerts=list(self._alerts)[-3:],
                rule_hit=rule_hit,
                gap=self._gap_pending,
                draw_landmarks=draw_landmarks,
                extractor=(self._detector.extractor if self._detector else None),
                results=results,
                boxes=boxes,
                label_colors=self._label_colors,
                extra_info=f'{self._loaded.mode}/{self._loaded.info.get("model_type","")}'
                           if self._loaded else None,
                panel_scale=getattr(self.td, 'overlay_scale', None),
            )
            jpeg = encode_jpeg(annotated,
                              quality=(getattr(self.td, 'jpeg_quality', 85)
                                       if self.td else 85))
        except Exception as e:
            self.add_alert(f'标注帧生成失败: {e}', 'ERROR')
            return

        if not jpeg:
            return

        with self._sinks_lock:
            sinks = list(self._frame_sinks)
        for sink in sinks:
            try:
                sink(jpeg)
            except Exception:
                pass

    def _emit_heartbeat(self):
        seg_stats = self._segmenter.stats() if self._segmenter else {}
        with self._result_lock:
            latest = self._latest_result
        self._emit_event({
            'type': 'heartbeat',
            'session': self.task.session,
            'station_no': self.station_no,
            'batch_no': self.batch_no,
            'task_id': self.task.id,
            'status': self.task.status,
            'stream_status': self._stream_status,
            'current_action': seg_stats.get('current_action'),
            'current_duration': seg_stats.get('current_duration', 0.0),
            'observation_state': latest.observation_state if latest else 'unknown',
            'decision_reason': latest.decision_reason if latest else 'not_observed',
            'time_basis': latest.time_basis if latest else 'unverified',
            'cleaning_action_count': self._cleaning_timer.count,
            'actual_connection_verified': False,
            'per_hose_completion': False,
            'fps': self.current_fps(),
            'infer_ms': round(self._last_elapsed_ms, 2),
            'buffered': (self._detector.stats().get('buffered') if self._detector else 0),
            'queue': {'frames': self.frame_queue.qsize(),
                      'infer': self.infer_queue.qsize()},
            'ts': _now(),
        })

    # ================= 订阅 =================

    def add_frame_sink(self, cb):
        with self._sinks_lock:
            self._frame_sinks.add(cb)

    def remove_frame_sink(self, cb):
        with self._sinks_lock:
            self._frame_sinks.discard(cb)

    def add_event_sink(self, cb):
        with self._sinks_lock:
            self._event_sinks.add(cb)

    def remove_event_sink(self, cb):
        with self._sinks_lock:
            self._event_sinks.discard(cb)

    # 参与重放的业务事件类型（幂等键 batch_no+action_type+start_time 天然去重）；
    # count_report/heartbeat/alert 等周期性/瞬时事件不缓存——丢失由后续上报或批末汇总兜底
    _REPLAY_TYPES = {'batch_started', 'action_segment', 'batch_ended', 'cleaning_action_duration_met'}

    def replay_events_json(self):
        """重放缓存：当前缓存全量事件 JSON 列表（按缓存序=时间序，docs/09 §五）。

        WS 断线重连时服务端先全量重放再进实时流（可能重复，业务按
        batch_no+action_type+start_time 幂等消化）；缓存关闭时返回空列表。
        """
        if self._event_cache is None:
            return []
        with self._sinks_lock:
            snap = list(self._event_cache)
        return [json.dumps(e, ensure_ascii=False) for e in snap]

    def _drop_event_cache(self):
        """业务断连超时（BIZ_LOST，docs/06 §七）：清空事件重放缓存。

        数据丢弃语义：断连期间积压的待补发事件整体作废，重连后不再重放
        （仅保留其后产生的事件，如随后的 batch_ended），由业务侧按
        reason=BIZ_LOST 落批次异常态，不承诺中间过程数据补发。
        """
        if self._event_cache is not None:
            with self._sinks_lock:
                self._event_cache.clear()

    def _emit_event(self, event: dict):
        # 事件缓存（不可丢语义的兜底）：仅缓存参与重放的业务事件（幂等键天然在事件字段中）
        if self._event_cache is not None and event.get('type') in self._REPLAY_TYPES:
            self._event_cache.append(dict(event))
        with self._sinks_lock:
            sinks = list(self._event_sinks)
        for sink in sinks:
            try:
                sink(event)
            except Exception:
                pass

    # ================= 运维 =================

    def _object_class_names(self, om_cfg) -> dict:
        """构建 类id→名称 映射（来自 service.yaml object_model.classes）"""
        names = {}
        classes = getattr(om_cfg, 'classes', None) or []
        for c in classes:
            if isinstance(c, dict):
                cid = c.get('id')
                name = c.get('name') or c.get('label')
                if cid is not None and name:
                    names[int(cid)] = str(name)
            elif isinstance(c, str):
                names[len(names)] = c
        return names

    def current_fps(self) -> float:
        now = time.time()
        while self._fps_window and now - self._fps_window[0] > 1.0:
            self._fps_window.popleft()
        return float(len(self._fps_window))

    def status(self) -> dict:
        d = self.task.to_dict()
        d.update({
            'stream_status': self._stream_status,
            'stream_extra': self._stream_extra,
            'fps': round(self.current_fps(), 1),
            'infer_frames': self._infer_frames,
            'infer_ms': round(self._last_elapsed_ms, 2),
            'queue': {'frames': self.frame_queue.qsize(),
                      'infer': self.infer_queue.qsize()},
            'detector': self._detector.stats() if self._detector else None,
            'segmenter': self._segmenter.stats() if self._segmenter else None,
            'cleaning_action_count': self._cleaning_timer.count,
            'per_hose_completion': False,
            'reader': self._reader.stats() if self._reader else None,
            'model': {
                'checkpoint': self.task.checkpoint,
                'mode': self._loaded.mode if self._loaded else None,
                'seq_len': self._loaded.seq_len if self._loaded else None,
                'feature_dim': self._loaded.feature_dim if self._loaded else None,
                'aug_version': self._loaded.aug_version if self._loaded else None,
                'sample_rate': self._loaded.sample_rate if self._loaded else None,
                'label_map': self._loaded.label_map if self._loaded else None,
                'train_fps': self._loaded.train_fps if self._loaded else None,
            },
            'rules': self.rules.to_dict(),
            'alerts': list(self._alerts)[-5:],
        })
        return d

    def set_batch(self, batch_no: str, station_no: str = None):
        """更新当前批次号（同工位复用任务、业务未发 batch_end 就扫下一批时调用）。

        必须同步到 segmenter，否则新批次的 action_segment 仍带旧 batch_no。
        """
        self.batch_no = batch_no
        self._cleaning_timer.reset()
        self._cleaning_timer.count = 0
        if station_no:
            self.station_no = station_no
        if self._segmenter:
            self._segmenter.set_batch(self.station_no, self.batch_no)
            # 新批次必须重置批次内状态：清空当前段与抖动候选
            self._segmenter.reset_batch_state()
        if self.task:
            self.task.batch_no = batch_no

        # 新批次需重置数量检测窗口：
        # 上个批次一旦出现 count_stop_on 动作帧就永久关闭，不重置则新批次不再产出 count_report
        if self._object_detector is not None:
            self._count_window_active = True
            self._last_hose_count = -1
            self._last_count_ts = 0.0
            self._object_frame_counter = 0
            self._latest_object_dets = []
