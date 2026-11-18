"""
infer_service/engine/stream.py - 视频流读取线程

职责:
  - 打开 RTSP / RTMP / HTTP-FLV / 摄像头 / 视频文件
  - 低延迟参数（UDP + nobuffer + 小 probesize）
  - 断流自动重连（指数退避），断流期间标记 GAP
  - 帧率统计与状态回调（LIVE / WEAK / OFFLINE）

双路分流（显示与模型解耦）:
  - frame_queue : 全速显示帧（= 源帧率，如 25fps），供推送线程做流畅预览
  - infer_queue : 模型帧（按 infer_fps = 源/采样率 抽取，当前 25/1=25fps），供推理线程
  模型帧是源帧的子集，故推理实际帧率与训练(源/采样率)严格一致；显示帧不抽帧→预览不卡。
"""

import logging
import os
import threading
import time
from queue import Empty, Full, Queue

logger = logging.getLogger('infer_service')

# 低延迟拉流参数（需在 cv2 加载前设置才生效）
os.environ.setdefault(
    'OPENCV_FFMPEG_CAPTURE_OPTIONS',
    'rtsp_transport;udp|fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0'
)

import cv2  # noqa: E402
from common.frame_timing import TimedFrame


class _AVCapture:
    """Decoder retaining the timestamps supplied by the source container."""
    def __init__(self, source, is_file):
        import av
        self.container = av.open(str(source), **({} if is_file else {'timeout': (5., 3.)}))
        self.decoder = iter(self.container.decode(video=0))
        self.pts = self.time_base = None
        self.native_frame = None

    def read_native(self):
        """Keep native pixels for frozen ROI filters; do not fabricate FPS timestamps."""
        try:
            frame = next(self.decoder)
        except StopIteration:
            return None
        self.native_frame = frame
        self.pts, self.time_base = frame.pts, str(frame.time_base) if frame.time_base else None
        return frame

    def read(self):
        try:
            frame = self.read_native()
            return (False, None) if frame is None else (True, frame.to_ndarray(format='bgr24'))
        except Exception:
            return False, None

    def release(self):
        self.container.close()


class StreamStatus:
    LIVE = 'LIVE'
    WEAK = 'WEAK'
    OFFLINE = 'OFFLINE'


class StreamReader(threading.Thread):
    """拉流线程：全速入 frame_queue，按 infer_fps 抽取模型帧入 infer_queue"""

    def __init__(self, source, frame_queue: Queue, infer_queue: Queue = None,
                 display_fps: float = 25.0, infer_fps: float = 25.0,
                 reconnect_max_attempts: int = 8,
                 reconnect_initial_delay: float = 2.0,
                 on_status=None, on_gap=None, name: str = 'stream'):
        super().__init__(name=name, daemon=True)
        self.source = source
        self.frame_queue = frame_queue          # 全速显示帧
        self.infer_queue = infer_queue          # 模型帧（可空）
        self.display_fps = max(1.0, float(display_fps))
        self.infer_fps = max(1.0, float(infer_fps))
        self.reconnect_max_attempts = reconnect_max_attempts
        self.reconnect_initial_delay = reconnect_initial_delay
        self.on_status = on_status or (lambda s, extra: None)
        self.on_gap = on_gap or (lambda start, end: None)

        self._stop_event = threading.Event()
        self._cap = None
        self._is_file = isinstance(source, str) and os.path.isfile(source)
        self._epoch = 0

        # 统计
        self.frame_count = 0
        self.reconnect_count = 0
        self.dropped_frames = 0          # 因队列满被丢弃的帧数（丢帧率监控）
        self.gap_total_seconds = 0.0
        self._fps_window = []
        self._last_status = None

        # 节流参数
        self._display_min = 1.0 / self.display_fps
        self._infer_min = 1.0 / self.infer_fps
        self._last_infer_ts = 0.0

    # ---------------- 控制 ----------------

    def stop(self):
        self._stop_event.set()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    # ---------------- 内部 ----------------

    def _open(self):
        self._epoch += 1
        if not isinstance(self.source, int):
            try:
                return _AVCapture(self.source, self._is_file)
            except Exception as exc:
                logger.warning('PTS decoder unavailable (%s); source not opened', type(exc).__name__)
                return None
        cap = cv2.VideoCapture(self.source)
        # 打开/读取超时（毫秒）：防 RTSP 半开挂起时 cap.read() 永久阻塞拉流线程；
        # 读超时 → 走断流路径（指数退避重连，超限 OFFLINE），替代文档原"独立心跳探测"设计
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _set_status(self, status, extra=None):
        if status != self._last_status:
            prev = self._last_status
            self._last_status = status
            if status == StreamStatus.OFFLINE:
                logger.warning('视频流 OFFLINE: source=%s reason=%s',
                               str(self.source), (extra or {}).get('reason', ''))
            elif status == StreamStatus.LIVE and prev == StreamStatus.OFFLINE:
                logger.info('视频流恢复 LIVE: source=%s', str(self.source))
            self.on_status(status, extra or {})

    def _check_fps(self, now):
        """每秒统计一次实际帧率，过低标记 WEAK（按显示帧率口径）"""
        self._fps_window.append(now)
        cutoff = now - 1.0
        while self._fps_window and self._fps_window[0] < cutoff:
            self._fps_window.pop(0)
        fps = len(self._fps_window)
        if fps > 0 and fps < self.display_fps * 0.5:
            self._set_status(StreamStatus.WEAK, {'fps': fps})
        else:
            self._set_status(StreamStatus.LIVE, {'fps': fps})
        return fps

    def _put_frame(self, frame):
        """放入显示队列，满则丢弃最旧帧（保证实时性）"""
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
                self.dropped_frames += 1
            except Empty:
                pass
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except Full:
            return False

    def _put_infer(self, frame):
        """放入模型队列，满则丢弃最旧帧"""
        if self.infer_queue is None:
            return
        if self._is_file:
            # Offline replay may be slower than source FPS: preserve samples rather than skip ahead.
            while not self.stopped:
                try:
                    self.infer_queue.put(frame, timeout=.1)
                    return
                except Full:
                    continue
            return
        if self.infer_queue.full():
            try:
                self.infer_queue.get_nowait()
                self.dropped_frames += 1
            except Empty:
                pass
        try:
            self.infer_queue.put_nowait(frame)
        except Full:
            pass

    # ---------------- 主循环 ----------------

    def run(self):
        attempt = 0
        delay = self.reconnect_initial_delay

        while not self.stopped:
            self._cap = self._open()

            if self._cap is None:
                attempt += 1
                if self.reconnect_max_attempts > 0 and attempt > self.reconnect_max_attempts:
                    self._set_status(StreamStatus.OFFLINE,
                                     {'reason': '重连次数超限', 'attempt': attempt})
                    break
                self._set_status(StreamStatus.OFFLINE,
                                 {'reason': '打开失败', 'attempt': attempt})
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue

            # 打开成功，重置退避
            attempt = 0
            delay = self.reconnect_initial_delay
            self._set_status(StreamStatus.LIVE, {'reason': '已连接'})
            min_interval = self._display_min
            last_ts = 0.0
            self._last_infer_ts = None
            next_infer = None
            file_clock_start = file_source_start = None

            while not self.stopped:
                ret, frame = self._cap.read()
                now = time.time()

                if not ret or frame is None:
                    # 断流：跳出读帧循环，进入重连（GAP 时长在重连块统一统计）
                    break

                pts = getattr(self._cap, 'pts', None)
                base = getattr(self._cap, 'time_base', None)
                packet = TimedFrame(frame, pts, base, time.monotonic(), self._epoch, self.frame_count,
                                    native_video_frame=getattr(self._cap, 'native_frame', None))
                source_time = packet.source_time
                sample_clock = float(source_time) if source_time is not None else packet.captured_monotonic
                if self._is_file and source_time is not None:
                    if file_clock_start is None:
                        file_clock_start, file_source_start = time.monotonic(), sample_clock
                    delay_seconds = file_clock_start+sample_clock-file_source_start-time.monotonic()
                    if delay_seconds > 0 and self._stop_event.wait(delay_seconds):
                        break

                # 显示帧限速到源帧率（避免视频文件场景空转；实时流本身即此速率）
                last_ts = now

                self.frame_count += 1
                self._put_frame(frame)                       # 全速显示帧

                # 模型帧：按 infer_fps(=源/采样率) 抽取，与训练一致
                if (self.infer_queue is not None
                        and (next_infer is None or sample_clock+1e-8 >= next_infer)):
                    next_infer = sample_clock+self._infer_min if next_infer is None else next_infer+self._infer_min
                    if next_infer <= sample_clock:
                        next_infer = sample_clock+self._infer_min
                    self._last_infer_ts = sample_clock
                    self._put_infer(packet)

                self._check_fps(now)

            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

            if self.stopped:
                break
            if self._is_file:
                self._set_status(StreamStatus.OFFLINE, {'reason': '文件读取结束；不循环重放'})
                break

            # 断流重连（指数退避）
            self.reconnect_count += 1
            if (self.reconnect_max_attempts > 0
                    and self.reconnect_count > self.reconnect_max_attempts):
                self._set_status(StreamStatus.OFFLINE,
                                 {'reason': '重连次数超限',
                                  'attempt': self.reconnect_count})
                break

            gap_start_ts = time.time()
            self._set_status(StreamStatus.OFFLINE,
                             {'reason': '断流重连中',
                              'attempt': self.reconnect_count,
                              'delay': delay})
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
            gap_seconds = time.time() - gap_start_ts
            self.gap_total_seconds += gap_seconds
            self.on_gap(gap_start_ts, time.time())

    def stats(self) -> dict:
        total = self.frame_count + self.dropped_frames
        return {
            'source': str(self.source),
            'frames': self.frame_count,
            'reconnects': self.reconnect_count,
            'dropped_frames': self.dropped_frames,
            # 丢帧率：过高说明推理跟不上，相邻帧间隔变大 → 速度/加速度特征失真
            'drop_rate': round(self.dropped_frames / total, 4) if total else 0.0,
            'gap_total_seconds': round(self.gap_total_seconds, 2),
            'status': self._last_status,
            'queue_size': self.frame_queue.qsize(),
        }
