"""
infer_service/config.py - 推理服务统一配置（单一来源 service.yaml）

推理服务唯一配置来源。配置结构见 docs/06-推理服务设计说明.md 第六章。

设计要点：
  - 单一来源：监听端口 / 上送地址 / 工位映射 / 视频流 / 两类模型 /
    前后置规则 / WS / 引擎运行参数 均来自 service.yaml
    （v1.5.2 起无单根/切分参数：批次式流程无单根序列，
    分组边界由业务按 SHELF_PUT/SHELF_TAKE 判定，见 docs/08/10）
  - 模型类参数（seq_len/feature_dim/normalizer/label_map）**不在本文件**，
    只从模型包读取（path 下的 model_info.json 或 label_map.json）；动作码契约由模型包提供，
    配置加载时按 action_model.path 读取填充（见 _load_model_label_map），加载模型时按动作码契约校验 id 集合（见 model_pool.py）
  - 优先级: 任务参数(Java下发) > service.yaml > 代码默认值
  - 注意: 模型类参数只从 checkpoint 读取，不信任外部配置，避免错配。
"""

from __future__ import annotations
import copy
import json
import os
from dataclasses import dataclass, field, asdict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 运行时唯一配置：config/service.yaml（config/ 统一存放三端配置 + 标签契约，无镜像副本）
DEFAULT_CONFIG_PATH = os.path.join(ROOT, 'config', 'service.yaml')


# ============================================================
# 数据结构
# ============================================================


@dataclass
class ServerConfig:
    host: str = '127.0.0.1'
    port: int = 8000
    api_token: str = 'change-me-please'     # 生产部署务必修改（与 Java 侧共享配置）
    workers: int = 1


@dataclass
class BizInterface:
    """业务鉴权（HMAC 入站签名密钥）。HTTP 上送兜底未实现，不再声明上送地址/开关。"""
    app_key: str = 'jl_sop'
    secret: str = 'change_me'


@dataclass
class StationMapping:
    station_no: str
    video_stream_id: int
    object_model_enable: bool = False    # 工位级目标检测开关：true=启用（模型路径取全局 object_model）


@dataclass
class VideoStream:
    name: str
    rtsp_url: str
    resolution: str = '1920x1080'
    fps: float = 25.0                # 源流帧率（摄像头实际帧率；支持小数）
    # 推理帧率=模型输入帧率(=源fps/sample_rate)；支持小数。
    # Optional：null/缺省 = 跟随全局 task_defaults.infer_fps（合并前此处与 task_defaults 双份存值易漂移）
    infer_fps: float | None = None
    roi_points: list = field(default_factory=list)   # 多边形顶点 [x,y]，空=全画面

    @property
    def has_roi(self) -> bool:
        return bool(self.roi_points)


@dataclass
class ActionModelCfg:
    path: str = 'models/mh_full_lstm_best'
    enabled_actions: list | None = None       # 空/None=全部启用；否则为 label_map.id 子集
    label_map: dict = field(default_factory=dict)  # 动作码契约（id↔name/color），运行时由模型包按 path 加载填充；配置不再内联


@dataclass
class ObjectModelCfg:
    path: str = 'models/yolo_hose_best'
    engine: str = 'onnxruntime'
    input_size: list = field(default_factory=lambda: [640, 640])
    conf_threshold: float = 0.5
    nms_threshold: float = 0.45
    classes: list = field(default_factory=list)
    target_class: str = 'hose'                 # 仅统计/标注的类别（只检测胶管）
    # 数量检测窗口关闭触发动作（契约编码）：该动作出现即停止 count_report 上送；
    # 空/null = 不自动关闭（整批次持续上报）。默认 WATER=首个水洗动作出现即停止。
    count_stop_on: str = 'WATER'
    roi_points: list = field(default_factory=list)   # 目标检测独立 ROI（多边形顶点，坐标=分辨率）；空=全画面


@dataclass
class WSConfig:
    enabled: bool = True
    frame_width: int = 640
    # 事件补发（docs/06 §6.2.10 / docs/09 §五）：环形缓存业务事件，断线重连时按序全量重放
    event_cache_enabled: bool = True   # false=关闭缓存，补发不可用（业务侧仅剩按唯一键幂等兜底）
    event_cache_size: int = 512        # 缓存上限（条）；溢出淘汰最旧，超出部分不可补发
    # frame_fps / jpeg_quality 已合并到 task_defaults.push_fps / task_defaults.jpeg_quality
    # （原 ws.frame_fps 与 task_defaults.push_fps 同义双份；ws.jpeg_quality 从未被代码读取）


@dataclass
class PreRules:
    l1_threshold: float = 0.5
    l3_idle_threshold: float = 0.3
    l3_min_frames: int = 10
    missing_pose_action: str = 'unknown'
    l3_mode: str = 'contextual'


@dataclass
class PostRules:
    """后处理规则（唯一的后处理参数来源，task_defaults 中的同义项已全部合并于此）"""
    vote_window: int = 3                  # 多数投票窗口（帧），抑制单帧跳变
    conf_threshold: float = 0.6           # 置信度下限：低于该值沿用上一稳定动作（OOD 兜底）
    merge_tolerance: int = 10             # 段合并容差缺省值（帧），短于此帧数的段并入相邻段
    min_action_duration: float = 0.0      # 动作段最短时长（秒），更短的段视为抖动
    # 按动作类型覆盖 merge_tolerance（缺省回落 merge_tolerance）；key 为动作编码
    merge_tolerance_frames: dict = field(default_factory=dict)
    low_conf_hold_seconds: float = .2
    require_hand_for_cleaning: bool = True
    cleaning_seconds: float = 3.0
    max_observation_gap_seconds: float = .12


@dataclass
class HandEvidenceConfig:
    enabled: bool = False
    manifest: str = 'models/glove_evidence/manifest.json'
    device: str = 'cpu'
    score_threshold: float = .5
    roi_normalized: list | None = None


@dataclass
class LocalActionConfig:
    """Opt-in raw RGB/hand backend; independent of the old XYZ feature contract."""
    enabled: bool = False
    manifest: str = 'models/local_action/manifest.json'
    device: str = 'cpu'


@dataclass
class ModelPoolConfig:
    cache_size: int = 4
    device: str | None = None             # None=自动(cuda优先)
    strict_checkpoint: bool = True        # 缺 normalizer 时是否直接报错


@dataclass
class TaskDefaults:
    """引擎运行参数（帧率 / 重连 / 预览编码 / 特征开关）。

    后处理类参数（smooth_window / confidence_threshold / merge_tolerance /
    min_action_duration）已合并到 post_rules，此处不再声明，避免双份漂移。
    """
    infer_fps: float = 25.0               # 全局推理帧率(=训练等效帧率 train_fps/sample_rate)，支持小数
    push_fps: int = 25                    # 标注帧推送帧率（= 源帧率，全速不抽帧，与显示一致）
    reconnect_max_attempts: int = 8
    reconnect_initial_delay: float = 2.0
    heartbeat_interval: float = 1.0
    draw_landmarks: bool = True           # 标注帧是否绘制骨架
    overlay_scale: float | None = None    # None retains the original width-adaptive panel
    jpeg_quality: int = 85                # 标注帧 JPEG 编码质量（ws.jpeg_quality 已合并于此）
    # 特征增强(augv7)的近似开关：详见 common/augment.py 说明
    enable_duration_feat: bool = True
    enable_visibility_mask: bool = True
    # augv7「窗内稳定帧比例(stability)」判定所用的运动量阈值。
    # ⚠️ 与 pre_rules.l3_idle_threshold（业务"是否静止→判 IDLE"）解耦：
    #    二者同源于 total_motion，但调业务阈值不应连带改变模型输入特征分布。
    aug_stability_threshold: float = 0.3
    # 推理侧主动批次结束（docs/09 §二 / 07 §七）：业务侧不计时，由推理服务检测触发
    scan_action_timeout: float = 300.0    # 无清洗动作超时（秒）：批次启动后持续无清洗动作段超此值 → SCAN_TIMEOUT
    stream_offline_timeout: float = 300.0 # 断流超时（秒）：视频流持续 OFFLINE 超此值 → INTERRUPTED
    # 业务重连超时（秒）：批次运行中事件 WS 无订阅者（业务断连）持续超此值
    # → 清空事件重放缓存（数据丢弃，不承诺补发）并主动结束批次 reason=BIZ_LOST；
    # 0 = 关闭该项检测（断连仅靠环形缓存容量自然淘汰）。事件通道重连即重置计时。
    biz_lost_timeout: float = 60.0
    # 视频质量检测（docs/09 事件表 video_quality）：默认关闭；
    # 摄像头通常自带异常告警（黑屏/遮挡等可直接声音提醒），优先使用摄像头自身能力，
    # 本项仅作辅助兜底——启用后经 WS 事件 video_quality 推送告警（瞬时告警，不进重放缓存）
    video_quality_enabled: bool = False


@dataclass
class RulesConfig:
    """【已废弃 · 仅向后兼容旧 service.yaml】旧版 `rules` 段。

    v1.4.0 起该段全部字段已合并：`l1_missing_threshold / l3_motion_threshold /
    idle_min_frames` → `pre_rules`；`enabled_actions` → `action_model.enabled_actions`；
    L1/L2/L3 开关在 `pre_rules` 路径下被覆盖，不再可配。
    service.yaml 已移除 `rules` 段；旧配置若仍存在该段，字段会被读取但仅在
    `pre_rules/post_rules` 缺失的兜底路径生效。
    """
    l1_enabled: bool = True               # L1 无人：关键点缺失
    l1_missing_threshold: float = 0.7     # 关键点全零比例阈值
    l2_enabled: bool = False              # L2 不在工位：ROI 判定
    l2_roi: list | None = None            # [x1, y1, x2, y2] 归一化坐标
    l3_enabled: bool = True               # L3 休息：低运动
    l3_motion_threshold: float = 0.01     # total_motion 低于此值视为静止
    idle_min_frames: int = 8              # 连续静止达到该帧数才判 IDLE
    # 动作开关：允许检测的动作名列表（与模型 label_map 名称一致，如 ["WATER","GAS"]）。
    # None/空列表 = 全部启用（默认）；未启用的动作按 IDLE 处理，不产生动作段/事件。
    enabled_actions: list | None = None


@dataclass
class ServiceConfig:
    """service.yaml 的高层访问接口（单一来源，线程安全由调用方保证）"""
    server: ServerConfig = field(default_factory=ServerConfig)
    biz: BizInterface = field(default_factory=BizInterface)
    model_pool: ModelPoolConfig = field(default_factory=ModelPoolConfig)
    task_defaults: TaskDefaults = field(default_factory=TaskDefaults)
    rules: RulesConfig = field(default_factory=RulesConfig)
    stations: list = field(default_factory=list)            # List[StationMapping]
    streams: list = field(default_factory=list)             # List[VideoStream]
    action_model: ActionModelCfg = field(default_factory=ActionModelCfg)
    object_model: ObjectModelCfg | None = None
    ws: WSConfig = field(default_factory=WSConfig)
    pre_rules: PreRules = field(default_factory=PreRules)
    post_rules: PostRules = field(default_factory=PostRules)
    hand_evidence: HandEvidenceConfig = field(default_factory=HandEvidenceConfig)
    local_action: LocalActionConfig = field(default_factory=LocalActionConfig)
    config_version: str = '0.0.0'
    updated_at: str | None = None

    # 运行时附加（非持久化）
    _raw: dict = field(default_factory=dict, repr=False)
    _path: str | None = field(default=None, repr=False)

    # ---------------- 访问器 ----------------

    def get_station(self, station_no: str):
        for st in self.stations:
            if st.station_no == station_no:
                return st
        return None

    def get_stream(self, stream_id: int):
        # video.streams[] 下标 1 起
        if 1 <= stream_id <= len(self.streams):
            return self.streams[stream_id - 1]
        return None

    def action_label_ids(self) -> set:
        """动作码契约（config/label_catalog.yaml）的全部动作编码，
        用于模型加载校验 / enabled_actions 校验。"""
        from common.label_catalog import LABEL_IDS
        return set(LABEL_IDS)

    def enabled_action_ids(self) -> set | None:
        """动作检测开关解析为编码集合；None/空 = 全部启用"""
        en = self.action_model.enabled_actions
        if not en:
            return None
        return set(str(x) for x in en)

    def infer_fps_for(self, stream) -> float:
        """解析某路流的推理帧率：流级 infer_fps 未配置(null)时回落全局 task_defaults.infer_fps。"""
        fps = getattr(stream, 'infer_fps', None)
        return float(fps) if fps else float(self.task_defaults.infer_fps)

    def secret_for(self, station_no: str) -> str:
        """WS/HTTP 签名密钥（当前全局同一 secret）"""
        return self.biz.secret

    def app_key(self) -> str:
        return self.biz.app_key

    # ---------------- 序列化 ----------------

    def to_dict(self) -> dict:
        return copy.deepcopy(self._raw)

    @classmethod
    def from_raw(cls, raw: dict, path: str = None) -> 'ServiceConfig':
        raw = raw or {}
        s = raw.get('server', {}) or {}
        server = ServerConfig(
            host=s.get('host', '127.0.0.1'),
            port=int(s.get('port', 8000)),
            api_token=s.get('api_token', 'change-me-please'),
            workers=int(s.get('workers', 1)),
        )

        b = raw.get('interfaces', {}).get('biz', {}) or {}
        ba = b.get('auth', {}) or {}
        biz = BizInterface(
            app_key=ba.get('app_key', 'jl_sop'),
            secret=ba.get('secret', 'change_me'),
        )

        if raw.get('rules'):
            # 旧 rules 段 v1.4.0 已删除（阈值并入 pre_rules、开关并入 action_model.enabled_actions）：
            # 显式忽略并告警，绝不静默兜底生效（配置单一来源原则，docs/06 §6.2.14）
            print('[config] 警告: 检测到已废弃的 rules 段，已忽略——'
                  '阈值请配置 pre_rules，动作开关请配置 action_model.enabled_actions')

        stations = [
            StationMapping(
                station_no=st['station_no'],
                video_stream_id=int(st['video_stream_id']),
                object_model_enable=bool(st.get('object_model_enable', False)),
            )
            for st in raw.get('stations', [])
        ]

        streams = [
            VideoStream(
                name=vs['name'],
                rtsp_url=vs['rtsp_url'],
                resolution=vs.get('resolution', '1920x1080'),
                fps=float(vs.get('fps', 25.0)),
                # null/缺省 = 跟随全局 task_defaults.infer_fps
                infer_fps=(float(vs['infer_fps'])
                           if vs.get('infer_fps') is not None else None),
                roi_points=vs.get('roi_points', []) or [],
            )
            for vs in raw.get('video', {}).get('streams', [])
        ]

        am = raw.get('action_model', {}) or {}
        action_model = ActionModelCfg(
            path=am.get('path', 'models/mh_full_lstm_best'),
            enabled_actions=am.get('enabled_actions'),
            label_map=_load_model_label_map(am.get('path', 'models/mh_full_lstm_best')),
        )

        om = raw.get('object_model', {}) or {}
        object_model = ObjectModelCfg(
            path=om.get('path', 'models/yolo_hose_best'),
            engine=om.get('engine', 'onnxruntime'),
            input_size=om.get('input_size', [640, 640]),
            conf_threshold=float(om.get('conf_threshold', 0.5)),
            nms_threshold=float(om.get('nms_threshold', 0.45)),
            classes=om.get('classes', []),
            target_class=om.get('target_class', 'hose'),
            count_stop_on=(str(om['count_stop_on'])
                           if om.get('count_stop_on') else 'WATER'),
            roi_points=om.get('roi_points', []) or [],
        ) if om else None

        ws = raw.get('ws', {}) or {}
        ws_cfg = WSConfig(
            enabled=bool(ws.get('enabled', True)),
            event_cache_enabled=bool(ws.get('event_cache_enabled', True)),
            event_cache_size=max(1, int(ws.get('event_cache_size', 512))),
            frame_width=int(ws.get('frame_width', 640)),
        )

        pr = raw.get('pre_rules', {}) or {}
        pre_rules = PreRules(
            l1_threshold=float(pr.get('l1_threshold', 0.5)),
            l3_idle_threshold=float(pr.get('l3_idle_threshold', 0.3)),
            l3_min_frames=int(pr.get('l3_min_frames', 10)),
            missing_pose_action=str(pr.get('missing_pose_action', 'unknown')),
            l3_mode=str(pr.get('l3_mode', 'contextual')),
        )

        po = raw.get('post_rules', {}) or {}
        post_rules = PostRules(
            vote_window=int(po.get('vote_window', 3)),
            conf_threshold=float(po.get('conf_threshold', 0.6)),
            merge_tolerance=int(po.get('merge_tolerance', 10)),
            min_action_duration=float(po.get('min_action_duration', 0.0)),
            merge_tolerance_frames=po.get('merge_tolerance_frames', {}) or {},
            low_conf_hold_seconds=max(0., float(po.get('low_conf_hold_seconds', .2))),
            require_hand_for_cleaning=bool(po.get('require_hand_for_cleaning', True)),
            cleaning_seconds=float(po.get('cleaning_seconds', 3.)),
            max_observation_gap_seconds=float(po.get('max_observation_gap_seconds', .12)),
        )

        cfg = cls(
            server=server, biz=biz,
            model_pool=_merge(ModelPoolConfig, raw.get('model_pool')),
            task_defaults=_merge(TaskDefaults, raw.get('task_defaults')),
            stations=stations, streams=streams,
            action_model=action_model, object_model=object_model,
            ws=ws_cfg, pre_rules=pre_rules, post_rules=post_rules,
            hand_evidence=_merge(HandEvidenceConfig, raw.get('hand_evidence')),
            local_action=_merge(LocalActionConfig, raw.get('local_action')),
            config_version=raw.get('config_version', '0.0.0'),
            updated_at=raw.get('updated_at'),
        )
        cfg._raw = raw
        cfg._path = path
        return cfg

    def to_dict_full(self) -> dict:
        return asdict(self)


# ============================================================
# 工具
# ============================================================


def _load_model_label_map(model_path: str) -> dict:
    """从模型包加载 label_map（动作码契约 id↔name/color）。

    优先 path/label_map.json，回退 path/model_info.json 的 label_map 字段；
    模型包缺失时返回空（模型加载时再按动作码契约校验）。
    """
    if not model_path:
        return {}
    label_path = os.path.join(model_path, 'label_map.json')
    info_path = os.path.join(model_path, 'model_info.json')
    try:
        if os.path.isfile(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {str(k): v for k, v in data.items()}
        if os.path.isfile(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            lm = info.get('label_map')
            if lm:
                return {str(k): v for k, v in lm.items()}
    except Exception:
        pass
    return {}


def _merge(dataclass_type, data: dict):
    if not data:
        return dataclass_type()
    valid = {k: v for k, v in data.items() if k in dataclass_type.__dataclass_fields__}
    return dataclass_type(**valid)


def load_config(path: str = None) -> ServiceConfig:
    """加载 service.yaml（缺失时使用默认配置，不做静默降级之外的特殊处理）"""
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        return ServiceConfig()
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    return ServiceConfig.from_raw(raw, path)
