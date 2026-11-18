"""
infer_service/rules.py - 前置规则与后处理规则（构造后只读，变更需重启生效）

前置规则（在模型推理之前拦截，优先级 L1 > L2 > L3）:
    L1 提取失败: 关键点大面积缺失 → 默认 UNKNOWN，不冒充 IDLE
    L2 不在工位: 人体中心不在 ROI 内 → 判 IDLE（需配置 ROI，差异 #11 多边形）
    L3 低运动:  有当前手部证据时允许动作模型判断；缺少观察时 UNKNOWN

后处理:
    - 多数投票（平滑窗口）
    - 段合并（短段并入相邻段，按动作类型容差）
    - 置信度阈值（限时显示保持，estimated 不作为连续计时证据）

参数来源（差异 #10/#13）：
    - 契约命名 `pre_rules.l1_threshold` / `l3_idle_threshold` / `post_rules.*`，
      由 service.yaml 提供（见 docs/06 6.2.8/6.2.9）
    - 动作开关 `action_model.enabled_actions`（差异 #13），None/空 = 全部启用

后处理参数唯一来源为 `post_rules`（`vote_window` / `conf_threshold` /
`merge_tolerance` / `min_action_duration`）；`task_defaults.*` 同义项不再读取。
"""

import numpy as np
import time
from collections import Counter, deque

from common.label_catalog import IDLE_INDEX, tag_to_index

# Observation failure is not a new learned action class and is never counted as IDLE.
UNKNOWN_INDEX = -1


# 各模式下 total_motion 在特征向量中的索引
# mid: pose(57)+hands(126)+derived(9)=192 起，velocity 块内偏移 10 → 202
MOTION_IDX = {'simplified': 72, 'full': 244, 'mid': 202}
# pose 关键点区域结束位置（用于判断"无人"）
POSE_END = {'simplified': 57, 'full': 99, 'mid': 57}


def normalize_roi_to_unit(roi_points, resolution):
    """将分辨率坐标的 ROI 顶点归一化到 0..1（与 _body_center 的归一化坐标一致）。

    roi_points: ``[[x,y], ...]`` 或扁平矩形 ``[x1,y1,x2,y2]``；
    resolution: ``'1920x1080'`` 等。已归一化（所有坐标最大值 ≤ 1）或分辨率不可解析时原样返回。
    """
    if not roi_points:
        return None
    # 收集所有数值坐标，判断是否已归一化（最大值 ≤ 1）
    coords = []
    for p in roi_points:
        if isinstance(p, (list, tuple)):
            coords.extend(float(c) for c in p[:2])
        elif isinstance(p, (int, float)):
            coords.append(float(p))
    if coords and max(coords) <= 1.0:
        return roi_points  # 已归一化，原样返回
    try:
        w, h = (int(x) for x in str(resolution).lower().split('x'))
    except Exception:
        return roi_points
    if w <= 0 or h <= 0:
        return roi_points
    # 扁平矩形 [x1,y1,x2,y2]
    if len(roi_points) == 4 and all(isinstance(v, (int, float)) for v in roi_points):
        x1, y1, x2, y2 = (float(v) for v in roi_points)
        return [x1 / w, y1 / h, x2 / w, y2 / h]
    out = []
    for p in roi_points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([float(p[0]) / w, float(p[1]) / h])
    return out


class Rules:
    """规则集合（构造后只读；变更需重启推理服务生效）"""

    def __init__(self, rules_cfg=None, task_cfg=None,
                 pre_rules=None, post_rules=None,
                 enabled_actions=None, merge_tolerance=None,
                 enable_duration_feat=True, enable_visibility_mask=True):
        """两种构造方式：
        1) 兼容旧 service.yaml：传入 RulesConfig（rules_cfg）
        2) 目标 service.yaml：传入 service.yaml 的 PreRules / PostRules（差异 #10/#13）

        task_cfg 仅保留向后兼容签名，不再提供后处理参数（已合并到 post_rules）。
        merge_tolerance：可选显式覆盖（如任务级下发），否则取 post_rules.merge_tolerance。
        """
        # 旧配置兜底默认值（task_cfg 不再参与取值，见类docstring的参数合并说明）
        rc = rules_cfg
        self.l1_enabled = True
        self.l1_missing_threshold = 0.7
        self.l2_enabled = False
        self.l2_roi = None
        self.l3_enabled = True
        self.l3_motion_threshold = 0.01
        self.idle_min_frames = 8

        self.smooth_window = 5
        self.confidence_threshold = 0.6
        self.merge_tolerance = 10
        self.min_action_duration = 0.0
        self.merge_tolerance_frames = {}
        self.missing_pose_action = 'unknown'
        self.l3_mode = 'contextual'
        self.low_conf_hold_seconds = .2
        self.require_hand_for_cleaning = True

        if pre_rules is not None:
            self.l1_enabled = True
            self.l1_missing_threshold = pre_rules.l1_threshold
            self.l3_motion_threshold = pre_rules.l3_idle_threshold
            self.idle_min_frames = pre_rules.l3_min_frames
            self.missing_pose_action = getattr(pre_rules, 'missing_pose_action', 'unknown')
            self.l3_mode = getattr(pre_rules, 'l3_mode', 'contextual')
            # L2 默认启用（有 ROI 才生效，见 pre_check）
            self.l2_enabled = True
        elif rc is not None:
            self.l1_enabled = rc.l1_enabled
            self.l1_missing_threshold = rc.l1_missing_threshold
            self.l2_enabled = rc.l2_enabled
            self.l2_roi = rc.l2_roi
            self.l3_enabled = rc.l3_enabled
            self.l3_motion_threshold = rc.l3_motion_threshold
            self.idle_min_frames = rc.idle_min_frames

        if post_rules is not None:
            self.smooth_window = post_rules.vote_window
            self.confidence_threshold = post_rules.conf_threshold
            self.merge_tolerance = post_rules.merge_tolerance
            self.min_action_duration = post_rules.min_action_duration
            self.merge_tolerance_frames = dict(post_rules.merge_tolerance_frames or {})
            self.low_conf_hold_seconds = max(0., float(getattr(post_rules, 'low_conf_hold_seconds', .2)))
            self.require_hand_for_cleaning = bool(getattr(post_rules, 'require_hand_for_cleaning', True))

        if merge_tolerance is not None:
            self.merge_tolerance = merge_tolerance

        self.enable_duration_feat = enable_duration_feat
        self.enable_visibility_mask = enable_visibility_mask

        # 动作开关（差异 #13）：enabled_action_names 为配置的名称列表（None/空=全部启用）
        self.enabled_action_names = (
            list(enabled_actions) if enabled_actions else None)
        self._enabled_ids = None

        # 状态（每任务独立，禁止跨任务共享）
        self._idle_frames = 0
        self._pred_history = deque(maxlen=max(1, self.smooth_window))
        self._last_stable_action = IDLE_INDEX
        self._roi = (0.0, 0.0, 1.0, 1.0)
        self._roi_poly = None  # 多边形顶点（归一化坐标）；None=矩形/全画面
        self.last_precheck_reason = None
        self.last_decision_state = 'unknown'
        self.last_decision_reason = 'not_observed'
        self._last_confident_time = None
        self._last_timestamp = None

    # ---------------- 动作开关 ----------------

    def bind_label_map(self, label_map: dict):
        """绑定模型标签映射，把 enabled_action_names 解析为【契约索引】集合。

        未启用的动作在 post_process 中一律映射为 IDLE；IDLE 恒启用。
        enabled_action_names 为 None/空时 = 全部启用（_enabled_ids=None）。

        ⚠ 必须使用契约索引空间（label_catalog.yaml 顺序，IDLE 恒为 0）：
        模型包 label_map 的输出索引按训练实际类别压缩（如 3 类模型 0=MOVING、
        无 IDLE），与契约索引不同空间；此前用模型输出索引解析动作开关 +
        pre_check 返回契约 IDLE_INDEX 混用，会把"未启用→IDLE"错映射到模型首类。
        """
        if not self.enabled_action_names:
            self._enabled_ids = None
            return
        ids = set()
        for n in self.enabled_action_names:
            idx = tag_to_index(n)
            if idx is not None:
                ids.add(idx)
        ids.add(IDLE_INDEX)   # IDLE 恒启用
        self._enabled_ids = ids

    def to_dict(self) -> dict:
        return {
            'pre_rules.l1_threshold': self.l1_missing_threshold,
            'pre_rules.l3_idle_threshold': self.l3_motion_threshold,
            'pre_rules.l3_min_frames': self.idle_min_frames,
            'post_rules.vote_window': self.smooth_window,
            'post_rules.conf_threshold': self.confidence_threshold,
            'post_rules.merge_tolerance': self.merge_tolerance,
            'post_rules.min_action_duration': self.min_action_duration,
            'enabled_actions': self.enabled_action_names,
            'pre_rules.missing_pose_action': self.missing_pose_action,
            'pre_rules.l3_mode': self.l3_mode,
            'post_rules.low_conf_hold_seconds': self.low_conf_hold_seconds,
            'post_rules.require_hand_for_cleaning': self.require_hand_for_cleaning,
        }

    # ---------------- 前置规则 ----------------

    def pre_check(self, feat: np.ndarray, mode: str, results=None, hand_evidence=None):
        """前置规则判定

        Returns:
            action_id 或 None（None 表示未拦截，继续走模型推理）
        """
        self.last_precheck_reason = None
        pose_end = POSE_END.get(mode, 57)
        hand_seen = self._hand_supported(hand_evidence)

        # L1 无人：关键点大面积缺失
        if self.l1_enabled:
            pose = feat[:pose_end].reshape(-1, 3)
            zero_ratio = float(np.mean(np.all(np.abs(pose) < 1e-6, axis=1)))
            if zero_ratio >= self.l1_missing_threshold:
                # L1 立即拦截，不累加 L3 连续静止计数器（两者独立语义，见 docs/06 §6.2.8）
                self.last_precheck_reason = 'pose_missing_hands_observed' if hand_seen else 'pose_extraction_unavailable'
                self._pred_history.clear()
                self._last_stable_action = UNKNOWN_INDEX
                self._last_confident_time = None
                return IDLE_INDEX if self.missing_pose_action == 'legacy_idle' else UNKNOWN_INDEX

        # L2 不在工位：人体中心不在 ROI 内（差异 #11，支持多边形）
        if self.l2_enabled and self.l2_roi:
            center = self._body_center(feat, mode)
            if center is not None and not self._in_roi(center, self._roi, self._roi_poly):
                if not self._hand_in_work_area(hand_evidence):
                    self.last_precheck_reason = 'body_and_observed_hand_outside_work_area'
                    return IDLE_INDEX

        # L3 休息：连续低运动
        if self.l3_enabled:
            motion_idx = MOTION_IDX.get(mode, 72)
            motion = abs(float(feat[motion_idx])) if feat.shape[0] > motion_idx else 0.0
            if motion < self.l3_motion_threshold:
                self._idle_frames += 1
            else:
                self._idle_frames = 0

            if self._idle_frames >= self.idle_min_frames:
                if self.l3_mode == 'legacy_idle':
                    self.last_precheck_reason = 'legacy_low_motion_idle'
                    return IDLE_INDEX
                if self.l3_mode == 'contextual' and not hand_seen:
                    self.last_precheck_reason = 'low_motion_without_hand_observation'
                    return UNKNOWN_INDEX
                self.last_precheck_reason = 'low_motion_with_hand_support_allow_classifier'

        return None

    @staticmethod
    def _hand_supported(evidence):
        return bool(evidence and evidence.get('present') is True
                    and evidence.get('fresh', True)
                    and evidence.get('observation_state', 'observed') == 'observed')

    def _hand_in_work_area(self, evidence):
        if not self._hand_supported(evidence):
            return False
        for hand in evidence.get('hands', []):
            points = hand.get('xy_normalized') or []
            if points and any(self._in_roi(p, self._roi, self._roi_poly) for p in points):
                return True
        return False

    @staticmethod
    def _body_center(feat: np.ndarray, mode: str):
        """人体中心（肩/髋均值），返回 (x, y) 归一化坐标"""
        idx = (11, 12, 23, 24) if mode == 'full' else (11, 12, 17, 18)
        pts = []
        for i in idx:
            base = i * 3
            if base + 1 < feat.shape[0]:
                pts.append((float(feat[base]), float(feat[base + 1])))
        if not pts:
            return None
        return (sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts))

    def set_roi(self, roi):
        """设置 ROI（差异 #11）。

        支持两种形态：
          - 矩形：``[x1,y1,x2,y2]``（4 个数值）→ 走 AABB 快速判定；
          - 多边形：``[[x,y], ...]``（≥3 个顶点）→ 走射线法精确判定。
        坐标须为归一化 0..1（与 _body_center 一致）；分辨率坐标请先用
        ``normalize_roi_to_unit`` 归一化。``roi`` 为空 = 全画面（L2 关闭）。
        """
        if not roi:
            self._roi = (0.0, 0.0, 1.0, 1.0)
            self._roi_poly = None
            self.l2_roi = False
            return
        if len(roi) == 4 and all(isinstance(v, (int, float)) for v in roi):
            self._roi = tuple(float(v) for v in roi)
            self._roi_poly = None
        else:
            # 多边形：保存顶点用于精确判定，并预计算包围盒做快速拒绝
            pts = [(float(p[0]), float(p[1])) for p in roi]
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            self._roi = (min(xs), min(ys), max(xs), max(ys))
            self._roi_poly = pts
        self.l2_roi = True

    @staticmethod
    def _point_in_polygon(x: float, y: float, poly) -> bool:
        """射线法判断点 (x,y) 是否在多边形 poly（[(x,y),...]）内。"""
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _in_roi(center, roi, poly=None) -> bool:
        x, y = center
        x1, y1, x2, y2 = roi
        # 先包围盒快速拒绝
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return False
        if poly:
            return Rules._point_in_polygon(x, y, poly)
        return True  # 矩形（roi 即为矩形）

    # ---------------- 后处理 ----------------

    def post_process(self, action_id: int, confidence: float, timestamp_s=None, hand_evidence=None):
        """多数投票 + 置信度阈值

        Returns:
            (action_id, confidence, changed)
        """
        now = time.monotonic() if timestamp_s is None else float(timestamp_s)
        if self._last_timestamp is not None and now <= self._last_timestamp:
            self.reset()
            self._last_timestamp = now
            self.last_decision_reason = 'nonincreasing_observation_timestamp'
            return UNKNOWN_INDEX, float(confidence), True
        self._last_timestamp = now
        if not np.isfinite(confidence) or confidence < self.confidence_threshold:
            age = None if self._last_confident_time is None else now-self._last_confident_time
            if age is not None and 0 <= age <= self.low_conf_hold_seconds and self._last_stable_action != UNKNOWN_INDEX:
                self.last_decision_state = 'estimated'
                self.last_decision_reason = 'bounded_low_confidence_display_hold'
                return self._last_stable_action, float(confidence), False
            changed = self._last_stable_action != UNKNOWN_INDEX
            self._pred_history.clear()
            self._last_stable_action = UNKNOWN_INDEX
            self.last_decision_state = 'unknown'
            self.last_decision_reason = 'low_confidence_hold_expired'
            return UNKNOWN_INDEX, float(confidence), changed

        clean_ids = {tag_to_index('WATER'), tag_to_index('GAS')}
        if self.require_hand_for_cleaning and action_id in clean_ids and not self._hand_supported(hand_evidence):
            self._pred_history.clear()
            self._last_stable_action = UNKNOWN_INDEX
            self._last_confident_time = None
            self.last_decision_state = 'unknown'
            self.last_decision_reason = 'cleaning_label_without_current_hand_evidence'
            return UNKNOWN_INDEX, float(confidence), True
        self._last_confident_time = now

        self._pred_history.append(action_id)
        if len(self._pred_history) >= max(1, self.smooth_window // 2 + 1):
            voted = Counter(self._pred_history).most_common(1)[0][0]
        else:
            voted = action_id

        # 动作开关：未启用的动作一律按 IDLE 处理（不产生该动作的段/事件）
        if self._enabled_ids is not None and voted not in self._enabled_ids:
            voted = IDLE_INDEX

        changed = (voted != self._last_stable_action)
        if changed:
            self._last_stable_action = voted

        self.last_decision_state = 'observed' if voted == action_id else 'estimated'
        self.last_decision_reason = 'current_model_observation' if voted == action_id else 'vote_transition_display_only'

        return voted, confidence, changed

    def invalidate_prediction_history(self):
        self._pred_history.clear()
        self._last_stable_action = UNKNOWN_INDEX
        self._last_confident_time = None

    def reset(self):
        self._idle_frames = 0
        self._pred_history.clear()
        self._last_stable_action = IDLE_INDEX
        self._last_confident_time = None
        self._last_timestamp = None
        self.last_precheck_reason = None
        self.last_decision_state = 'unknown'
        self.last_decision_reason = 'reset'
