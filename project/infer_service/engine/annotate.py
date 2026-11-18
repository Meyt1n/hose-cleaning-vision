"""
infer_service/engine/annotate.py - 标注帧绘制

画面与标注在同一帧上绘制后输出，天然同步，不存在"画面与绘制错位"问题。

文本绘制说明：
  - cv2.putText 不支持中文（中文一律画成 ???），因此统一走
    Pillow + 系统中文字体（微软雅黑/黑体/宋体，按候选顺序探测）；
    字体/依赖不可用时回落 cv2.putText 并将非 ASCII 替换（避免乱码"???"）。
  - 信息面板底色为半透明（35% 黑），不再整块遮挡画面内容。
  - 面板/字号随输出宽度自适应（以 ws.frame_width=640 为基准缩放）。
"""

from __future__ import annotations
import os

import cv2
import numpy as np

# ---------------- 中文文本绘制（Pillow + 系统字体） ----------------

_FONT_CANDIDATES = [
    r'C:\Windows\Fonts\msyh.ttc',       # 微软雅黑
    r'C:\Windows\Fonts\simhei.ttf',     # 黑体
    r'C:\Windows\Fonts\simsun.ttc',     # 宋体
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]
_font_cache: dict = {}
_font_path: str | None = None
_font_path_checked = False


def _find_font_path() -> str | None:
    global _font_path, _font_path_checked
    if not _font_path_checked:
        _font_path_checked = True
        for p in _FONT_CANDIDATES:
            if os.path.exists(p):
                _font_path = p
                break
    return _font_path


def _pil_fonts(sizes) -> dict | None:
    """按像素字号加载 Pillow 字体（带缓存）；依赖/字体不可用时返回 None。"""
    path = _find_font_path()
    if not path:
        return None
    try:
        from PIL import ImageFont
    except Exception:
        return None
    for s in sizes:
        if s not in _font_cache:
            try:
                _font_cache[s] = ImageFont.truetype(path, int(s))
            except Exception:
                return None
    return {s: _font_cache[s] for s in sizes}


def _draw_texts(frame, lines):
    """统一文本绘制层。

    lines: [(x, y_top, text, bgr_color, px_size), ...]
    有系统字体 → Pillow 绘制（支持中文）；否则回落 cv2.putText
    并将非 ASCII 字符替换为 '?'（cv2 画中文必然乱码）。
    """
    fonts = _pil_fonts({s for _, _, _, _, s in lines}) if lines else None
    if fonts:
        from PIL import Image, ImageDraw
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for x, y, t, c, s in lines:
            d.text((int(x), int(y)), t, font=fonts[s],
                   fill=(int(c[2]), int(c[1]), int(c[0])))
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    for x, y, t, c, s in lines:
        t = t.encode('ascii', 'replace').decode('ascii')
        cv2.putText(frame, t, (int(x), int(y) + int(s)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.35, s / 22.0), c, 1, cv2.LINE_AA)
    return frame


def hex_to_bgr(hex_color):
    """HEX(#RRGGBB) -> OpenCV BGR 元组；非法返回白色"""
    h = (hex_color or '').lstrip('#')
    if len(h) == 6:
        try:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return (b, g, r)
        except ValueError:
            pass
    return (255, 255, 255)


def build_label_colors(label_map: dict) -> dict:
    """由模型包 label_map（动作码契约 id↔name↔color，随 action_model.path 加载）构建 动作标识 -> BGR 颜色映射。

    label_map 结构: {idx: {'id': 'WATER', 'name': '水洗', 'color': '#409EFF'}, ...}
    颜色统一取自配置 label_map.color（见 docs/06 5.4），不再硬编码。
    同时以**契约编码**（WATER）与**中文名**（水洗）为键——draw_overlay 两种入参均可查中，
    防止键类型不匹配导致颜色恒回落白色。
    """
    out = {}
    for entry in (label_map or {}).values():
        if isinstance(entry, dict) and entry.get('color'):
            bgr = hex_to_bgr(entry['color'])
            if entry.get('id'):
                out[entry['id']] = bgr
            if entry.get('name'):
                out[entry['name']] = bgr
    return out


def draw_overlay(frame, action_name: str, confidence: float, duration: float,
                 buffered: int = 0, seq_len: int = 30, alerts=None,
                 rule_hit: str = None, gap: bool = False,
                 draw_landmarks: bool = True, extractor=None, results=None,
                 boxes=None, extra_info: str = None, label_colors: dict = None,
                 action_code: str = None, panel_scale: float = None):
    """在帧上叠加检测信息（面板半透明、文字支持中文）

    Args:
        frame: BGR 帧
        action_name: 动作名（中文展示名）
        confidence: 置信度
        duration: 当前动作持续秒数
        buffered/seq_len: 滑窗进度
        alerts: 告警列表 [{'type','message'}]
        rule_hit: 命中的前置规则（L1/L2/L3）
        gap: 是否处于断流恢复状态
        draw_landmarks: 是否绘制骨架
        extractor/results: MediaPipe 提取器与结果（绘制骨架用）
        boxes: 目标检测框（原图像素坐标）
        label_colors: 动作标识 -> BGR 颜色（契约编码/中文名均可查）
    """
    h, w = frame.shape[:2]
    # 查色优先契约编码（WATER），回落中文名——保证模型包 label_map[].color 真正生效
    color = ((label_colors or {}).get(action_code)
             or (label_colors or {}).get(action_name)
             or (255, 255, 255))

    # 骨架（MediaPipe 归一化坐标，帧缩放前后均正确）
    if draw_landmarks and extractor is not None and results is not None:
        try:
            extractor.draw_landmarks(frame, results)
        except Exception:
            pass

    # 面板尺寸随输出宽度自适应（640 宽为基准 s=1；1280 → s=2）
    s = max(1.0, min(4.0, w / 640.0 if panel_scale is None else float(panel_scale)))
    panel_w = min(int(460 * s), w - 20)
    panel_h = int(150 * s)
    x1, y1 = 5 + panel_w, 5 + panel_h

    # 半透明底（黑 35%，不再整块遮挡画面）；边框实线取动作色
    roi = frame[5:y1, 5:x1]
    frame[5:y1, 5:x1] = cv2.addWeighted(roi, 0.65, np.zeros_like(roi), 0.35, 0)
    cv2.rectangle(frame, (5, 5), (x1, y1), color, max(1, int(1.5 * s)))

    white = (255, 255, 255)
    gray = (180, 180, 180)
    orange = (0, 165, 255)

    lines = [
        (int(15 * s), int(10 * s), f'Action: {action_name}', color, int(24 * s)),
        (int(15 * s), int(44 * s), f'Duration: {duration:.1f}s', white, int(18 * s)),
        (int(15 * s), int(68 * s), f'Confidence: {confidence:.2%}', white, int(18 * s)),
        (int(15 * s), int(92 * s),
         f'Buffer: {buffered}/{seq_len}'
         + (f' | Rule: {rule_hit}' if rule_hit else '')
         + (f' | {extra_info}' if extra_info else ''),
         gray, int(14 * s)),
    ]
    if gap:
        lines.append((int(15 * s), int(118 * s), '[GAP] Stream reconnecting',
                      orange, int(15 * s)))

    # 目标检测框（差异 #9）：框体直接画，标签进统一文本层（类名可能是中文）
    for det in (boxes or []):
        try:
            bx1, by1, bx2, by2 = [int(v) for v in det.get('box', [])]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 255),
                          max(1, int(1.5 * s)))
            lines.append((bx1, max(0, by1 - int(22 * s)),
                          f"{det.get('class_name', '')} {det.get('score', 0):.2f}",
                          (0, 200, 255), int(15 * s)))
        except Exception:
            pass

    # 告警条（底部，最多 3 条）
    ay = h - int(24 * s)
    for alert in (alerts or [])[-3:]:
        msg = alert.get('message', '') if isinstance(alert, dict) else str(alert)
        level = alert.get('type', 'WARNING') if isinstance(alert, dict) else 'WARNING'
        acolor = (0, 0, 255) if level == 'ERROR' else orange
        lines.append((int(10 * s), ay, f'[{level}] {msg}', acolor, int(15 * s)))
        ay -= int(22 * s)

    return _draw_texts(frame, lines)


def encode_jpeg(frame, quality: int = 85, resize_width: int = None) -> bytes:
    """编码为 JPEG 字节流；resize_width 指定时等比缩放宽度（瘦身省编码/带宽）

    注意：推送链路已改为"先缩放后叠加"（pipeline._push_frame），叠加后的帧
    不应再缩放（否则文字又不可读），本参数仅保留给其他调用方使用。
    """
    if resize_width and 0 < resize_width < frame.shape[1]:
        h, w = frame.shape[:2]
        new_h = max(1, int(h * resize_width / w))
        frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return b''
    return buf.tobytes()
