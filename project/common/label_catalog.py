"""
common/label_catalog.py - 动作标签契约全集加载器（唯一事实源：config/label_catalog.yaml）

标签编码、中文名、颜色、索引全部来自 config/label_catalog.yaml（无硬编码标签定义），
训练 / 推理 / 业务三端共享同一份契约（见 docs/05-训练服务设计说明.md §十一 模型包规范）：

  - 训练侧：label_dirs 键合法性校验（tag_to_index），训练标签整数 = catalog 顺序索引
  - 推理侧：模型包 label_map.id 集合校验（须为契约子集，支持部分类模型）、
    label_map 缺省时的展示兜底（name_of）
  - 业务侧：jl_action_type 字典、jc_sop_item.action_type 引用 catalog 编码

约定：
  - IDLE 恒为索引 0（catalog 校验强制）——前置规则 / 后处理兜底按索引 0 输出 IDLE
  - 动作编码（id 字符串）是上下游契约的唯一权威（record_action.action_type、
    jl_action_type 字典、service.yaml / model_info.json 的 label_map 均用此编码）
  - 历史模型包兼容（旧 4/8 类整数标签、CLEAN_WATER 等历史 id）不在 catalog 中，
    由 LEGACY_ID_MAP / _LEGACY_INDEX_NAMES 提供
"""

from __future__ import annotations
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 契约文件路径（单一来源，相对路径以工程根 aicv-jl/ 为基准）。
# 所有消费方（训练/推理/业务工具）必须经本模块访问契约，禁止各自另行定位文件。
CATALOG_PATH = os.path.join(ROOT, 'config', 'label_catalog.yaml')

_ID_PATTERN = re.compile(r'^[A-Z][A-Z0-9_]*$')

# 历史 4/8 类整数标签 → 编码（兼容旧 .pth 内嵌 label_map / 旧特征缓存，勿删）
_LEGACY_INDEX_NAMES = [
    'IDLE', 'WATER', 'GAS', 'MOVING',
    'ARRANGING', 'SURFACE_CLEAN', 'SHELF_TAKE', 'SHELF_PUT',
]

# 历史 id 字符串 → 目标编码（加载早期模型包 label_map 时归一）
LEGACY_ID_MAP = {
    'CLEAN_WATER': 'WATER',
    'CLEAN_GAS': 'GAS',
    'PLACE_BACK': 'SHELF_PUT',
}


class LabelCatalogError(Exception):
    """标签契约全集加载/校验失败"""


def _load(catalog_path: str = None) -> dict:
    """加载并校验 label_catalog.yaml，返回 {labels, ids, name_zh, color_hex}"""
    path = catalog_path or CATALOG_PATH
    if not os.path.exists(path):
        raise LabelCatalogError(f'标签契约全集文件不存在: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    labels = raw.get('labels') or []
    if not labels:
        raise LabelCatalogError('label_catalog.yaml 的 labels 为空')

    ids, name_zh, color_hex = [], {}, {}
    for i, item in enumerate(labels):
        lid = str(item.get('id', '')).strip()
        if not _ID_PATTERN.match(lid):
            raise LabelCatalogError(f'第 {i + 1} 个标签 id 非法: {lid!r}（须匹配 {_ID_PATTERN.pattern}）')
        if lid in ids:
            raise LabelCatalogError(f'标签 id 重复: {lid}')
        ids.append(lid)
        name_zh[lid] = str(item.get('name_zh') or lid)
        color_hex[lid] = str(item.get('color') or '#909399')

    if ids[0] != 'IDLE':
        raise LabelCatalogError(f'第一个标签必须为 IDLE（索引 0），实际: {ids[0]}')
    if 'IDLE' not in ids:
        raise LabelCatalogError('契约全集必须包含 IDLE 背景类')

    return {'labels': labels, 'ids': ids, 'name_zh': name_zh, 'color_hex': color_hex}


_catalog = _load()

# ============================================================
# 契约访问器（模块级常量，启动时加载一次；变更需重启）
# ============================================================

LABELS = _catalog['labels']                # 原始条目列表 [{id, name_zh, color, ...}]
LABEL_IDS = _catalog['ids']                # 有序编码列表（IDLE 恒为首）
NUM_CLASSES = len(LABEL_IDS)
NAME_ZH = _catalog['name_zh']              # {id: 中文名}
COLOR_HEX = _catalog['color_hex']          # {id: HEX 颜色}

IDLE_INDEX = 0                             # IDLE 恒为 0（_load 已强制校验）
ID_TO_INDEX = {aid: i for i, aid in enumerate(LABEL_IDS)}
INDEX_TO_ID = {i: aid for i, aid in enumerate(LABEL_IDS)}

# ============================================================
# 标签契约变更 ↔ 特征缓存失效 对应规则（维护必读）
# ------------------------------------------------------------
# 训练特征缓存（output/train_features/cache_<mode>/*.npz、clean_*.npz）以
# 「源视频名 + sr」命名，**不含标签版本号**。缓存失效分两类：
#   · 自动失效（无需操作）：
#       - 更改 sample_rate：逐视频缓存与聚合缓存 clean_*_srN_*.npz 均含 sr，改名即失效；
#       - 更改 aug_version：代码严格校验（不一致直接报错退出）且进序列/模型文件名。
#   · 不会自动失效，须手动清空 / --reextract 的情形：
#       ① 调整本文件 labels 中【已有标签的顺序/位置】（中间插入、删除、重排）；
#       ② 更改某动作的契约编码 id 或动作语义。
#   · 无需清空的情形：
#       ③ 仅在末尾追加新标签、或仅改 name_zh/颜色（整数映射不变；NUM_CLASSES 变化需重训模型）；
#       ④ 新增视频目录/新视频会被增量逻辑自动重提。
#   强制重提方式：删 output/train_features/ 整目录，或训练加 CLI 参数 --reextract。
# 整数标签 = 本文件 labels 列表中的排列位置（IDLE 恒为 0），见下方 ID_TO_INDEX。

VERSION = str(_catalog.get('version', ''))


def index_of(action_id: str) -> int:
    """动作编码 → 契约索引（未知返回 -1）"""
    return ID_TO_INDEX.get(action_id, -1)


def id_of(index: int) -> str:
    """契约索引 → 动作编码（越界返回 'IDLE'）"""
    return INDEX_TO_ID.get(int(index), 'IDLE')


def tag_to_index(tag: str):
    """契约编码 → 契约索引；未知返回 None（训练侧据此跳过未知 label_dirs 键）"""
    return ID_TO_INDEX.get(str(tag).strip().upper())


def label_zh(index: int) -> str:
    """契约索引 → 中文名（越界返回字符串形式的索引，展示兜底不抛异常）"""
    i = int(index)
    if 0 <= i < len(LABEL_IDS):
        return NAME_ZH[LABEL_IDS[i]]
    return str(i)


def legacy_index_to_id(index: int) -> str:
    """历史 4/8 类整数标签 → 契约编码（旧 .pth / 旧特征缓存兼容）"""
    i = int(index)
    if 0 <= i < len(_LEGACY_INDEX_NAMES):
        return LEGACY_ID_MAP.get(_LEGACY_INDEX_NAMES[i], _LEGACY_INDEX_NAMES[i])
    return str(index)


# ============================================================
# label_map 工具（供训练导出 / 推理加载共用）
# ============================================================


def normalize_label_map(label_map: dict) -> dict:
    """将任意格式 label_map 规整为 ``{idx: canonical_id}``。

    兼容两种存储形态：``{idx: id_str}`` 与 ``{idx: {id, name, color}}``；
    并将历史 4 类 id（CLEAN_WATER/CLEAN_GAS/PLACE_BACK）映射为目标编码。
    无法识别的 id 回落为原始值（保证索引不丢，交由模型加载校验报错）。
    详见 docs/05-训练服务设计说明.md §十一 模型包规范。
    """
    out = {}
    for idx, entry in (label_map or {}).items():
        raw = entry.get('id') if isinstance(entry, dict) else entry
        if raw in ID_TO_INDEX:
            out[int(idx)] = raw
        elif raw in LEGACY_ID_MAP:
            out[int(idx)] = LEGACY_ID_MAP[raw]
        else:
            out[int(idx)] = raw
    return out


def build_contract_label_map(unique_indices: list) -> dict:
    """由训练侧实际训练的标签索引集合构建富格式 label_map。

    Args:
        unique_indices: 训练标签整数（= 契约索引）的集合/列表
                        （如 {0, 4, 6} 表示 IDLE/WATER/GAS 参与训练）。

    Returns:
        ``{输出索引(str): {id, name, color}}``：
          - 输出索引 = 契约索引升序集合中的下标（与训练侧 label_map 一致）；
          - name/color 取自 label_catalog.yaml。
    """
    out = {}
    for new_idx, orig in enumerate(sorted(unique_indices)):
        tag = id_of(orig)
        out[str(new_idx)] = {
            'id': tag,
            'name': NAME_ZH.get(tag, tag),
            'color': COLOR_HEX.get(tag, '#909399'),
        }
    return out


def name_of(action_id: int, label_map: dict | None = None) -> str:
    """动作索引 → 名称。

    - 给定 label_map（模型包 {idx: id} 或 {idx: {id,name}}）→ 返回动作编码/id；
    - 否则回退到契约全集 catalog。
    """
    if label_map is not None and int(action_id) in label_map:
        entry = label_map[int(action_id)]
        if isinstance(entry, dict):
            return entry.get('id', 'IDLE')
        return entry
    if 0 <= int(action_id) < len(LABEL_IDS):
        return LABEL_IDS[int(action_id)]
    return 'UNKNOWN'


def id_set(label_map: dict) -> set:
    """从 label_map 提取动作编码集合（兼容 {idx:id} 与 {idx:{id,...}}）"""
    s = set()
    for entry in label_map.values():
        if isinstance(entry, dict):
            s.add(entry.get('id'))
        else:
            s.add(entry)
    return s
