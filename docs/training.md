# 数据准备、服务器训练与评估

## 选择要训练的模块

| 模块 | 可复现的训练源码 | 初始化与训练范围 |
| --- | --- | --- |
| R18 手指姿态 | [training/r18_pose](../training/r18_pose/) | Hand5 官方纯张量父权重加 R17 姿态增量。冻结早期骨干与 BN，训练 stage4 与 RTMCC 头。对照常规增强和窄条遮挡 |
| R19 主手框 | [training/r19_detector](../training/r19_detector/) | COCO Faster R-CNN R50-FPN-v2。冻结 ResNet 主干和 BN，训练 FPN、RPN 和 ROI 头。对照只用公共数据和混入原工位手框 |
| R26 宏观作业 | [training/r26_action](../training/r26_action/) | R25 动作父权重及对齐后的特征缓存。冻结感知编码器，训练因果动作头。辅助头采用作业边界区间监督，当前选用仅动作细训的导出头 |
| R29 漏手检测 | [training/r29_fallback](../training/r29_fallback/) | 从 R19 当前检测器适配戴手套重叠和遮挡手框。与 R19 共用固定底座，导出可变层，进入完全无手的补位分支 |
| R31 时序姿态对照 | [training/research/r31](../training/research/r31/) | 保存技术研究源码。两组各 1280 步，在当前已打开的 49 个可见指尖上均未优于 R18 |

每个训练目录中的 `config.reference.json` 保留历史参数和数据、父权重摘要。路径已参数化为 `$TRAIN_ROOT`。运行环境的进程、显卡资源记录和共享锁由服务器提供。现有训练配方均在项目的 Linux CUDA 环境使用。推理 Release 的六份文件用于前向，续训从对应轮次的完整可变状态与冻结父权重恢复。

## 让数据装载器拿到正确的输入

- **姿态。** `r18_pose/mix_data.py` 读取 `data_root/manifest.jsonl` 和 `dataset.json`，每行给出 `id`、`split`、`image_location`、`image`、`keypoints_xy_256`、`joint_weights`、`normalizer_pixels`、`sigma_pixels`。图像为 256×256 RGB 裁剪，关节顺序与 Hand5 手部 21 点一致。遮挡而无法确认的点置权重 0。历史批次为 MC 16、FreiHAND 12、现场已知点 4。配置中的 `mc_root`、`r17_data_root` 和 `data_root` 分别解析 `mc_parent`、`r17_parent` 和 `current`。
- **手框。** `r19_detector/detection_data.py` 用同类图像引用和 `boxes_xyxy`、`image_sha256`。只画可见手框。HaGRID 的作者框可用于检测。旧标签里自动估计的手点仅供定位线索。R29 的数据入口位于 `r29_fallback/data.py`，读取新旧难例分组和各自 SHA。R29 的下游开发回放需要原样的工位图像及 R18、R26 特征约定。
- **作业。** `r26_action/data.py` 用相同源 PTS 对齐 `context.npz`、手部缓存及 `labels.npz`。每时刻为 RGB 上下文 1024 维、最多 6 个手候选各 816 维、两个观察掩码和 `delta_pts`。有边界监督的配方还要给出 `fine_boundary_bags.json`，用区间表示无法精确到唯一一帧的边界。原预训练短片缓存的来源、文件与 SHA 在 `dataset.json` 中核验。

现有历史数据实际使用了 MC-hands Big 的 5000 个训练裁剪、FreiHAND 的 1024 个训练样本和现场可见指尖。HaGRID 子集用于手框。公开全集、现场媒体和标签原件在各自授权的数据目录。恢复历史配置时按模板中对应的 SHA 找原件。新采样按原视频、捆次、操作者和日期分组，同源裁图与邻帧归同组。公开数据的来源、标注类型和使用条件见 [许可和数据来源](../licenses/NOTICE.md)。

## 在服务器建立新的实验

1. 通过项目既有受控通道暂存新的源码目录、实际训练数据及父权重。为每次实验取新名称，并使用新的 `src/<intent>`、`configs/<intent>.json`、`runs/<intent>`。服务端使用项目指定的 `env/bin/python`。源码目录应保留训练文件的同级相对位置。
2. 以对应 `config.reference.json` 为起点。把实际数据目录、父权重和 SHA、训练计划、源码包及显卡分配写到服务器侧 `overrides.json`。R18、R19 的源码包和训练数据包路径也要更新。R26、R29 填本次资源检查得到的 `co_residents` 和 `gpu_uuid_sha256`。
3. 生成配置并冻结源文件、数据、训练计划、父状态与配置的 SHA。按既有受控安装流程生成本次 `receipts/<intent>_installation.json`。R18 与 R19 的安装回执验证源码包 `source_sha256` 和配置 `config_sha256`，协调器从 `<intent>_intent.json` 读双卡分配。R26 用 `source_members` 与配置的 `data_files`，R29 用 `source_members` 和安装回执的 `data_members`，成员记录均含文件名和 SHA256。
4. 取得平台的全局和每卡共享锁，连续检查三次 GPU 占用，每次间隔至少 30 秒。协调器以当前资源分配启动短测和主阶段，子进程继承锁。分别核对磁盘余量、显存上限与最新进程记录。双 GPU 历史配方使用其原协调器声明的物理卡号，换卡时在新配置和协调器中同步更新。

以下示例以 R18 为例，变量取本次已批准的真实项目路径和资源文件。准备配置只写文件。

```bash
export HOSE_TRAIN_ROOT=/path/to/approved/server/project
export HOSE_INTENT=r18_pose_replay_new
export HOSE_GLOBAL_LOCK=/path/to/existing/shared/dispatch.lock
export HOSE_RESOURCE_BASELINE=/path/to/current/resource-baseline.json
PY="$HOSE_TRAIN_ROOT/env/bin/python"

"$PY" training/prepare_config.py \
    --recipe r18_pose --train-root "$HOSE_TRAIN_ROOT" --intent "$HOSE_INTENT" \
    --overrides /path/to/overrides.json \
    --output "$HOSE_TRAIN_ROOT/configs/$HOSE_INTENT.json"
export HOSE_CONFIG_SHA256="$(sha256sum "$HOSE_TRAIN_ROOT/configs/$HOSE_INTENT.json" | cut -d ' ' -f1)"
cd "$HOSE_TRAIN_ROOT/src/$HOSE_INTENT"
"$PY" -B -c 'from runtime import ensure_frozen; ensure_frozen()'
```

确认安装回执、配置、数据与资源后，通过既有调度入口用同一环境运行 `"$PY" -B -u coordinator.py`。R19 同样使用 `ensure_frozen()`。R26、R29 另设 `HOSE_DATA_ROOT` 和 `HOSE_RESOURCE_HELPER`。后者指向服务端现有、已核验的共享资源帮助模块，提供全局锁引用、进程身份和锁 FD 校验。两个配方用 `from runtime import ensure; ensure()` 检查安装。

## 短测、恢复和选型

先跑 16 步完整保存，再由新进程恢复模型可变层、优化器、调度器、随机数与采样游标。通过输出探针才续到声明步数。16 步纳入本轮预算。中断后先核对进程、最近完整检查点和写盘记录。报告里把实跑步数、完整恢复步数和重新计算成本逐项列出。

| 模块 | 历史主比较口径 |
| --- | --- |
| R18 | 每组 3200 步。当前使用常规增强组的开发集第 400 步，保留 21 点输入与已知点平衡 |
| R19 | 固定共同阈值，比较公共开发框和跨日期现场框，分别列漏框和误检 |
| R26 | 每组 800 步并按最后步比较。同输入报告作业 P/R/F1、迟滞、候选数和拒判覆盖 |
| R29 | 对困难可见框和自动回放做同阈值对照，检查完全漏手是否被补回以及额外误检 |

## 训练下一阶段的接口分支

先按 [现场试点](plan.md) 拿到同机位连续录像与独立复核的接入、退出和快换标注。沿用当前手部网络和 R26 作为冻结特征。准备带源 PTS 的管端、工具口局部原图、观测位及状态标签。先做手部与上下文对照，再在相同样本和预算上加接口局部图，最后比较独立进出边界监督。遮挡和临界时长标未知，真实短断开要保留两次事件。新接口模型的质量用整段留出录像上的一对一匹配、相邻换管拆分、误计、漏计、覆盖和误差验收。
