# 推理与现场操作

## 安装和模型

按 [README](../README.md) 准备 Python 环境和成对的 PyTorch、torchvision。代码迁移检查使用 Python 3.9.13、PyTorch 2.7.1、torchvision 0.22.1 的 CPU 前向路径。无 GitHub CLI 时，可登录私有仓库，通过 [Release 页面](https://github.com/Meyt1n/hose-cleaning-vision/releases/tag/v0.1.0) 下载 `inference-weights.zip`。

```text
python scripts/weights.py --archive inference-weights.zip
python scripts/weights.py
```

安装器先确认 ZIP 成员、逐文件字节数与 SHA256，再写入 `models/`。已有匹配文件会复用，发现内容冲突会报错。R19 和 R29 共用 COCO ResNet50 底座。两个检测器文件是各自的可变层增量。R18 是完整 Hand5 姿态推理权重。R26 需要配套的 ImageNet ResNet18 锁定特征提取器。

## 单图检查

```text
python run_hand_image.py --image "站位原图.png" --output artifacts/hand-01
```

输出目录需使用新名称。运行后查看 `prediction.png` 和 `predictions.json`。后者包含手框、21 个二维点的原图坐标、归一化坐标、检测分数和 R29 是否实际补到手。4000×3000 原图默认使用已校验的工位裁剪，例如换了相机位时可通过 `--roi "[0.2,0.1,0.6,0.8]"` 指定归一化 `[x,y,w,h]`。

## 连续视频离线回放

```text
python project/scripts/predict_local_action_video.py --source "工位原视频.mp4" --start 600 --duration 180 --output artifacts/action-01
```

`--start`、`--duration` 按原片秒数选取有效范围，输入为带 PTS 的工位原相机 4000×3000 录像。可先选择 2 至 5 秒检查依赖、ROI 与时间对齐。视频入口使用 `project/config/offline_action.json`，模型文件相对配置路径解析。更换相机视角时，先对齐 R26 使用的双 ROI 特征契约再做验证。窗口历史只读当前和过去帧，可用片段前的短预热。

| 输出 | 用途 |
| --- | --- |
| `prediction.mp4` | 手框、细点、作业判断和连续动作候选叠图 |
| `predictions.jsonl` | 源 PTS、置信度、可观测状态、手部候选 |
| `display_alignment.jsonl` | 每张展示帧与预测帧的时间对齐 |
| `events.json`、`completion.json` | 连续作业候选、范围、模型调用和覆盖统计 |

原工程入口 `project/infer_service/engine/pipeline.py` 使用可选 `raw_hand_tool` 后端。仓库内给出了供离线回放使用的模块及标签契约。接入完整生产工程时按原工程配置部署，并保持旧 281 维 LSTM 和本后端各自的输入契约。

## 试点的标注与验收

按原始时间戳保存 `video_id`、匿名操作者、管型、捆次、同根可确认时的临时编号、接入和退出的时间范围、可见性及证据来源。工具靠近、手离开、胶管握姿变化可作为复核线索。接口看不清时标 `unknown`，实际短断开保留两个事件。

代码内的 `ConnectionEpisodeCounter` 消费经核验的 `connected/disconnected/unknown` 观察。用以下命令复核规则。

```text
python -m unittest discover -s tests
```

R26 视频回放展示的是连续作业候选。实际接入计次和逐根完成要使用试点的接口真值。 [现场试点方案](pilot-plan.pdf)
