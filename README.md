# 胶管逐根清洗视觉

这是现有手套手部识别、清洗作业回放与下一阶段现场试点的交接仓库。当前组合由 **R19 手框、R29 漏手补位、R18 二维 21 点、R26 因果动作分支**组成。R26 输出清洗作业候选。后续试点以实际接入和退出为标签，验证同一根任意一端连续连接满 **3.000 秒**的计数。

| 要做的事 | 入口 |
| --- | --- |
| 了解现场试点、逐根取放、验收指标 | [现场试点方案 PDF](docs/pilot-plan.pdf)、[方案与进展](docs/plan.md) |
| 跑单图、回放已有工位视频 | [操作说明](docs/operation.md) |
| 下载和核验六份推理权重 | [模型 Release](https://github.com/Meyt1n/hose-cleaning-vision/releases/tag/v0.1.0)、[逐文件版本与 SHA256](models/model_inventory.json) |
| 恢复数据、复现实验、继续训练 | [训练操作](docs/training.md)、[训练源码](training/) |
| 查论文与上游材料许可 | [论文](docs/papers.md)、[许可和来源](licenses/NOTICE.md) |

## 开始推理

先在拥有仓库访问权的 GitHub 账号下克隆并下载 Release 资产。

```text
git clone https://github.com/Meyt1n/hose-cleaning-vision.git
cd hose-cleaning-vision
gh release download v0.1.0 -R Meyt1n/hose-cleaning-vision -p inference-weights.zip
python scripts/weights.py --archive inference-weights.zip
```

Windows 或 Linux 可先按以下命令安装成对的 CPU 版 PyTorch 和 torchvision。

```text
python -m venv .venv
```

激活环境后执行以下命令。PowerShell 使用 `.\.venv\Scripts\Activate.ps1`，Bash 使用 `source .venv/bin/activate`。

```text
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-inference.txt
python run_hand_image.py --image "站位原图.png" --output artifacts/single-image
python project/scripts/predict_local_action_video.py --source "工位原视频.mp4" --start 0 --duration 5 --output artifacts/video-review
python -m unittest discover -s tests
```

单图写出 21 点和框的原图坐标。视频回放使用已冻结的 **4000×3000** 工位 ROI，经原工程离线推理入口写出带源时间戳的预测及叠图。实际接口接合和逐根完成仍待现场试点核验。

训练数据本体、现场整段录像、服务器启动凭据和原生产配置保存在项目环境。仓库中的训练源码可结合受控数据和父权重在服务器恢复。Release 里是推理权重，完整优化器恢复点由服务端实验目录提供。
