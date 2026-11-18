# 来源和使用条件

此私有仓库用于项目交接及研究验证。工位材料、客户工程衍生模块及模型由项目参与者按项目权限使用。第三方代码、数据及上游权重各自适用原始发布条款。用于生产、商业交付或进一步转发前应按具体资产核实相应许可。

| 来源 | 此处的用途与证据 |
| --- | --- |
| MMPose | R18 使用 RTMPose-m Hand5 原生结构移植。派生文件保留项目声明与 [Apache-2.0 原文](MMPose-Apache-2.0.txt)。上游为 [MMPose](https://github.com/open-mmlab/mmpose)，对应实现参考提交 `759b39c` |
| MC-hands-1M | 只使用 Big 集冻结子集训练手套姿态。作者提示 Small 有已知 GT 问题。条款与下载见 [Zenodo 发布](https://zenodo.org/records/7194271) |
| FreiHAND | R18 混合适配与 Hand5 上游利用了真实手监督。官方数据发布限制商业使用。见 [FreiHAND 官方说明](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) |
| HaGRID | R19 使用 HaGRID 子集的作者框。见 [作者项目](https://github.com/hukenovs/hagrid) 与 [项目当时留存的条款](HaGRID-dataset-terms.txt) |
| torchvision 官方权重 | COCO Faster R-CNN ResNet50-FPN-v2 提供 R19/R29 底座。ImageNet ResNet18 提供 R26 的双 ROI 图像编码器。见 [torchvision 模型与权重文档](https://docs.pytorch.org/vision/stable/models.html) |

六个具体推理资产的字节数和 SHA256 见 [权重清单](../models/model_inventory.json)。原始现场媒体、全量数据和原训练恢复点由绑定的项目数据位置提供。
