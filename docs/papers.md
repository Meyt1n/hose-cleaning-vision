# 论文、模型和数据官方来源

- RTMPose（2023）。R18 使用 Hand5 CSPNeXt-M、RTMCC 和手部推理结构。 [技术报告](https://arxiv.org/abs/2303.07399)

- SimCC（ECCV 2022）。R18 使用 X、Y 分轴坐标分类，训练用高斯目标与 KL 损失。 [论文](https://arxiv.org/abs/2107.03332)

- Hand5 原生移植与父模型来自 MMPose 提交 `759b39c13fea6ba094afc1fa932f51dc1b11cbf9`。 [实现](https://github.com/open-mmlab/mmpose/tree/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/projects/rtmpose)

- FPN（CVPR 2017）为手部检测提供多尺度特征。 [论文](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html)

- R19 和 R29 使用 torchvision Faster R-CNN ResNet50-FPN-v2、COCO 初始化和 FPN、RPN、ROI 框头。 [Torchvision 0.16](https://docs.pytorch.org/vision/0.16/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html)

- MC-hands-1M 的 Big 集子集用于手套姿态微调。作者记录了 Small 集的真值问题。 [Zenodo](https://zenodo.org/records/7194271)

- FreiHAND（ICCV 2019）为混合训练提供真实手样本，二维监督来自三维拟合投影。 [论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zimmermann_FreiHAND_A_Dataset_for_Markerless_Capture_of_Hand_Pose_and_ICCV_2019_paper.pdf)

- HaGRID（WACV 2024）的作者手框子集用于手部检测。 [论文](https://openaccess.thecvf.com/content/WACV2024/html/Kapitanov_HaGRID_--_HAnd_Gesture_Recognition_Image_Dataset_WACV_2024_paper.html)

- GlovEgo-HOI 用于 R15 早期冻结特征对照。该轮得分和标签核查推动了 Hand5 选型。 [作者仓库](https://github.com/NextVisionLab/GlovEgo-HOI)

- DSTA（CVPR 2024）。R31 借鉴关节 token 和空间、时间分别建模，两组已各训练 1280 步，当前版采用 R18。 [论文](https://openaccess.thecvf.com/content/CVPR2024/html/He_Video-Based_Human_Pose_Regression_via_Decoupled_Space-Time_Aggregation_CVPR_2024_paper.html)

- Surgical Hands 可参考手部时序跟踪标注。当前训练数据清单见训练说明。 [作者仓库](https://github.com/MichiganCOG/Surgical_Hands_RELEASE)

## 逐根试点

- Ego4D Hands & Objects 的物体状态变化用于设计连接前、接合时、退出后的标注。 [官方任务](https://ego4d-data.org/docs/benchmarks/hands-and-objects/)
- ASRF 把视频逐帧分类和边界估计置于两个分支。借鉴独立边界监督，先标清同一作业类别内的每次退出与接入。 [论文与实现](https://yiskw713.github.io/asrf/)
- ORViT 对对象区域外观和运动分别编码，支持将管端、工具口作为独立 ROI 输入。 [论文与实现](https://roeiherz.github.io/ORViT/)
- Assembly101 提供多视角的装配、错误和纠正过程，适合参考连续工序和跨操作者拆分。 [数据集与论文](https://assembly-101.github.io/)
