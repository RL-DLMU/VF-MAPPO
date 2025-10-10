# 自适应交通信号控制
本项目使用多智能体强化学习解决自适应交通信号控制中的不同车辆等待时间不公平的现实问题。以下内容介绍了项目的环境要求以及使用方式。

## 环境要求：
- 操作系统：Windows/Linux/MacOS（已在Windows 以及Linux 上测试）。
- 主要依赖包括：
  - Tensorflow
  - NumPy
  - Matplotlib
  - Math
## 使用方式：
## 训练模型
- 直接运行 `VF-MAPPO.py` 即可
- `python VF-MAPPO/VF-MAPPO.py`

本项目为论文 [Vehicle-Level Fairness-Oriented Constrained Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control](https://ieeexplore.ieee.org/document/10907776) 的官方开源实现，如有引用请使用以下 BibTeX：
```bibtex
@ARTICLE{10907776,
  author={Liu, Wanting and Zhang, Chengwei and Fang, Wanqing and Zhou, Kailing and Li, Yihong and Zhan, Furui and Wang, Qi and Xue, Wanli and Chen, Rong},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Vehicle-Level Fairness-Oriented Constrained Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control}, 
  year={2025},
  volume={26},
  number={4},
  pages={4878-4890}}

