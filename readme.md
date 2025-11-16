

# ZeroLLM

<div align="center">

<img src="data/asset/ZeroLLM.png" width="320"/>

</div>



<p align="center">

  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/buladuo/ZeroLLM?style=flat-square"></a>

  <a href="#"><img alt="Language" src="https://img.shields.io/badge/language-C%2FC%2B%2B-green?style=flat-square"></a>

</p>

## 📖 项目简介

**ZeroLLM** 是一个完全使用 C/C++ 编写的轻量级大语言模型（LLM）训练与推理框架。

与主流深度学习框架（如 PyTorch, TensorFlow）不同，ZeroLLM 摒弃了复杂的 Python 封装与抽象，旨在通过**从零构建**每一个深度学习算子与组件，实现对底层原理的极致透明化。本项目旨在帮助研究者和开发者深入理解 LLM 的训练/推理核心机制，并提供一个高性能、可高度定制的实验平台。

## 💡 设计哲学

> **"What I cannot create, I do not understand." — Richard Feynman**

本项目不仅仅是一个轮子，更是一次对底层技术的探索。

  * **极简与透明 (Simplicity & Transparency)**：
    我们不得不承认，从零实现是一个艰巨的挑战（本项目部分代码亦得益于 AI 辅助生成）。ZeroLLM 并不追求“开箱即用”的广泛兼容性，而是通过统一的代码风格与接口，让前向传播、反向传播的每一个细节都清晰可见。

  * **极致的垂直优化 (Vertical Optimization)**：
    通用框架往往为了兼容性而牺牲性能或引入臃肿的代码。ZeroLLM 的目标不是同时支持数百种模型，而是**专注于单一模型架构**，从算子层、调度层到部署层进行垂直整合与极致优化。这使得开发者在修改模型时，虽然需要手动调整前后向传播逻辑，但能获得对系统行为的完全掌控。

## ✨ 核心特性

  * **⚡ 纯 C/C++ 实现**：零 Python 依赖，直接操作内存与指针，追求极致的运行时性能。
  * **🧩 模块化架构**：清晰的分层设计（Kernel -\> Layer -\> Model），结构严谨，易于扩展。
  * **🔄 完整训练闭环**：内置数据加载、BPE Tokenizer、损失计算、优化器（SGD/Adam/AdamW）及反向传播链路。
  * **🚀 多后端支持**：支持 CPU、CUDA、HIP 等加速后端（开发中）。
  * **📊 实时可视化**：内置训练过程可视化工具（Loss 曲线绘制），告别盲盒训练。
  * **🎓 教学友好**：详尽的代码注释与直观的逻辑实现，是学习 Transformer 架构与 CUDA 编程的理想范本。

## 🏗️ 架构组成

ZeroLLM 采用清晰的自底向上架构：

| 层级 | 组件名称 | 描述 |
| :--- | :--- | :--- |
| **Utils** | 工具集 | 日志系统、进度条、权重序列化、配置解析 |
| **Runtime** | 运行时 | 数据加载器 (DataLoader)、训练循环控制 (Trainer) |
| **Kernel** | 算子层 | 基础 CUDA/C++ 算子 (MatMul, Softmax, RoPE 等) |
| **Layer** | 网络层 | Transformer Block, Multi-head Attention, FFN, RMSNorm |
| **Loss/Optim** | 优化层 | CrossEntropy Loss, AdamW, SGD 优化器实现 |
| **Model** | 模型层 | 完整的 ZeroLLM 模型定义 |

## 🔬 默认模型规格

当前默认提供一个参数量约为 **26M** 的实验性模型配置。
*(注：该规模主要用于验证框架逻辑与调试，Loss 目前收敛表现一般 [8.7 -\> 5.4]，适合作为 Demo 运行)*

| 参数项 | 数值 | 说明 |
| :--- | :--- | :--- |
| **Vocab Size** | 6400 | 自定义 BPE 词表 |
| **Dimension** | 512 | 嵌入层维度 |
| **Layers** | 6 | Transformer 层数 |
| **Heads** | 8 | 注意力头数 |
| **Hidden Dim** | 2048 | FFN 隐藏层维度 |
| **Max Seq Len** | 2048 | 上下文窗口大小 |
| **Total Params** | **\~26M** | 包含 Embedding 与 Output 层 |

## 🚀 快速开始

### 环境要求

  * **编译器**: C++17 或更高版本 (GCC/Clang)
  * **GPU 环境**: CUDA 11.0+ (强烈推荐 NVIDIA GPU)
  * **构建工具**: CMake 3.18+

### 编译与运行

```bash
# 1. 克隆仓库
git clone https://github.com/hkust-nlp/zerollm.git
cd zerollm

# 2. 创建构建目录
mkdir build && cd build

# 3. 编译 (利用多核加速)
cmake ..
make -j$(nproc)

# 4. 运行训练 (示例)
./zerollm_train --config ../configs/nano_model.json
```

## 📅 开发进度与路线图

  - [x] **核心功能**
      - [x] 基础 Layer 定义 (Attention, FFN, Norm)
      - [x] BPE Tokenizer 训练与推理
      - [x] 模型权重保存与加载 (Checkpointing)
  - [x] **可视化与监控**
      - [x] 训练 Loss 实时可视化绘制
  - [ ] **性能与扩展**
      - [ ] 多卡分布式训练 (Multi-GPU Support)
      - [ ] 多后端支持 (HIP/ROCm, Metal)
      - [ ] 算子融合与 FlashAttention 实现
      - [ ] INT8/FP16 量化支持

## 🤝 贡献

欢迎提交 Issue 或 Pull Request。由于本项目强调“手动管理底层逻辑”，提交代码时请务必保持与现有代码风格的一致性，并确保手动维护相关的反向传播逻辑。

## 📄 许可证

[MIT License](LICENSE)
