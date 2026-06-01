# 基于 Python 的通信系统仿真与教学演示系统

本项目是一个面向通信原理课程教学演示的桌面端仿真系统。系统基于 PyQt5 构建图形界面，围绕完整数字通信链路组织功能，支持文本、图像和语音信源输入，并提供信源编码、信道编码、数字调制、信道传输、解调译码、媒体恢复和多种可视化展示。

项目还集成了教学实验模块。用户可以通过主界面的“实验中心”运行预设 BER 对比实验，系统会自动完成多次通信链路仿真、误比特率统计、曲线绘制和结果导出，便于课堂演示和论文实验复现。

## 功能概览

- 信源输入：文本、灰度图像、16-bit PCM WAV 语音。
- 信源编码：哈夫曼编码、香农-范诺编码、算术编码。
- 信道编码：CRC、(7,4) 汉明码、卷积码。
- 译码方式：传统硬判决 Viterbi 译码，以及卷积码场景下可选的 AI-BiGRU 译码。
- 调制方式：MASK、MPSK、MQAM，支持 2/4/8/16 阶配置，其中 MQAM 当前使用 16QAM。
- 信道模型：AWGN、瑞利衰落、莱斯衰落。
- 可视化内容：原始/恢复媒体、比特序列、时域波形、功率谱、星座图、眼图。
- 教学实验：格雷码映射对比、信道编码方式对比、调制方式对比、AI-BiGRU 与 Viterbi 对比。
- 结果导出：单次仿真报告导出，实验中心 PNG 图像、CSV 数据和说明文本导出。

## 项目结构

```text
.
├── main.py                         # 图形界面主入口
├── README.md                       # 项目说明
├── requirements.txt                # Python 依赖
├── AI卷积码译码交接说明.md           # AI 译码功能与实验交接说明
├── comm_demo/
│   ├── core/                       # 通信链路核心逻辑
│   │   ├── bits.py                 # 比特和整数转换、格雷码辅助函数
│   │   ├── channel.py              # AWGN、瑞利衰落、莱斯衰落信道
│   │   ├── channel_coding.py       # CRC、汉明码、卷积码与译码入口
│   │   ├── constants.py            # 全局常量
│   │   ├── models.py               # 仿真配置和结果数据结构
│   │   ├── modulation.py           # 调制、脉冲成形、解调和星座映射
│   │   ├── session.py              # 分步仿真流程控制
│   │   ├── source.py               # 文本、图像、语音信源处理
│   │   └── source_coding.py        # 信源编码与解码
│   ├── gui/                        # PyQt5 图形界面
│   │   ├── app.py                  # QApplication 初始化
│   │   ├── dialogs.py              # 眼图窗口
│   │   ├── experiments_dialog.py   # 实验中心窗口
│   │   ├── main_window.py          # 主窗口
│   │   ├── media.py                # 图像和音频显示辅助
│   │   ├── plotting.py             # Matplotlib 可视化组件
│   │   └── workers.py              # 后台执行线程
│   └── test_16qam_gray_mod_demod.py # 16QAM 格雷映射测试脚本
├── ai卷积码译码/                    # AI-BiGRU 卷积码译码模块
│   ├── README.md
│   ├── data.py                     # 训练数据生成和软/硬输入处理
│   ├── inference.py                # 模型选择与推理入口
│   ├── media_validation.py         # 图像/语音媒体 BER 验证
│   ├── model.py                    # BiGRU 模型定义
│   ├── train.py                    # 模型训练入口
│   └── models/                     # 已训练模型权重
├── experiments/                    # 教学实验脚本
│   ├── common.py                   # 实验结果返回结构
│   ├── experiment_gray_ber.py
│   ├── experiment_channel_code_semilogy.py
│   ├── experiment_modulation_semilogy.py
│   └── experiment_ai_viterbi_media_ber.py
├── experiment_results/             # 已生成的实验图像和 CSV 数据
└── test_data/                      # 测试文本、图像和语音
```

## 环境准备

建议使用 Python 3.9。本项目当前主要在 `python-biyesheji` conda 环境中验证。

```powershell
conda create -n python-biyesheji python=3.9
conda activate python-biyesheji
pip install -r requirements.txt
```

如果需要使用 CUDA 运行 AI-BiGRU 推理或训练，请根据本机 CUDA 版本安装匹配的 PyTorch。若只运行普通通信链路仿真，且不启用 AI 译码和 AI 对比实验，PyTorch 不参与基础链路计算。

## 运行系统

在项目根目录执行：

```powershell
conda run -n python-biyesheji python main.py
```

或先激活环境后运行：

```powershell
conda activate python-biyesheji
python main.py
```

启动后，用户可以在左侧控制区选择信源类型、编码方式、调制方式、信道模型、SNR 和滚降系数等参数。系统支持连续运行、暂停、单步执行和重置，便于逐阶段观察通信链路处理过程。

## 实验中心

主界面左侧“流程控制”区域提供“实验中心”按钮。点击后会打开独立实验窗口，当前内置四类教学实验：

1. 格雷码映射 BER 对比：比较自然映射和格雷码映射对不同 16 阶调制方式 BER 的影响。
2. 信道编码 BER 对比：比较 CRC、汉明码和卷积码在 16QAM/AWGN 条件下的 BER。
3. 调制方式 BER 对比：比较 16ASK、16PSK 和 16QAM 在 AWGN 信道下的 BER 曲线。
4. AI-BiGRU 与 Viterbi BER 对比：比较卷积码链路中 AI-BiGRU 译码和硬判决 Viterbi 译码的 BER。

实验运行会复用 `comm_demo.core` 中的通信链路实现，并通过后台线程执行。实验完成后，窗口会展示生成的 BER 曲线图，并列出输出目录、PNG 图像和 CSV 数据文件。点击“导出结果”可以将本次实验的图像、数据和说明文本复制到指定文件夹。

也可以单独运行实验脚本：

```powershell
conda run -n python-biyesheji python .\experiments\experiment_gray_ber.py
conda run -n python-biyesheji python .\experiments\experiment_channel_code_semilogy.py
conda run -n python-biyesheji python .\experiments\experiment_modulation_semilogy.py
conda run -n python-biyesheji python .\experiments\experiment_ai_viterbi_media_ber.py
```

实验结果默认写入 `experiment_results/`。

## AI-BiGRU 卷积码译码

AI 译码模块位于 `ai卷积码译码/`。当前模型面向卷积码、硬判决输入和 AWGN 信道，按调制方式和阶数自动选择对应权重。

已保留的模型包括：

- `MASK`: 2、4、8、16 阶
- `MPSK`: 2、4、8、16 阶
- `MQAM`: 16 阶

在 GUI 中，只有当“信道编码”选择“卷积码”时，“AI译码”选项才会启用。当前 AI 专用模型仅支持 AWGN 信道；若在非 AWGN 信道下启用 AI 译码，系统会提示不支持。

## 测试数据与实验结果

`test_data/` 中提供了用于演示和实验的样例数据：

- `demo_image.png`: 图像信源测试样本。
- `demo_speech_zh.wav`: 语音信源测试样本。
- `text_test.txt`: 文本测试样本。

`experiment_results/` 中保存了当前论文实验所需的 BER 曲线图和 CSV 数据。重新运行实验脚本或在实验中心中运行实验时，会更新对应目录下的结果文件。

## 注意事项

- 语音输入当前要求为 16-bit PCM WAV；多声道音频会取第一通道。
- AI-BiGRU 与 Viterbi 对比实验会加载 PyTorch 和模型权重，运行时间相对较长。
- BER 曲线中若真实 BER 为 0，绘图时会使用很小的下限值辅助对数坐标显示，CSV 中仍保留真实 BER 数值。
- 项目中的 AI 模型权重文件较大，移动项目或提交归档时请确认 `ai卷积码译码/models/` 没有遗漏。
- 使用 `conda run -n python-biyesheji ...` 时，部分环境可能输出 OpenCL `temp.txt` 相关提示；只要命令退出码为 0，通常不影响本项目运行。
