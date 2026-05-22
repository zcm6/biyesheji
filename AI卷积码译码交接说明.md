# AI 卷积码译码功能交接说明

本文档用于向后续协作模型说明本项目新增 AI 卷积码译码功能的真实实现状态、实验结果和论文写作上下文。文档只整理事实与约束，不作为论文正文直接使用。

## 1. 项目背景

- 项目名称：基于 Python 的通信系统仿真与教学演示系统。
- 主入口：`main.py`。
- 当前实际 GUI 路径：`comm_demo/gui/`。
- 当前实际通信链路 core 路径：`comm_demo/core/`。
- `comm_demo/ui.py`、`comm_demo/pipeline.py`、`comm_demo/pipeline_copy.py` 属于旧实现或历史文件，当前 `main.py` 不走这些路径。
- 项目支持文本、图像、语音三类信源，包含信源编码、信道编码、MASK/MPSK/MQAM 调制、AWGN/衰落信道、解调、媒体恢复和 BER 等功能。

## 2. AI 译码功能定位

- 本次新增的是卷积码 AI 译码功能。
- 当前没有为汉明码接入 AI 译码。
- 系统原有卷积码译码方式是硬判决 Viterbi。
- 新功能在卷积码场景下提供传统 Viterbi 与 AI-BiGRU 译码的可选切换。
- 当前 AI 模型训练和接入均面向 AWGN 信道。
- 当前接入的 AI 模型使用硬判决输入。
- 当前比较对象是硬判决 Viterbi，不是软判决 Viterbi。

## 3. UI 与 core 接入状态

### 3.1 UI 行为

文件：`comm_demo/gui/main_window.py`

- 在参数配置区的“信道编码”一行中，左侧保留信道编码下拉框。
- 同一行右侧新增 `AI译码` 选项，下拉内容为 `否`、`是`。
- 当信道编码为 `卷积码` 时，AI 译码选项可用。
- 当信道编码为 `CRC` 或 `汉明码` 时，AI 译码选项置灰，且自动恢复为 `否`。

### 3.2 配置传递

文件：

- `comm_demo/core/models.py`
- `comm_demo/core/session.py`

实现状态：

- `SimulationConfig` 新增布尔字段：

```python
ai_decoder: bool = False
```

- GUI 采集参数时会把 AI 译码开关传给 `create_session()`。
- `SimulationSession` 在信道解码阶段把以下参数传给 core 解码函数：

```python
ai_decoder
modulation
order
channel_name
```

### 3.3 core 分支

文件：`comm_demo/core/channel_coding.py`

- `channel_decode()` 已扩展 AI 参数。
- `ai_decoder=False` 时，卷积码仍走原 Viterbi 译码。
- `ai_decoder=True` 且编码方式为卷积码时，调用 `ai卷积码译码.inference.bigru_decode()`。
- 如果非卷积码场景误传 `ai_decoder=True`，core 会报错拦截。

## 4. AI 译码目录结构

当前目录：`ai卷积码译码/`

```text
ai卷积码译码/
├─ models/
├─ data.py
├─ inference.py
├─ media_validation.py
├─ model.py
├─ README.md
├─ train.py
└─ __init__.py
```

已删除的旧实验脚本：

- `evaluate.py`
- `evaluate_image.py`

当前职责划分：

- 项目接入：`inference.py`、`model.py`、`models/`
- 模型训练：`train.py`、`data.py`
- 模型媒体 BER 评估：`media_validation.py`

## 5. 模型来源与设计

### 5.1 模型思路来源

- 选用的迁移参考项目：`yihanjiang/Sequential-RNN-Decoder`。
- 对应参考论文为 Kim、Jiang 等人的 `Communication Algorithms via Deep Learning`。
- 该思路将卷积码译码看作带噪编码序列到信息比特序列的序列恢复问题，使用循环神经网络学习译码映射。
- 论文写作中若引用该思路，应重点用于说明选择 RNN/GRU/BiGRU 处理卷积码序列结构的依据，不宜在系统设计章节过度展开研究背景。

### 5.2 本项目模型结构

文件：`ai卷积码译码/model.py`

- 模型为双向 GRU 卷积码译码器。
- 输入张量形状：

```text
[batch, steps, 2]
```

- 输出张量形状：

```text
[batch, steps]
```

- 模型末端为线性读出层，训练使用 `BCEWithLogitsLoss`。
- 项目卷积码为码率 1/2，每个时间步输入两个卷积编码接收观测。
- 硬判决输入会把比特映射为：

```text
0 -> -1
1 -> +1
```

- 训练时已改为只对真实信息位计算 loss，不再把卷积码尾部终止补零位纳入 loss。

## 6. 训练数据与验证方法

### 6.1 训练数据

文件：`ai卷积码译码/data.py`

- 训练仍使用在线随机 bit。
- 训练样本通过本项目 core 链路生成：
  - `comm_demo.core.channel_coding.convolutional_encode`
  - `comm_demo.core.modulation.modulate`
  - `comm_demo.core.channel.apply_channel`
  - `comm_demo.core.modulation.demodulate`
- 输入模式保留 `hard` 与 `soft` 两种入口，但当前接入系统的是 `hard` 模型。

### 6.2 模型验证

文件：`ai卷积码译码/media_validation.py`

- 当前统一评估入口是 `media_validation.py`。
- 验证媒体：
  - `test_data/demo_image.png`
  - `test_data/demo_speech_zh.wav`
- 默认验证条件：
  - AWGN 信道
  - SNR 为 `0, 2, 4, 6, 8, 10, 12 dB`
  - 每个 SNR 重复 3 次
  - 固定随机种子
- AI 与 Viterbi 在同一套接收比特上比较。
- 主要指标：
  - 图像平均 BER
  - 语音平均 BER
  - 二者平均 BER

### 6.3 随机性固定

文件：`comm_demo/core/channel.py`

- `apply_channel()` 新增了可选 `rng` 参数：

```python
rng: np.random.Generator | None = None
```

- 不传 `rng` 时项目正常仿真行为不变。
- 媒体 BER 验证时传入固定 `rng`，保证不同模型比较使用同一套噪声样本。

## 7. 当前最终保留模型

目录：`ai卷积码译码/models/`

最终只保留 9 个标准文件：

```text
bigru_mask_2_awgn.pt
bigru_mask_4_awgn.pt
bigru_mask_8_awgn.pt
bigru_mask_16_awgn.pt
bigru_mpsk_2_awgn.pt
bigru_mpsk_4_awgn.pt
bigru_mpsk_8_awgn.pt
bigru_mpsk_16_awgn.pt
bigru_mqam_16_awgn.pt
```

模型选择映射位于 `ai卷积码译码/inference.py`：

```python
("MASK", 2) -> bigru_mask_2_awgn.pt
("MASK", 4) -> bigru_mask_4_awgn.pt
("MASK", 8) -> bigru_mask_8_awgn.pt
("MASK", 16) -> bigru_mask_16_awgn.pt
("MPSK", 2) -> bigru_mpsk_2_awgn.pt
("MPSK", 4) -> bigru_mpsk_4_awgn.pt
("MPSK", 8) -> bigru_mpsk_8_awgn.pt
("MPSK", 16) -> bigru_mpsk_16_awgn.pt
("MQAM", 16) -> bigru_mqam_16_awgn.pt
```

## 8. 16QAM 模型的特殊情况

- `MQAM-16` 的最终模型与其他大多数模型参数不同。
- 其他最终模型大多数为：

```text
hidden_size = 64
num_layers = 2
```

- `MQAM-16` 最终模型为优化后的：

```text
hidden_size = 128
num_layers = 2
information_len = 256
train_snr = 0~6 dB
seed = 20260522
best step = 3000
```

- `inference.py` 加载模型时会从 checkpoint 的 `config` 构造模型，因此 `MQAM-16` 不会因为参数量不同而加载错误。
- 已验证 `MQAM-16` 实际选择：

```text
bigru_mqam_16_awgn.pt
```

- 已验证加载出的 16QAM 模型 hidden size 为 128。

## 9. 已得到的媒体 BER 结果

以下表格对应当前最终模型筛选阶段的关键 BER 结果。比较对象均为硬判决 Viterbi。

| 配置 | AI 平均 BER | Viterbi 平均 BER | 结果 |
|---|---:|---:|---|
| MASK-2 | 0.00001123 | 0.00000000 | Viterbi 更优 |
| MASK-4 | 0.00093634 | 0.00015157 | Viterbi 更优 |
| MASK-8 | 0.02796541 | 0.03200770 | AI 更优 |
| MASK-16 | 0.09815365 | 0.13423376 | AI 更优 |
| MPSK-2 | 0.00001170 | 0.00000000 | Viterbi 更优 |
| MPSK-4 | 0.00001970 | 0.00000007 | Viterbi 更优 |
| MPSK-8 | 0.00368560 | 0.00302471 | Viterbi 更优 |
| MPSK-16 | 0.02584368 | 0.04257810 | AI 更优 |
| MQAM-16 | 0.00424722 | 0.00498349 | AI 更优 |

结论约束：

- 不能写成 AI 译码在所有配置下优于 Viterbi。
- 可写成 AI 译码在部分高阶调制配置下取得了低于硬判决 Viterbi 的平均 BER。
- 低阶调制下 Viterbi 基线已很强，BER 接近 0，AI 超过它的空间很小。

## 10. 推理性能优化

图像传输时 AI 信道解码曾出现约 7~8 秒等待。已做以下推理优化：

文件：`ai卷积码译码/inference.py`

1. 自动设备选择：

```python
"cuda" if torch.cuda.is_available() else "cpu"
```

2. 全局模型缓存：

- 缓存键为 `(model_path, device)`。
- 同一模型和设备重复推理不再重复加载 checkpoint。

3. 推理上下文改为：

```python
torch.inference_mode()
```

4. 长序列仍采用窗口式推理，默认：

```text
window_steps = 8192
```

已验证：

- `python-biyesheji` 环境中 `preferred_device` 为 `cuda`。
- 重复调用 `MQAM-16` AI 解码后缓存大小保持不变。

## 11. 运行环境

项目主要环境：`python-biyesheji`

为支持 AI 译码，已安装：

```text
pytorch 2.1.0
pytorch-cuda 12.1
```

已确认：

```text
torch.__version__ == 2.1.0
torch.version.cuda == 12.1
torch.cuda.is_available() == True
```

`python-biyesheji` 中 NumPy 已调整为：

```text
numpy 1.24.3
```

原因：

- 项目 `requirements.txt` 本就固定 `numpy==1.24.3`。
- PyTorch 2.1.0 与 NumPy 2.0.2 配合时，`torch.from_numpy()` 失败。

## 12. 论文写作上下文

用户论文文件：

```text
e:\毕业设计\论文\通信22-2BF_14222302539_周晨明_基于Python的通信系统仿真与教学演示系统-5.18.pdf
```

当前论文第三章结构中：

```text
3.4 信道传输与接收处理
3.5 数据恢复与性能统计
3.6 本章小结
```

用户计划：

- 在 `3.4` 与原 `3.5` 之间新增独立章节：

```text
3.5 基于 BiGRU 的卷积码 AI 译码模块设计与实现
```

- 原 `3.5 数据恢复与性能统计` 后移。
- 第五章新增一个专门小节写 AI 译码与 Viterbi 的实验与分析。
- 用户希望第三章新增章节强调“设计与实现”，不要写得像研究现状或实验分析。
- 用户倾向于将功能设计、模型原理、训练与系统接入连续写在一个章节内，不拆成四个独立小节。
- 在讲 BiGRU 译码模型原理时，应借鉴 `Communication Algorithms via Deep Learning` 论文的卷积码序列译码建模思路，但不应在第三章过度铺陈背景、性能结论和实验讨论。

## 13. 论文表述注意事项

适合写：

- 本文在保留 Viterbi 译码的基础上，为卷积码链路增加 AI 译码可选功能。
- BiGRU 模型按卷积码时间步接收编码比特对，输出信息比特预测。
- 不同调制方式和阶数对应不同专用模型。
- 16QAM 模型做了针对性优化。
- 使用图像和语音媒体链路 BER 对模型进行验证。

不宜写：

- AI 译码全面优于 Viterbi。
- 本文提出了全新的卷积码神经译码理论。
- 当前 AI 模型支持所有信道。
- 当前比较已覆盖软判决 Viterbi。

## 14. Git 与缓存文件

- 项目根目录已新增 `.gitignore`：

```gitignore
__pycache__/
*.py[cod]
```

- Python `pyc` 和 `__pycache__` 不应提交到 GitHub。

