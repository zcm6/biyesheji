# AI卷积码译码媒体BER实验说明

## 1. 实验目的

本实验用于比较卷积码传输链路中传统维特比译码与 AI-BiGRU 译码在实际媒体数据传输任务下的误比特率表现。

实验在 AWGN 信道下分别测试 16ASK、16PSK 和 16QAM 三种 16 阶调制方式，并以图像与语音测试数据的平均 BER 作为性能指标，为论文第五章的 AI 译码实验分析提供数据和图像结果。

## 2. 实验条件

| 项目 | 设置 |
|---|---|
| 信道 | AWGN |
| 信噪比 | 0、2、4、6、8、10 dB |
| 信道编码 | 卷积码 |
| 译码方式 | 维特比译码、AI-BiGRU 译码 |
| 调制方式 | 16ASK、16PSK、16QAM |
| 测试数据 | `test_data/demo_image.png`、`test_data/demo_speech_zh.wav` |
| 重复次数 | 每个媒体、每个 SNR 重复 3 次 |
| BER 统计对象 | 恢复后的原始信息比特与发送端原始信息比特 |
| AI 输入模式 | `hard` |
| 格雷码映射 | 开启 |
| 滚降系数 | 默认固定值 `0.35` |
| 信道随机性 | 固定验证随机种子 `20260521` |

程序中三种调制方式的参数对应关系如下：

| 论文表述 | 程序参数 |
|---|---|
| 16ASK | `MASK`, `order=16` |
| 16PSK | `MPSK`, `order=16` |
| 16QAM | `MQAM`, `order=16` |

## 3. 实验统计方式

对于每一种调制方式和每一个 SNR 点，实验分别传输图像数据和语音数据。

每种媒体重复传输 3 次，因此每个 SNR 点对每种译码方式共得到 6 个 BER 结果：

- 图像传输 BER 结果 3 个；
- 语音传输 BER 结果 3 个。

绘图时使用这 6 个 BER 的平均值作为该 SNR 下的平均 BER：

\[
\overline{BER}(SNR)=\frac{1}{6}\sum_{m \in \{image,audio\}}\sum_{r=1}^{3}BER_{m,r}(SNR)
\]

其中，\(m\) 表示媒体类型，\(r\) 表示重复实验次数。

维特比译码与 AI-BiGRU 译码使用同一批固定信道输出进行比较，保证两种译码方式面对相同的接收比特序列。

## 4. 实验脚本

本次实验脚本位于：

```text
experiments/experiment_ai_viterbi_media_ber.py
```

脚本复用了现有 AI 卷积码媒体 BER 验证链路，主要完成以下工作：

1. 加载 16ASK、16PSK 和 16QAM 对应的 AI-BiGRU 模型；
2. 读取图像和语音测试数据；
3. 在指定 SNR 下生成固定的 AWGN 媒体传输样本；
4. 分别计算维特比译码 BER 与 AI-BiGRU 译码 BER；
5. 保存逐次 BER 原始结果与按 SNR 汇总结果；
6. 输出三张独立的 SNR-BER 曲线图。

运行命令：

```powershell
conda run -n python-biyesheji python .\experiments\experiment_ai_viterbi_media_ber.py
```

本次运行中 AI-BiGRU 推理使用 CUDA 设备完成。

## 5. 输出文件

实验结果目录：

```text
experiment_results/ai_viterbi_media_ber/
```

### 5.1 数据文件

| 文件 | 内容 |
|---|---|
| `ai_viterbi_media_ber_raw.csv` | 每个媒体、每个 SNR、每次重复实验的 BER 原始结果 |
| `ai_viterbi_media_ber_summary.csv` | 按调制方式、SNR 和译码方式汇总后的平均 BER 与标准差 |

### 5.2 图像文件

| 文件 | 内容 |
|---|---|
| `ai_viterbi_media_ber_16ask_semilogy.png` | 16ASK 下维特比译码与 AI-BiGRU 译码 BER 对比图 |
| `ai_viterbi_media_ber_16psk_semilogy.png` | 16PSK 下维特比译码与 AI-BiGRU 译码 BER 对比图 |
| `ai_viterbi_media_ber_16qam_semilogy.png` | 16QAM 下维特比译码与 AI-BiGRU 译码 BER 对比图 |

BER 曲线使用对数纵轴绘制。对于真实 BER 为 0 的数据点，CSV 中仍保留真实值 0，绘图时仅为满足对数坐标显示使用绘图下限。

## 6. 本次实验结果摘要

### 6.1 16ASK

| SNR/dB | Viterbi 平均 BER | AI-BiGRU 平均 BER |
|---:|---:|---:|
| 0 | 0.33037280 | 0.28039199 |
| 2 | 0.26660691 | 0.20269695 |
| 4 | 0.18171977 | 0.11778528 |
| 6 | 0.10084586 | 0.05630625 |
| 8 | 0.04303795 | 0.02222251 |
| 10 | 0.01377842 | 0.00686329 |

在本次 16ASK 测试中，AI-BiGRU 在全部 SNR 点的平均 BER 均低于维特比译码。

### 6.2 16PSK

| SNR/dB | Viterbi 平均 BER | AI-BiGRU 平均 BER |
|---:|---:|---:|
| 0 | 0.17272577 | 0.11576960 |
| 2 | 0.08413611 | 0.04676941 |
| 4 | 0.03129278 | 0.01426572 |
| 6 | 0.00830746 | 0.00326999 |
| 8 | 0.00121448 | 0.00043972 |
| 10 | 0.00009764 | 0.00005577 |

在本次 16PSK 测试中，AI-BiGRU 在全部 SNR 点的平均 BER 均低于维特比译码。

### 6.3 16QAM

| SNR/dB | Viterbi 平均 BER | AI-BiGRU 平均 BER |
|---:|---:|---:|
| 0 | 0.03052948 | 0.02614392 |
| 2 | 0.00409275 | 0.00330195 |
| 4 | 0.00018223 | 0.00014694 |
| 6 | 0.00000124 | 0.00000360 |
| 8 | 0.00000000 | 0.00000050 |
| 10 | 0.00000000 | 0.00000000 |

在本次 16QAM 测试中，AI-BiGRU 在 0 dB、2 dB 和 4 dB 时平均 BER 略低于维特比译码；在 6 dB 和 8 dB 时维特比译码平均 BER 更低；在 10 dB 时两种译码方式均未出现误比特。

## 7. 可用于论文的图题建议

三张图可分别命名为：

1. 16ASK 调制下 AI-BiGRU 译码与维特比译码 BER 对比；
2. 16PSK 调制下 AI-BiGRU 译码与维特比译码 BER 对比；
3. 16QAM 调制下 AI-BiGRU 译码与维特比译码 BER 对比。

## 8. 备注

运行实验时，`conda` 环境会输出与 OpenCL `temp.txt` 相关的提示信息，但本次实验脚本已正常完成，实验图像和 CSV 数据均已成功生成。
