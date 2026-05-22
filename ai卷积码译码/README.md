# AI卷积码译码

这个目录保留面向当前项目卷积码的双向 GRU 译码器和 AWGN 专用模型。
训练数据直接调用 `comm_demo.core` 的卷积编码、调制、信道和解调函数，因此和
项目的 MASK、MPSK、16QAM 链路一致。

## 文件

- `model.py`: 小参数量 BiGRU 模型定义。
- `data.py`: 单一调制配置的数据生成，保留 `hard` 与 `soft` 输入。
- `train.py`: 单一调制方式和阶数的训练入口。
- `inference.py`: 按调制方式和阶数选择模型的统一推理入口。
- `media_validation.py`: 固定图像和语音媒体链路 BER 验证入口。

## 已保留模型

- `MASK`: `M=2,4,8,16`
- `MPSK`: `M=2,4,8,16`
- `MQAM`: `M=16`

模型文件位于 `models/`，当前均为 AWGN 训练权重。`inference.py` 会按
`(modulation, order)` 自动选择这些模型。

## 训练与评估

训练硬判决输入模型：

```powershell
python .\ai卷积码译码\train.py --steps 2000 --input-mode hard --modulation MQAM --order 16 --channel-name AWGN --output .\ai卷积码译码\models\bigru_mqam_16_awgn.pt
```

保留软输入训练入口：

```powershell
python .\ai卷积码译码\train.py --steps 2000 --input-mode soft --modulation MQAM --order 16 --channel-name AWGN
```

使用固定媒体 BER 选择模型：

```powershell
python .\ai卷积码译码\train.py --media-validation --max-steps 3000 --validate-every 200 --validation-snrs 0 2 4 6 8 10 12 --validation-repeats 3 --input-mode hard --modulation MQAM --order 16 --channel-name AWGN
```

训练仍使用在线随机 bit。验证器会预先固定 `demo_image.png` 和
`demo_speech_zh.wav` 在各个 SNR、各个重复次数下的接收比特，再在训练过程中重复
使用这些验证样本。训练会跑满 `max_steps`，保存验证平均 BER 最低的 `.best.pt`；
只有该 best 模型满足 `AI_avg_BER <= Viterbi_avg_BER` 时，才写出新的
`*_media_ber.pt` 正式模型，原有专用权重仍可作为对照。

单独验证现有模型的媒体 BER：

```powershell
python .\ai卷积码译码\media_validation.py --model .\ai卷积码译码\models\bigru_mqam_16_awgn.pt --input-mode hard --modulation MQAM --order 16 --channel-name AWGN --snrs 0 2 4 6 8 10 12 --repeats 3
```
