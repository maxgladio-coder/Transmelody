# Transmelody

**Turn separated vocals into an editable, beat-aligned melody MIDI.**

Transmelody 是面向人工精修的主旋律扒谱原型：输入时间对齐的 **vocal + inst** 分轨，
输出带速度事件、`WAV START` 标记和完整小节边界的旋律 MIDI。
目标是可编辑的乐谱旋律，而不是记录每一次颤音、气声和伴唱。

当前版本：**v0.1.1 · experimental · pitch-fusion mainline**（沿用 v0.1.0 模型权重）。
不是全轨转录器，不包含分轨模型，和弦生成尚未实现。

## 项目目录

业务代码统一放在 `transmelody/` 中，根目录只保留说明、依赖、测试和双击启动入口。

```text
transmelody/
  audio/        音频转换与试听
  grid/         节拍网格与速度轨
  midi/         MIDI 吸附
  models/       神经网络与特征模块
  inference/    推理、融合、导出
  training/     数据准备与训练
  workflow/     登记、队列与人工校对循环
  ui/           桌面界面
  evaluation/   对比指标与实验评估
  lyrics/       可选歌词工具
```

统一入口：`python -m transmelody --help`。详细说明见 [代码目录与入口](docs/CODE_LAYOUT.md)。
`dataset/`、`output/`、音频与预测目录位置不变，原有 `launch_*.cmd` 可以继续双击。

## 当前主线

| 部分 | 实现 |
| --- | --- |
| 节拍 / 小节 | Beat This! + 节拍网格与分段 tempo map |
| 音符边界 | MIDI 监督的 segmental Transformer / 完整音符解码 |
| 发音上下文 | 冻结的日语 wav2vec2 隐藏特征与软变化信号，不是精确五十音识别器 |
| 音高 | 基础模型 25% + RMVPE 辅助 Transformer 75% 的音高分数融合 |
| 量化 | 每小节选择 straight 十六分或 triplet 八分三连音 |
| 导出 | PPQ 480；速度轨、WAV START、首尾补齐完整小节 |

融合只修改既有音符的音高，**不增加、删除或重新切割音符**。
神经网络使用声谱特征；本项目并不是完全取消 STFT、网格规则或解码约束的端到端系统。

## 安装

已在 Windows、Python 3.11、PyTorch 2.7.1 / CUDA 12.8、RTX 5060 Laptop GPU 上运行。
CPU 会被自动选择为无 CUDA 时的后备，但未完成整首歌曲的 CPU 性能验证。
其他系统和 GPU 组合尚未做端到端验证。

```cmd
git clone https://github.com/maxgladio-coder/Transmelody.git
cd Transmelody
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/setup_models.py
```

非 CUDA 环境先安装适合硬件的同版本 torch / torchaudio，不要使用上面的 CUDA wheel 源。
FFmpeg 是音频转换的可选后备程序，需要自行安装并加入 PATH；正常 WAV 推理不需要它。

仓库自带两个经过隐私清理的自训练权重（合计约 10 MB），模型张量与当前主线逐项相等。
`setup_models.py` 校验它们、下载并校验 RMVPE 权重，然后安装到 `output/melody_transformer/`。
首次预测还会下载 Beat This! 和日语语音编码器；需要网络与磁盘空间。
第三方来源和许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

已有 RMVPE 权重时可离线安装这部分：

```cmd
python scripts/setup_models.py --rmvpe-file path\to\rmvpe.pt
python -m transmelody.inference.fusion_runtime status
```

安装脚本拒绝覆盖已经变化或继续训练过的模型。不要通过重新运行安装来回滚自己的训练。
只加载你信任来源的 PyTorch checkpoint；历史训练入口含 `weights_only=False` 加载逻辑。

## 单首预测，不推进工作流

准备同采样率、同长度、同起点的 WAV 分轨（建议 44.1 kHz / PCM16）：

```text
test audio/
  test_inst/1_inst.wav
  test_vocal/1_vocal.wav
```

```cmd
python -m transmelody.inference.predict_test_audio --song-id 1
```

知道恒定速度时，显式指定，不必完全依赖自动识别：

```cmd
python -m transmelody.inference.predict_test_audio --song-id 1 --bpm 176
```

默认输出到 `test_output/`：

- `1_predicted.mid`：旋律、tempo map 和 WAV START 标记。
- `1_tempo_map.mid`：相同的速度信息，便于 DAW 对齐。
- `grid_cache/1_grid.json`：统一 tick / 秒 / sample 坐标。
- `prediction_report.json`：BPM、速度分段、音符数量和融合信息。
- `diagnostics/1_prediction.npz`：诊断分数；其中 pitch logits 是基础模型分数，音符音高是融合后的结果。

导入 DAW 后启用速度轨，并把原 WAV 放到 **WAV START**。
更改 BPM 应重新同步网格并预测，不能只改 MIDI 头部速度。
`--no-pitch-fusion` 可临时对比基础模型；`--output-grid 1/16` 可关闭自动三连音量化。

## 人工校对 / 训练工作流

安装脚本创建空登记表和工作目录，**不包含任何歌曲、歌词或人工标注**。

1. 原曲放入 `original audio/`，用 `launch_song_renamer.cmd` 登记并按数字命名。
2. 用外部分轨工具生成 `original stem/N_inst.wav` 与 `N_vocal.wav`。
3. 用下面的 `stage` 命令检查并移入预测队列。
4. 打开 `launch_melody_queue.cmd`，或执行 `next`；每轮取最小编号的一对分轨。
5. 预测后分轨移入 dataset，状态变为待校对。人工 MIDI 存到 `vocal_mid/N_vocal.mid`。
6. 下一轮先验证并接收标注，再按选项训练和预测下一首。

```cmd
python -m transmelody.workflow.melody_queue_workflow stage
python -m transmelody.workflow.melody_queue_workflow stage --apply
python -m transmelody.workflow.melody_queue_workflow status
python -m transmelody.workflow.melody_queue_workflow next --epochs 0
```

**0 轮只跳过训练，不跳过校对检查、预测或文件移动。**
一般预测并不需要重新训练；选择训练轮数应看验证表现，而不是认为轮数越多越好。
工作流训练只更新基础模型，辅助音高网络保持冻结；更换基础权重后需重新评估融合效果。

独立训练的数据目录：

```text
dataset/melody_dataset/
  inst_audio/N_inst.wav
  vocal_audio/N_vocal.wav
  vocal_mid/N_vocal.mid
```

```cmd
python -m transmelody.training.prepare_dataset dataset/melody_dataset
python -m transmelody.training.train_melody --help
```

训练前明确训练 / 验证按歌曲划分。发布权重删除了私人划分和运行历史，不能精确恢复原优化器状态。
若继续训练公开模型，尽量用全新的歌曲作验证，避免把曾参与训练的曲子误作盲测。

## 其他组件

- `launch_audio_converter.cmd`：WAV / FLAC 转 44.1 kHz PCM16；转换可能替换源文件，先预览。
- `launch_midi_quantizer.cmd`：MIDI 网格吸附、PPQ 480、音量归一和碎片处理。
- `launch_melody_audition.cmd`：MIDI / 音频对照试听。
- `compare_melody_midi.py`、`melody_evaluation.py`：音符匹配与评价。
- `score_transcriber_v2.py`：辅助模型定义和实验训练；生产预测使用其音高输出。
- `lyric_alignment.py` 等：可选的歌词 / SOFA 研究工具，不属于默认推理依赖；
  SOFA 源码和权重未随仓库分发，仅安装 `requirements-lyrics.txt` 不足以运行这些实验。

## 已知限制

这是少量歌曲开发出的预标注工具，需要人工校对，不能作为可靠的全自动成谱服务。

- 抢拍鼓、切分、弱节拍尾部可能造成错误拍点、半速或假变速。
  已知恒速时优先指定 BPM；真实变速需要人工核对分段和小节相位。
- 长音仍可能切碎，连续同音仍可能合并；融合音高不能补救缺失的边界。
- 半音 ABA、气声、念白、漏入的和声和 delay 仍可能误判。
- 日语发音特征不是歌词级对齐保证；三连音判断也可能错误。
- 重复使用的开发集不是独立盲测，不能把内部指标当作通用准确率。

更多信息：[模型说明](docs/MODEL_CARD.md) · [标注规范](docs/ANNOTATION.md)。

## 测试

```cmd
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

需要外部模型的部分测试默认跳过。测试数据为程序生成或简短合成样例，不分发真实歌曲。

## 许可

原创代码使用 [MIT License](LICENSE)。第三方组件不因本项目许可而改变原有条款。
公开的自训练模型仅用于研究与预标注实验；权重许可说明见 `models/README.md`。
请只处理、训练和分发你有权使用的音乐及标注。
