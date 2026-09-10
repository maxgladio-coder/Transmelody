# 代码目录与新入口

根目录不再散放 Python 模块，所有业务代码归入 `transmelody/` 包。

```text
transmelody/
├── audio/        音频格式转换、重采样、试听合成
├── grid/         节拍、小节、速度轨、统一时间轴
├── midi/         MIDI 网格吸附
├── models/       主网络、音高辅助网络、事件与发音特征
├── inference/    特征到音符的推理、融合、预测导出
├── training/     数据准备、训练、模型发布
├── workflow/     登记、编号、分轨入队、人工校对循环
├── ui/           桌面小工具界面
├── evaluation/   MIDI 对比、指标、实验评估
├── lyrics/       可选歌词与发音标注工具
├── config.py     PPQ、拍号和速度容差
├── paths.py      稳定的项目根目录与源码位置
└── __main__.py   统一命令行入口
```

`dataset/`、`output/`、`test audio/`、`test_output/` 和 `.cache/` 的位置不变。
模型权重、速度网格、标注、登记状态和预测结果不会因代码搬家而移动。
现有根目录 `launch_*.cmd` 双击入口保留。

## 常用命令

在项目根目录、激活 Python 环境后执行：

```cmd
python -m transmelody --help
python -m transmelody predict --song-id 20 --bpm 176
python -m transmelody workflow status
python -m transmelody workflow check
python -m transmelody workflow next --epochs 0
python -m transmelody prepare dataset/melody_dataset
python -m transmelody train --help
python -m transmelody fusion status
python -m transmelody ui
```

`predict` 是独立预测，不推进工作流；`workflow next` 会按原逻辑检查校对、
可选训练、预测并移动队列文件。不要把二者混淆。

旧脚本名也可作为统一入口的命令，例如：

```cmd
python -m transmelody predict_test_audio.py --song-id 20
```

旧的 `python predict_test_audio.py` 路径不再存在；保留几十个根目录兼容文件会抵消整理效果。
外部脚本若直接导入旧模块名，应改为完整包名，例如
`from transmelody.grid.musical_timeline import canonicalize_grid`。
历史日志和旧版本 Release 保留原样，不将其当作新版本运行说明。

## 改动边界

这次是代码组织与入口迁移，不更改音符算法、模型参数、量化规则或自动 tempo 判定。
发音缓存仍在项目根目录，RMVPE 仍从当前融合策略指定的位置加载。
源码校验改为解析新模块位置，训练的来源记录覆盖整个包。

运行 `python -m pytest -q` 只收集 `tests/`，不会把发布副本和临时验证目录重复收集。
