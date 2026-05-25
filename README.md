# Bastard TV Head

> Fuck up!

一个像素风电视机脑袋。

## 怎么玩

```bash
# 安装
uv sync

# 完整启动（实时麦克风 → 情绪口型 → 像素表情）
uv run python -m app.main

# Demo 模式（手动轮播所有表情）
uv run python -m app.main --mode demo

# 单独测试各个模块
uv run python -m app.display   # 渲染器：发送像素窗口
uv run python -m app.audio     # 麦克风采集 + 波形
uv run python -m app.analysis  # 口型 + 情绪分析
```

## 模块拆解

| 模块 | 内容 | 单独调试 |
|---|---|---|
| `audio/capture` | PyAudio 麦克风流 | `-m app.audio` |
| `analysis/mouth` | RMS 振幅 → 口型 0/1/2 | 无依赖，纯信号 |
| `analysis/emotion` | speechbrain 情绪分类 | `-m app.analysis` |
| `display/faces` | 18 组像素图案矩阵 | `from app.display.faces import *` |
| `display/renderer` | pygame 窗口渲染 | `-m app.display` |
| `core/orchestrator` | 主循环调度 + profiling | 内嵌在 `main.py` |

## 依赖

- Python >= 3.10
- `pyaudio` — 麦克风
- `pygame` — 渲染
- `numpy` — 信号处理
- `speechbrain` + `torch` — 情绪模型

用 `uv` 一把梭：

```bash
uv sync
```
