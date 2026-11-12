# 适配改动 / 性能实测 / 已知问题

本文件是 [README](../README.md) 的配套说明，记录相对原项目的改动、性能实测数据与已知问题。

## 一、相对原项目的主要改动

| # | 改动 | 位置 | 说明 |
|---|---|---|---|
| 1 | **YOLO11 split-9 输出解码** | 新增 `engine/yolo/yolo_split9_decode.{hpp,cpp}` | 原项目模型（`yolo26n_split`）是 2 输出（`bbox[4,8400]` + `score[80,8400]`）；YOLO11 官方导出是 **9 输出**（3 尺度 × box64/cls80/sum1），需在 CPU 侧做 DFL 解码。解码逻辑独立成文件，未污染原有实现 |
| 2 | **`output_mode` 参数开关** | `yolo_engine.hpp/.cpp`、`yolo_node.cpp`、`config/params.yaml` | `auto` 按输出张量个数自动选择（9→split9，其它→split2），也可强制 `split9` / `split2` |
| 3 | **视频文件源替代摄像头** | `engine/yolo/camera_capture.{h,cpp}`、`yolo_node.cpp` | 新增 `camera_source: file` + `video_file` / `video_fps` / `video_loop`，直接读裸 UYVY 帧序列并按帧率节流、支持循环；`device` 模式保持原样 |
| 4 | **去掉重复双线性缩放** | `yolo_engine.cpp` | RGA 已输出 640×640，原逻辑又缩一次 640→640；改为尺寸相同时直接 `memcpy`（省约 50 ms/帧） |
| 5 | 模型/工具路径修正 | `config/params.yaml` | `/home/topeet/...` → 本机实际路径 |
| 6 | 麦克风卡号修正 | `config/params.yaml` | `plughw:2,0`（该板 card 2 是 hdmi0，无法采集）→ `plughw:1,0`（ES8388） |
| 7 | DDS 绑定网卡 | `/cfg/fastdds_wlan1.xml`（容器外挂载） | 多网卡/同网段场景下用白名单固定走一块网卡，见本文件第三节 |
| 8 | **lifecycle 转换加重试** | `launch/voice_assistant.launch.py` | 原实现用固定定时器（1/12/13/28/29/30 s），LLM 加载 1.3 GB 时 `ros2 lifecycle set --no-daemon` 常因 DDS 发现超时报 `Node not found`，或 `activate` 抢在 `configure` 前报 `Unknown transition`。改为只对这两类瞬时错误重试（最多 30 次 ×1 s），真失败（如无麦克风）立即退出不空转；同时修正 `DeclareLaunchArgument` 里残留的 `/home/topeet/...` 默认值 |

### 1.1 split-9 解码做了什么

```
rknn_outputs_get(want_float=0)          # 直接取 int8，省掉 runtime 的 INT8→FP32（约 9 ms/帧）
  → 按尺度分组（80×80 / 40×40 / 20×20）
  → sum 通道预筛（sum ≥ 最大类分数 ⇒ sum 低时不可能有目标）
  → 候选 anchor 做 80 类 argmax（整数比较）+ DFL（16 bin softmax 加权 → l/t/r/b）
  → 坐标映射 + 逐类 NMS（IoU 0.45）
```

无 `sum` 通道时会退化为「类别外层、anchor 内层」的全量扫描（cache 友好）。

## 二、性能实测（RK3588，aarch64）

### 2.1 端到端分段（768×432 UYVY 帧 → 检测结果）

| 阶段 | yolo11n | yolo11m |
|---|---|---|
| RGA（UYVY→RGB + resize） | 3.5 ms | 3.9 ms |
| NPU 前向（`rknn_run`） | **42 ms** | **107 ms** |
| 解码（DFL + NMS） | ~6.5 ms | ~6.5 ms |
| **合计** | **51.6 ms（19.4 fps）** | 114.6 ms（8.7 fps） |

同一测试视频（行人场景）检出结果几乎一致（person 置信度 0.852 vs 0.838，框位置差 1~3 px），
所以默认使用 **yolo11n**；需要更强的小目标能力时再切 `yolo11m`。

### 2.2 解码优化历程（yolo11m 固定模型）

| 版本 | 总耗时 | fps | 说明 |
|---|---|---|---|
| 初始（`want_float=1`，重复缩放） | 191.7 ms | 5.2 | DFL 后处理按 float 逐点读 |
| `want_float=0` + argmax 重排 | 184.0 ms | 5.4 | 省 9 ms 转换 + 改善 cache 局部性 |
| 去掉重复双线性缩放 | 131.1 ms | 7.6 | **最大收益项**（省约 50 ms） |
| sum 通道预筛 | 114.6 ms | 8.7 | 解码 24 ms → 6.5 ms |

### 2.3 NPU 核数说明

RK3588 有 3 个 NPU 核，但**公开的 YOLO rknn 模型全部是单核导出**（`npu_cores: 1`）：

- 运行时 `rknn_set_core_mask(RKNN_NPU_CORE_0_1_2)` 返回 0，但耗时无变化（模型已按单核编译）
- 想用满 3 核必须在 x86 上用 `rknn-toolkit2` 重新导出（`core_mask` 配置）
- LLM 侧不受影响（`rkllm` 日志显示 `npu_core_num: 3`）

## 三、常见问题与注意事项

| 问题 | 现象 | 处理 |
|---|---|---|
| `run.sh` 仍是原机写法 | 引用 `~/code/...`、`amixer -c 2`、`'Line 2'` | 本机应为 `/ws/src/...`、`amixer -c 1`、`'Line 2L'` |
| 残留进程 | 反复 launch 后新节点抢不到 NPU/模型，报莫名错误 | 每次测试前 `docker restart ros2-humble`（`kill` launch 进程不会杀掉子节点） |
| 首次启动的转换竞态 | 个别节点停在 `unconfigured`/`inactive` | 已在 launch 中加入瞬时错误重试（见 1.8），正常情况可自愈；若超 30 次仍失败，多半是模型/设备不可用，看节点日志 |
| swap | 内存紧张时 RGA/LLM 有风险 | 已设 `memlock: -1`；重负载场景留意 `free -h` |
| 模型不在仓库 | clone 后无 `.rkllm` / `.rknn` | 按「二、使用到的模型」下载并放到 `~/models/rkllm_weight/` |
