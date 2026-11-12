# RK3588 yolo_asr_llm

本项目受 [hejianglin2001/rk3588_voice_assistant_ros2](https://github.com/hejianglin2001/rk3588_voice_assistant_ros2) 启发，
在原项目基础上根据**dc-a588开发板硬件与驱动环境**做了可行性适配，并添加了新功能。

原始设计面向Ubuntu环境；本项目实际落地的板子为 Debian 11（无 ROS 2 官方二进制包），
因此整体迁移到**Docker**内运行，并解决了容器化后 NPU / RGA / 声卡 / DDS 的适配问题。

```
麦克风 → VAD 断句 → ASR（sherpa-onnx） → LLM（Qwen3.5-0.8B, NPU）
                        ↓ 视觉请求
                    YOLO（RKNN, NPU） ← 摄像头 / 视频文件
```

> 适配改动清单、性能实测数据与已知问题记录**[docs/ADAPTATION.md](docs/ADAPTATION.md)**。

---

## 一、目录结构

```
rk3588_voice_assistant_ros2/
├── build.sh / run.sh                 # 板端编译 / 启动脚本（原项目，路径需按本机调整）
├── docs/
│   └── ADAPTATION.md                 # 适配改动 / 性能实测 / 已知问题
├── src/
│   ├── rk3588_voice_assistant_interfaces/     # 自定义 msg / action
│   └── rk3588_voice_assistant/
│       ├── config/params.yaml                 # 所有参数
│       ├── launch/voice_assistant.launch.py   # 四节点编排 + lifecycle 转换
│       ├── nodes/                             # llm / asr / audio_vad / yolo / decision
│       ├── engine/
│       │   ├── audio/  asr/  llm/  vad/
│       │   └── yolo/
│       │       ├── camera_capture.{h,cpp}     # V4L2 采集 + 视频文件源
│       │       ├── yolo_engine.{hpp,cpp}      # RKNN 推理 + 输出模式开关
│       │       ├── yolo_split9_decode.{hpp,cpp}  # ★ 新增：YOLO11 split-9 解码
│       │       └── rga_convert.h              # RGA UYVY→RGB
│       └── third_party/                       # rknn / rkllm / rga / sherpa-onnx / miniaudio / fvad
└── README.md
```

## 二、使用到的外设

| 外设 | 型号 / 说明 | 状态 |
|---|---|---|
| 主控板 | RK3588（设备树 model：`ztl, A588`），8 核 CPU + **3 核 NPU（6 TOPS）** + RGA | 正常 |
| 声卡 | 板载 **ES8388**（声卡名 `rockchipes8388`，实际绑定 `ES8323` 兼容驱动，`card 1`） | 可用，语音链路待联调 |
| 麦克风 | 板载差分主麦（ES8388 PGA 路由，`amixer -c 1`） | 待联调 |
| 摄像头 | IMX415，`/dev/video11`，UYVY| 暂时用视频代替 |
| 网络 | 无线网卡 `wlan1`（DHCP，当前 `10.135.118.108/16`） | 正常 |
| 存储 | eMMC 57 GB| 正常 |
| NPU/RGA 设备 | `/dev/dri`、`/dev/dma_heap`、`/dev/rga`、`/dev/mpp_service` | 已映射进容器 |

## 三、使用到的模型

| 模型 | 用途 | 规格 | 来源 / 体积 |
|---|---|---|---|
| `qwen3.5-0.8b_w8a8_rk3588.rkllm` | 对话 / 意图抽词 | Qwen3.5-0.8B，**W8A8**，RKLLM runtime 1.3.0，NPU 3 核 | [t-firefly/qwen3.5-0.8b-rkllm-rk3588](https://huggingface.co/t-firefly/qwen3.5-0.8b-rkllm-rk3588)，1.3 GB |
| `yolo11n.rknn` | 目标检测（当前使用） | YOLO11n，640×640，W8A8，COCO 80 类，RKNN 2.4.0 | [RKNNAI/RK3588-CNN-yolo11n](https://huggingface.co/RKNNAI/RK3588-CNN-yolo11n)，4.1 MB |
| `yolo11m.rknn` | 目标检测（可选，精度更高） | YOLO11m，640×640，W8A8 | [RKNNAI/RK3588-CNN-yolo11m](https://huggingface.co/RKNNAI/RK3588-CNN-yolo11m)，22 MB |
| `zh-small`（sherpa-onnx） | 中文流式 ASR | Paraformer/Transducer INT8（encoder/decoder/joiner + tokens.txt） | **仓库内自带** `third_party/sherpa-onnx/models/zh-small/`，约 80 MB |

部分模型需另外部署：

```bash
mkdir -p ~/models/rkllm_weight && cd ~/models/rkllm_weight

# LLM（1.3 GB）
curl -L -o qwen3.5-0.8b_w8a8_rk3588.rkllm \
  https://huggingface.co/t-firefly/qwen3.5-0.8b-rkllm-rk3588/resolve/main/qwen3.5-0.8b_w8a8_rk3588.rkllm

# YOLO11n / YOLO11m
curl -L -o yolo11n.rknn \
  https://huggingface.co/RKNNAI/RK3588-CNN-yolo11n/resolve/main/yolo11n-640x640-w8a8-1/yolo11n.rknn
curl -L -o yolo11m.rknn \
  https://huggingface.co/RKNNAI/RK3588-CNN-yolo11m/resolve/main/yolo11m-640x640-w8a8-1/yolo11m.rknn
```

校验值（SHA256）：

```
qwen3.5-0.8b_w8a8_rk3588.rkllm : d55074d6e0fb540bec4f757ed6acad141e96d2d7810a9819ac2d401afe9f3956
yolo11m.rknn                   : 0e9c53fc08e8d6e63c00b943f2490f880817071c2e277b2f9915dc3ff6d65b3b
```

## 四、运行环境
```
宿主机    : Debian GNU/Linux 11 (bullseye) / aarch64 / kernel 5.10.226 / cgroup v2
容器基础  : ros:humble-ros-base（Ubuntu 22.04，digest sha256:d9bcf2e6…44aa855）
ROS 2     : Humble
RKNN 运行时: 2.3.2（third_party/rknn/aarch64/librknnrt.so）
RKLLM 运行时: 1.3.0（third_party/rkllm/aarch64/librkllmrt.so）
rknpu 驱动 : 0.9.8
```

## 五、编译与运行

### 5.1 容器准备
如果使用docker版ROS2，需要自行下载，下载后部署：

```bash
docker build -t ros2-humble-dev:local .     # 首次构建约 5~10 分钟
docker-compose up -d                        # 启动常驻容器（restart: unless-stopped）
docker-compose exec ros2 bash               # 进入容器（已自动 source ROS 环境）
```

### 5.2 编译

```bash

cd /ws/src/rk3588_voice_assistant_ros2
source /opt/ros/humble/setup.bash
colcon build                 # 首次全量
# 或仅主包：colcon build --packages-select rk3588_voice_assistant
```

> 注意：`colcon build` 会把 `config/` 复制到 `install/`，**改完 `params.yaml` 必须重新 build** 才生效。

### 5.3 运行

```bash

source /opt/ros/humble/setup.bash
source /ws/src/rk3588_voice_assistant_ros2/install/setup.bash
cd /ws/src/rk3588_voice_assistant_ros2
ros2 launch rk3588_voice_assistant voice_assistant.launch.py
```

启动顺序（launch 内已编排）：LLM 加载（约 15~20 s，1.3 GB）→ LLM/ASR/YOLO `configure` + `activate` → decision_node 就绪。

验证：

```bash
ros2 node list                      # 应有 5 个节点
ros2 lifecycle get /llm_node        # active [3]
ros2 lifecycle get /yolo_node       # active [3]

# 视觉检测（走完整 action 链路）
ros2 action send_goal /yolo_detect \
  rk3588_voice_assistant_interfaces/action/YoloDetect \
  "{target_class: person, duration_sec: 8}"

# LLM 文本推理
ros2 topic pub --once /text_input std_msgs/msg/String "{data: 你好}"
ros2 topic echo /llm_response --once
```

## 六、参数说明（`src/rk3588_voice_assistant/config/params.yaml`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `/llm_node.model_path` | `~/models/rkllm_weight/qwen3.5-0.8b_w8a8_rk3588.rkllm` | RKLLM 模型路径 |
| `/llm_node.max_tokens` / `context_len` | 128 / 512 | 生成上限与上下文长度 |
| `/asr_node.asr_model_dir` | `/ws/.../third_party/sherpa-onnx/models/zh-small` | 需含 encoder/decoder/joiner + tokens.txt |
| `/audio_vad_node.mic_device` | `plughw:1,0` | 板载 ES8388 为 card 1 |
| `/yolo_node.model_path` | `~/models/rkllm_weight/yolo11n.rknn` | 换 `yolo11m.rknn` 即切换模型 |
| `/yolo_node.output_mode` | `auto` | `auto` / `split9`（YOLO11）/ `split2`（原模型） |
| `/yolo_node.camera_source` | `file` | `device`=V4L2 摄像头，`file`=裸 UYVY 帧序列 |
| `/yolo_node.camera_width/height` | 768 / 432 | **必须与视频文件分辨率一致**（用摄像头时填相机输出尺寸） |
| `/yolo_node.video_file` / `video_fps` / `video_loop` | 见 yaml | 文件源路径、播放帧率、是否循环 |
| `/yolo_node.save_dir` | `/ws/yolo_out` | 命中目标时落盘 JPG 的目录 |

### 6.1 视频文件源制作（裸 UYVY 帧序列）

```bash
# mp4 → 裸 UYVY（RK 硬件解码；需 video 组权限访问 /dev/mpp_service）
gst-launch-1.0 -e filesrc location=test.mp4 ! qtdemux ! h264parse ! mppvideodec \
  ! videoconvert ! video/x-raw,format=UYVY,width=768,height=432 ! filesink location=full.yuv

# 截取前 10 秒（12 fps → 120 帧，每帧 768*432*2 = 663552 字节）
head -c $((120 * 663552)) full.yuv > clip_10s.yuv
```

## 七、相关项目

- 灵感来源：[hejianglin2001/rk3588_voice_assistant_ros2](https://github.com/hejianglin2001/rk3588_voice_assistant_ros2)
- 模型：Rockchip `rknn-llm` / `rknn_model_zoo`、[t-firefly](https://huggingface.co/t-firefly)、[RKNNAI](https://huggingface.co/RKNNAI)、k2-fsa `sherpa-onnx`
- 推理栈：RKNN-Toolkit2 Runtime、RKLLM Runtime、Rockchip RGA
