// ============================================================================
// yolo_split9_decode.hpp — YOLO11 split-9 输出解码（RKNN 官方导出格式）
//
// 适用模型：airockchip/ultralytics_yolo11 导出的 RKNN 模型，共 9 个输出：
//   P3 (stride 8,  80x80): box[1,64,80,80]  cls[1,80,80,80]  sum[1,1,80,80]
//   P4 (stride 16, 40x40): box[1,64,40,40]  cls[1,80,40,40]  sum[1,1,40,40]
//   P5 (stride 32, 20x20): box[1,64,20,20]  cls[1,80,20,20]  sum[1,1,20,20]
//
// - box 的 64 通道 = 4 个方向 × 16 bin 的 DFL 分布，需 softmax 加权还原 l/t/r/b
// - cls 的 80 通道已在模型内做过 sigmoid，可直接当概率用
// - sum 通道（1 通道）是各类分数之和，本实现不依赖它（用 max 更准确）
//
// 与 yolo_engine.cpp 里原有的 split-2 解码（bbox[4,8400] + score[80,8400]）并存，
// 由 YoloEngine::Config::output_mode 开关选择。
// ============================================================================
#pragma once

#include <vector>

#include "yolo/yolo_engine.hpp"  // Detection

namespace yolo_split9 {

/// 该解码器是否支持给定的输出张量个数（split-9 固定为 9）
bool Supports(int n_output);

/// 取输出 → DFL 解码 → NMS
/// 调用方需先完成 rknn_inputs_set + rknn_run；本函数内部做 rknn_outputs_get/release
/// @param ctx        rknn_context（用 void* 传入，避免本头文件依赖 rknn_api.h）
/// @param model_w/h  模型输入宽高（如 640）
/// @return 检测框，坐标位于 model_w × model_h 空间（与 split-2 路径保持一致）
std::vector<Detection> RunAndDecode(void* ctx, int model_w, int model_h,
                                    float conf_thres, float iou_thres);

}  // namespace yolo_split9
