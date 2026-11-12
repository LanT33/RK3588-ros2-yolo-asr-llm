// ============================================================================
// yolo_split9_decode.cpp — YOLO11 split-9 输出解码实现
//
// 流程：rknn_outputs_get(9, want_float=0) → 按尺度分组 → sum 预筛 → argmax
//       → DFL 解码 → NMS
//
// 性能优化（相对最初的 want_float=1 朴素实现，实测 191.7ms → ~110ms 的解码部分）：
//   1. want_float=0：直接读 int8 量化输出，省掉 runtime 把 ~2.8M 元素转 FP32
//   2. sum 通道预筛：sum = 各类分数之和 ≥ 最大类分数，故 sum < 阈值时可安全跳过；
//      只对候选 anchor 做 80 类 argmax（从 8400×80 降到 候选×80）
//   3. 无 sum 通道时退化为「类别外层、anchor 内层」扫描（cache 友好，避免跳读）
//   4. 阈值在量化域比较（整数比较），仅对候选反量化
//
// 兼容：若模型输出不是 INT8（如 FP16），自动走 float 直读路径。
// ============================================================================
#include "yolo/yolo_split9_decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "rknn_api.h"

namespace yolo_split9 {
namespace {

constexpr int kNumClass = 80;
constexpr int kDflLen = 16;           // DFL 分布 bin 数
constexpr int kNumScales = 3;
constexpr int kStrides[kNumScales] = {8, 16, 32};

// 与 yolo_engine.cpp 中的类名表保持一致（该表为 static，此处单独复制一份）
const char* const kCocoNames[kNumClass] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
};

struct RawBox {
    float x1, y1, x2, y2, conf;
    int cls;
};

/// 单个输出张量（含反量化参数），布局统一按 [C,H,W] 访问
struct TensorRef {
    const void* data = nullptr;
    bool quant = false;      // true: int8 + scale/zp；false: float32
    float scale = 1.f;
    int32_t zp = 0;
    int c = 0, h = 0, w = 0;

    inline float at(int idx) const {
        if (quant) {
            return (static_cast<int32_t>(static_cast<const int8_t*>(data)[idx]) - zp) * scale;
        }
        return static_cast<const float*>(data)[idx];
    }

    /// 阈值换算到量化域（向上取整，避免漏掉临界样本）
    inline int QuantThreshold(float thres) const {
        if (!quant || scale <= 0.f) return -128;
        return static_cast<int>(std::ceil(thres / scale)) + zp;
    }
};

float IoU(const RawBox& a, const RawBox& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    if (x1 >= x2 || y1 >= y2) return 0.f;
    float inter = (x2 - x1) * (y2 - y1);
    float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return inter / (uni + 1e-6f);
}

// DFL：16 个 bin 的 softmax 加权和，返回该方向的偏移量（单位：stride 的倍数）
inline float DflDistance(const TensorRef& t, int base_idx, int plane) {
    float v[kDflLen];
    float mx = -1e30f;
    for (int i = 0; i < kDflLen; ++i) {
        v[i] = t.at(base_idx + i * plane);
        if (v[i] > mx) mx = v[i];
    }
    float sum = 0.f, acc = 0.f;
    for (int i = 0; i < kDflLen; ++i) {
        float e = std::exp(v[i] - mx);
        sum += e;
        acc += e * static_cast<float>(i);
    }
    return (sum > 0.f) ? (acc / sum) : 0.f;
}

// 单 anchor 的 80 类 argmax（分类张量 [80,H,W]，按 c*plane+idx 取）
inline void ArgmaxAt(const TensorRef& c, int idx, int plane, float* out_conf, int* out_cls) {
    if (c.quant) {
        const int8_t* q = static_cast<const int8_t*>(c.data);
        int best_q = -1000, best_c = 0;
        for (int ch = 0; ch < kNumClass; ++ch) {
            int v = q[static_cast<size_t>(ch) * plane + idx];
            if (v > best_q) { best_q = v; best_c = ch; }
        }
        *out_conf = (static_cast<float>(best_q) - c.zp) * c.scale;
        *out_cls = best_c;
    } else {
        const float* f = static_cast<const float*>(c.data);
        float best = -1e30f;
        int best_c = 0;
        for (int ch = 0; ch < kNumClass; ++ch) {
            float v = f[static_cast<size_t>(ch) * plane + idx];
            if (v > best) { best = v; best_c = ch; }
        }
        *out_conf = best;
        *out_cls = best_c;
    }
}

int StrideOf(int gh, int gw) {
    const int x = std::max(gh, gw);
    if (x == 80) return 8;
    if (x == 40) return 16;
    if (x == 20) return 32;
    return 0;
}

}  // namespace

bool Supports(int n_output) {
    return n_output == kNumScales * 3;
}

std::vector<Detection> RunAndDecode(void* ctx_v, int model_w, int model_h,
                                    float conf_thres, float iou_thres) {
    auto ctx = static_cast<rknn_context>(reinterpret_cast<uintptr_t>(ctx_v));

    rknn_input_output_num io{};
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) < 0) return {};
    const int n_out = static_cast<int>(io.n_output);
    if (!Supports(n_out)) return {};

    // want_float = 0：取量化原始数据，避免 runtime 做 INT8→FP32 转换
    std::vector<rknn_output> outs(n_out);
    for (int i = 0; i < n_out; ++i) {
        outs[i].index = i;
        outs[i].want_float = 0;
    }
    if (rknn_outputs_get(ctx, n_out, outs.data(), nullptr) < 0) return {};

    std::vector<TensorRef> box(3), cls(3), sc_sum(3);
    for (int i = 0; i < n_out; ++i) {
        rknn_tensor_attr attr{};
        attr.index = i;
        if (rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) < 0) continue;
        if (attr.n_dims != 4) continue;

        TensorRef t;
        t.data = outs[i].buf;
        t.c = static_cast<int>(attr.dims[1]);
        t.h = static_cast<int>(attr.dims[2]);
        t.w = static_cast<int>(attr.dims[3]);
        t.quant = (attr.type == RKNN_TENSOR_INT8) && (attr.scale > 0.f);
        t.scale = attr.scale;
        t.zp = attr.zp;

        const int stride = StrideOf(t.h, t.w);
        if (stride == 0) continue;
        if (t.h != model_h / stride || t.w != model_w / stride) continue;

        const int si = (stride == 8) ? 0 : (stride == 16 ? 1 : 2);
        if (t.c == kNumClass) cls[si] = t;
        else if (t.c == 4 * kDflLen) box[si] = t;
        else if (t.c == 1) sc_sum[si] = t;
    }

    std::vector<RawBox> raw;
    raw.reserve(512);

    for (int s = 0; s < kNumScales; ++s) {
        const TensorRef& b = box[s];
        const TensorRef& c = cls[s];
        if (!b.data || !c.data) continue;

        const int gh = b.h, gw = b.w;
        const int plane = gh * gw;
        const float stride = static_cast<float>(kStrides[s]);

        // 单 anchor 的框还原（DFL + 坐标映射），仅对通过阈值的 anchor 调用
        auto emit = [&](int idx, float conf, int best_c) {
            if (conf < conf_thres) return;

            float d[4];
            for (int k = 0; k < 4; ++k) {
                d[k] = DflDistance(b, (k * kDflLen) * plane + idx, plane);
            }

            const float cx = (static_cast<float>(idx % gw) + 0.5f) * stride;
            const float cy = (static_cast<float>(idx / gw) + 0.5f) * stride;

            RawBox rb;
            rb.x1 = std::max(0.f, std::min(cx - d[0] * stride, static_cast<float>(model_w)));
            rb.y1 = std::max(0.f, std::min(cy - d[1] * stride, static_cast<float>(model_h)));
            rb.x2 = std::max(0.f, std::min(cx + d[2] * stride, static_cast<float>(model_w)));
            rb.y2 = std::max(0.f, std::min(cy + d[3] * stride, static_cast<float>(model_h)));
            if (rb.x2 - rb.x1 < 2.f || rb.y2 - rb.y1 < 2.f) return;

            rb.conf = conf;
            rb.cls = best_c;
            raw.push_back(rb);
        };

        const TensorRef& ss = sc_sum[s];
        if (ss.data) {
            // ---- 快路径：sum 通道预筛 ----
            // sum = 80 个类别分数之和，必然 ≥ 最大类分数；
            // 因此 sum < 阈值时不可能存在达标目标，可安全跳过。
            std::vector<int> cand;
            cand.reserve(256);
            if (ss.quant) {
                const int8_t* q = static_cast<const int8_t*>(ss.data);
                const int q_th = ss.QuantThreshold(conf_thres);
                for (int i = 0; i < plane; ++i) {
                    if (static_cast<int>(q[i]) >= q_th) cand.push_back(i);
                }
            } else {
                const float* f = static_cast<const float*>(ss.data);
                for (int i = 0; i < plane; ++i) {
                    if (f[i] >= conf_thres) cand.push_back(i);
                }
            }
            for (int idx : cand) {
                float conf = 0.f;
                int best_c = 0;
                ArgmaxAt(c, idx, plane, &conf, &best_c);
                emit(idx, conf, best_c);
            }
        } else {
            // ---- 慢路径：无 sum 通道时全量扫描（类别外层，cache 友好）----
            std::vector<float> best_score(static_cast<size_t>(plane), 0.f);
            std::vector<int> best_class(static_cast<size_t>(plane), 0);

            if (c.quant) {
                std::vector<int> best_q(static_cast<size_t>(plane), -1000);
                for (int ch = 0; ch < kNumClass; ++ch) {
                    const int8_t* row = static_cast<const int8_t*>(c.data) +
                                        static_cast<size_t>(ch) * plane;
                    for (int i = 0; i < plane; ++i) {
                        int v = row[i];
                        if (v > best_q[i]) { best_q[i] = v; best_class[i] = ch; }
                    }
                }
                const float sc = (c.scale > 0.f) ? c.scale : 1.f;
                for (int i = 0; i < plane; ++i) {
                    best_score[i] = (static_cast<float>(best_q[i]) - c.zp) * sc;
                }
            } else {
                for (int ch = 0; ch < kNumClass; ++ch) {
                    const float* row = static_cast<const float*>(c.data) +
                                       static_cast<size_t>(ch) * plane;
                    for (int i = 0; i < plane; ++i) {
                        if (row[i] > best_score[i]) { best_score[i] = row[i]; best_class[i] = ch; }
                    }
                }
            }

            for (int idx = 0; idx < plane; ++idx) {
                emit(idx, best_score[idx], best_class[idx]);
            }
        }
    }

    rknn_outputs_release(ctx, n_out, outs.data());

    std::sort(raw.begin(), raw.end(),
              [](const RawBox& a, const RawBox& b) { return a.conf > b.conf; });

    std::vector<bool> keep(raw.size(), true);
    std::vector<Detection> dets;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (!keep[i]) continue;
        const RawBox& r = raw[i];
        dets.push_back({kCocoNames[r.cls], r.conf,
                        static_cast<int>(r.x1), static_cast<int>(r.y1),
                        static_cast<int>(r.x2 - r.x1), static_cast<int>(r.y2 - r.y1)});
        for (size_t j = i + 1; j < raw.size(); ++j) {
            if (keep[j] && r.cls == raw[j].cls && IoU(r, raw[j]) > iou_thres) keep[j] = false;
        }
    }
    return dets;
}

}  // namespace yolo_split9
