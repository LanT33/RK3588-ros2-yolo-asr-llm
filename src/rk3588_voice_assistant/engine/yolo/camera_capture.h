// camera_capture.h — V4L2 极简摄像头采集 (ponytail: 10 行接口)
#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <sys/types.h>
#include <linux/videodev2.h>

struct CameraFrame {
    uint8_t* data;   // mmap buffer
    size_t   size;
};

class CameraCapture {
public:
    CameraCapture() = default;
    ~CameraCapture();
    CameraCapture(const CameraCapture&) = delete;
    CameraCapture& operator=(const CameraCapture&) = delete;

    /// 打开设备，设置 UYVY 格式 @ w×h，返回 true 成功
    bool Open(const std::string& device, int w, int h);
    /// 文件源模式：以裸 UYVY 帧序列文件（每帧 w*h*2 字节）作为帧源
    /// @param fps  按该帧率节流（<=0 表示不限速）
    /// @param loop 读到文件尾是否回到开头循环
    bool OpenFile(const std::string& path, int w, int h, double fps, bool loop);
    bool IsOpen() const { return fd_ >= 0 || file_fd_ >= 0; }
    bool IsFileMode() const { return file_mode_; }
    /// 开始数据流
    bool Start();
    /// 抓一帧（阻塞），成功返回有效 data，失败返回 data==nullptr
    CameraFrame Capture();
    /// 停止 + 关闭
    void Close();

private:
    int fd_ = -1;
    int w_ = 0, h_ = 0;
    struct Buffer { void* start; size_t length; };
    Buffer* buffers_ = nullptr;
    int num_bufs_ = 0;
    bool streaming_ = false;

    // ---- 文件源模式（视频替代相机）----
    bool file_mode_ = false;
    int file_fd_ = -1;
    uint8_t* file_buf_ = nullptr;
    size_t frame_bytes_ = 0;
    off_t file_size_ = 0;
    off_t file_pos_ = 0;
    double fps_ = 0.0;
    bool loop_ = true;
    int64_t next_ns_ = 0;      // 下一帧的目标时刻（CLOCK_MONOTONIC）

    CameraFrame CaptureFile();
    bool Xioctl(int req, void* arg);
};
