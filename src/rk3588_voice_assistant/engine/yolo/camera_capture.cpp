// camera_capture.cpp — V4L2 mmap 采集
#include "yolo/camera_capture.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

bool CameraCapture::Xioctl(int req, void* arg) {
    for (int retries = 0; retries < 3; retries++) {
        if (ioctl(fd_, req, arg) == 0) return true;
        if (errno != EINTR) break;
        usleep(1000);
    }
    return false;
}

CameraCapture::~CameraCapture() { Close(); }

bool CameraCapture::Open(const std::string& device, int w, int h) {
    // O_NONBLOCK: MIPI 相机可能中途停流，阻塞 DQBUF 会卡死整个 executor（action 永不返回）。
    // 非阻塞下 DQBUF 无帧时立即返回 EAGAIN → Capture() 返回空帧 → 上层走灰图兜底，deadline 正常退出。
    fd_ = open(device.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) { perror("open camera"); return false; }

    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_UYVY;
    fmt.fmt.pix_mp.width = w;
    fmt.fmt.pix_mp.height = h;
    if (!Xioctl(VIDIOC_S_FMT, &fmt)) { perror("VIDIOC_S_FMT"); return false; }
    w_ = fmt.fmt.pix_mp.width; h_ = fmt.fmt.pix_mp.height;

    return true;
}

bool CameraCapture::OpenFile(const std::string& path, int w, int h, double fps, bool loop) {
    file_fd_ = ::open(path.c_str(), O_RDONLY);
    if (file_fd_ < 0) { perror("open video file"); return false; }

    struct stat st {};
    if (fstat(file_fd_, &st) != 0) {
        perror("fstat video file");
        ::close(file_fd_); file_fd_ = -1; return false;
    }

    w_ = w; h_ = h;
    frame_bytes_ = static_cast<size_t>(w) * static_cast<size_t>(h) * 2;  // UYVY
    file_size_ = st.st_size;
    if (file_size_ < static_cast<off_t>(frame_bytes_)) {
        fprintf(stderr, "视频文件太小: %s (%ld < %zu)\n", path.c_str(),
                static_cast<long>(file_size_), frame_bytes_);
        ::close(file_fd_); file_fd_ = -1; return false;
    }

    file_buf_ = static_cast<uint8_t*>(malloc(frame_bytes_));
    if (!file_buf_) { ::close(file_fd_); file_fd_ = -1; return false; }

    file_pos_ = 0;
    fps_ = (fps > 0.1) ? fps : 0.0;
    loop_ = loop;
    next_ns_ = 0;
    file_mode_ = true;

    fprintf(stderr, "[camera_capture] 文件源: %s (%dx%d UYVY, %ld 帧, %.1f fps, loop=%d)\n",
            path.c_str(), w, h,
            static_cast<long>(file_size_ / static_cast<off_t>(frame_bytes_)),
            fps_, static_cast<int>(loop));
    return true;
}

CameraFrame CameraCapture::CaptureFile() {
    CameraFrame f = {nullptr, 0};

    // 按目标帧率节流（模拟实时输出），落后超过 5s 则重新对齐
    if (fps_ > 0.1) {
        struct timespec ts {};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        const int64_t now = static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
        const int64_t period = static_cast<int64_t>(1e9 / fps_);
        if (next_ns_ == 0) next_ns_ = now;
        next_ns_ += period;
        if (next_ns_ > now) {
            int64_t d = next_ns_ - now;
            struct timespec req { static_cast<time_t>(d / 1000000000LL),
                                  static_cast<long>(d % 1000000000LL) };
            nanosleep(&req, nullptr);
        } else if (now - next_ns_ > 5000000000LL) {
            next_ns_ = now;
        }
    }

    if (file_pos_ + static_cast<off_t>(frame_bytes_) > file_size_) {
        if (!loop_) return f;
        file_pos_ = 0;
    }

    ssize_t n = pread(file_fd_, file_buf_, frame_bytes_, file_pos_);
    if (n != static_cast<ssize_t>(frame_bytes_)) {
        if (!loop_) return f;
        file_pos_ = 0;
        n = pread(file_fd_, file_buf_, frame_bytes_, file_pos_);
        if (n != static_cast<ssize_t>(frame_bytes_)) return f;
    }
    file_pos_ += static_cast<off_t>(frame_bytes_);

    f.data = file_buf_;
    f.size = frame_bytes_;
    return f;
}

bool CameraCapture::Start() {
    if (file_mode_) return true;   // 文件源无需 STREAMON

    struct v4l2_requestbuffers req = {};
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;
    req.count = 4;
    if (!Xioctl(VIDIOC_REQBUFS, &req)) return false;

    num_bufs_ = req.count;
    buffers_ = new Buffer[num_bufs_];
    for (int i = 0; i < num_bufs_; i++) {
        struct v4l2_buffer buf = {};
        struct v4l2_plane plane = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.m.planes = &plane;
        buf.length = 1;
        if (!Xioctl(VIDIOC_QUERYBUF, &buf)) return false;
        buffers_[i].length = plane.length;
        buffers_[i].start = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, fd_, plane.m.mem_offset);
        if (buffers_[i].start == MAP_FAILED) return false;
    }

    for (int i = 0; i < num_bufs_; i++) {
        struct v4l2_buffer buf = {};
        struct v4l2_plane plane = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.m.planes = &plane;
        buf.length = 1;
        if (!Xioctl(VIDIOC_QBUF, &buf)) return false;
    }

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (!Xioctl(VIDIOC_STREAMON, &type)) return false;
    streaming_ = true;
    return true;
}

CameraFrame CameraCapture::Capture() {
    if (file_mode_) return CaptureFile();   // 文件源模式

    CameraFrame f = {nullptr, 0};
    struct v4l2_buffer buf = {};
    struct v4l2_plane plane = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.m.planes = &plane;
    buf.length = 1;

    if (!Xioctl(VIDIOC_DQBUF, &buf)) return f;
    f.data = static_cast<uint8_t*>(buffers_[buf.index].start);
    f.size = plane.bytesused;

    // 用完放回队列
    Xioctl(VIDIOC_QBUF, &buf);
    return f;
}

void CameraCapture::Close() {
    if (streaming_) {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        Xioctl(VIDIOC_STREAMOFF, &type);
        streaming_ = false;
    }
    if (buffers_) {
        for (int i = 0; i < num_bufs_; i++)
            if (buffers_[i].start) munmap(buffers_[i].start, buffers_[i].length);
        delete[] buffers_; buffers_ = nullptr;
    }
    if (fd_ >= 0) { close(fd_); fd_ = -1; }

    // 文件源资源
    if (file_buf_) { free(file_buf_); file_buf_ = nullptr; }
    if (file_fd_ >= 0) { ::close(file_fd_); file_fd_ = -1; }
    file_mode_ = false;
}
