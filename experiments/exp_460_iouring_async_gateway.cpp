// experiments/exp_460_iouring_async_gateway.cpp — EXP-460: Sovereign io_uring Async Byte Gateway
// Standard: KEP v16.0 Sovereign Master & Principle 1 (Hardware Acceleration) & Principle 12 (Universal Modality Substrate)
// Zero-Python / Zero-Torch / Pure C++20 Sovereign Architecture
//
// HYPOTHESIS:
// Replacing traditional blocking POSIX read/write syscalls with an asynchronous Linux io_uring submission/completion
// ring buffer (zero-syscall overhead in Kernel 6.x) will allow Karyon-CoRE to directly stream raw byte perception from disk,
// sockets, and OS pipes in non-blocking real-time, achieving >500 MB/s async throughput without CPU thread starvation.

#include "../karyon_native_core.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <liburing.h>

namespace karyon {

class SovereignIoUringGateway {
public:
    SovereignIoUringGateway(size_t queue_depth = 64) : queue_depth_(queue_depth), initialized_(false) {
        int ret = io_uring_queue_init(queue_depth_, &ring_, 0);
        if (ret == 0) {
            initialized_ = true;
        } else {
            std::cerr << "[WARN] io_uring_queue_init failed with code " << ret << ". Fallback mode active.\n";
        }
    }

    ~SovereignIoUringGateway() {
        if (initialized_) {
            io_uring_queue_exit(&ring_);
            initialized_ = false;
        }
    }

    // High-speed non-blocking asynchronous stream ingest using Linux io_uring ring
    bool async_read_stream(const std::string& filepath, std::vector<uint8_t>& out_buffer, size_t chunk_size = 4096) {
        int fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) return false;

        struct stat st;
        if (fstat(fd, &st) < 0) {
            close(fd);
            return false;
        }

        size_t file_size = st.st_size;
        out_buffer.resize(file_size);

        if (!initialized_) {
            // Fallback POSIX read if io_uring unavailable
            ssize_t bytes_read = read(fd, out_buffer.data(), file_size);
            close(fd);
            return (bytes_read == static_cast<ssize_t>(file_size));
        }

        // Asynchronous io_uring Read Queue Submission
        size_t offset = 0;
        size_t bytes_remaining = file_size;

        while (bytes_remaining > 0) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            if (!sqe) {
                // Submit pending SQEs if queue is full
                io_uring_submit(&ring_);
                sqe = io_uring_get_sqe(&ring_);
            }

            size_t bytes_to_read = std::min(bytes_remaining, chunk_size);
            io_uring_prep_read(sqe, fd, out_buffer.data() + offset, bytes_to_read, offset);
            sqe->user_data = offset;

            offset += bytes_to_read;
            bytes_remaining -= bytes_to_read;
        }

        // Submit all queued SQEs to Linux kernel ring
        io_uring_submit(&ring_);

        // Process completions (CQEs)
        size_t completed_chunks = (file_size + chunk_size - 1) / chunk_size;
        for (size_t c = 0; c < completed_chunks; ++c) {
            struct io_uring_cqe* cqe = nullptr;
            int ret = io_uring_wait_cqe(&ring_, &cqe);
            if (ret < 0) {
                close(fd);
                return false;
            }
            if (cqe->res < 0) {
                io_uring_cqe_seen(&ring_, cqe);
                close(fd);
                return false;
            }
            io_uring_cqe_seen(&ring_, cqe);
        }

        close(fd);
        return true;
    }

    bool is_active() const { return initialized_; }

private:
    struct io_uring ring_;
    size_t queue_depth_;
    bool initialized_;
};

} // namespace karyon

int main() {
    std::cout << "==================================================================\n";
    std::cout << "   KARYON-CoRE EXP-460: SOVEREIGN io_uring ASYNC GATEWAY BENCHMARK\n";
    std::cout << "   Direct Linux Kernel 6.x Zero-Syscall Ring Ingest & Learning\n";
    std::cout << "==================================================================\n";

    // Create a 10 MB temporary binary stream file for async I/O stress benchmark
    std::string test_file = "/tmp/karyon_async_stream.bin";
    {
        std::ofstream out(test_file, std::ios::binary);
        std::string block = "karyon_sovereign_io_uring_kernel_6_stream_data_test_block_0123456789\n";
        for (int i = 0; i < 150000; ++i) {
            out.write(block.data(), block.size());
        }
    }

    // Benchmark 1: Standard POSIX blocking read
    std::vector<uint8_t> posix_buf;
    auto t0 = std::chrono::high_resolution_clock::now();
    int fd = open(test_file.c_str(), O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    posix_buf.resize(st.st_size);
    ssize_t bytes_read = read(fd, posix_buf.data(), st.st_size);
    close(fd);
    auto t1 = std::chrono::high_resolution_clock::now();
    double posix_time = std::chrono::duration<double>(t1 - t0).count();
    double posix_mbps = (static_cast<double>(bytes_read) / (1024.0 * 1024.0)) / posix_time;

    std::cout << "[POSIX BASELINE] File size: " << (st.st_size / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(4) << (posix_time * 1000.0) << " ms | Speed: " << posix_mbps << " MB/s\n\n";

    // Benchmark 2: Sovereign Linux io_uring Async Ring
    karyon::SovereignIoUringGateway gateway(128);
    std::vector<uint8_t> uring_buf;

    auto t2 = std::chrono::high_resolution_clock::now();
    bool io_ok = gateway.async_read_stream(test_file, uring_buf, 65536);
    auto t3 = std::chrono::high_resolution_clock::now();

    double uring_time = std::chrono::duration<double>(t3 - t2).count();
    double uring_mbps = (static_cast<double>(uring_buf.size()) / (1024.0 * 1024.0)) / uring_time;

    std::cout << "[SOVEREIGN io_uring] Read Status: " << (io_ok ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  Time: " << (uring_time * 1000.0) << " ms | Speed: " << uring_mbps << " MB/s\n";
    std::cout << "  Speedup Delta: " << (uring_mbps / posix_mbps) << " x\n\n";

    // Execute single-pass learning over io_uring stream
    std::cout << "[LEARNING PHASE] Executing single-pass online learning over io_uring stream...\n";
    karyon::NativeConfig cfg;
    cfg.dim = 258;
    cfg.num_heads = 4;
    cfg.dim_k = 16;
    cfg.dim_v = 32;
    cfg.dim_emb = 128;
    cfg.lr = 0.015f;

    karyon::SovereignNativeEngine engine(cfg);
    size_t sample_steps = std::min(uring_buf.size() - 1, static_cast<size_t>(3000));
    size_t correct = 0;
    float loss_sum = 0.0f;

    auto t_learn0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < sample_steps; ++i) {
        auto m = engine.step(uring_buf[i], uring_buf[i + 1]);
        loss_sum += m.loss;
        if (m.is_correct) correct++;
    }
    auto t_learn1 = std::chrono::high_resolution_clock::now();
    double learn_time = std::chrono::duration<double>(t_learn1 - t_learn0).count();
    double learn_throughput = static_cast<double>(sample_steps) / learn_time;

    std::cout << "  Learning Complete: " << sample_steps << " steps in " << learn_time << " s (" << learn_throughput << " steps/s)\n";
    std::cout << "  Final Loss: " << (loss_sum / sample_steps) << " nats | Accuracy: " << (100.0f * correct / sample_steps) << " %\n";

    std::cout << "\n=============================================================\n";
    std::cout << "   EXP-460 BENCHMARK TELEMETRY & DECISION VERDICT\n";
    std::cout << "=============================================================\n";
    std::cout << "POSIX Read Speed           : " << posix_mbps << " MB/s\n";
    std::cout << "io_uring Async Read Speed  : " << uring_mbps << " MB/s\n";
    std::cout << "IOPS Throughput Delta      : " << (uring_mbps / posix_mbps) << " x\n";
    std::cout << "Async Gateway Active       : " << (gateway.is_active() ? "TRUE" : "FALSE") << "\n";
    std::cout << "=============================================================\n";

    unlink(test_file.c_str());

    std::ofstream out_json("experiments/exp_460_telemetry.json");
    out_json << "{\n"
             << "  \"exp_id\": \"EXP-460\",\n"
             << "  \"posix_mbps\": " << posix_mbps << ",\n"
             << "  \"uring_mbps\": " << uring_mbps << ",\n"
             << "  \"io_speedup_factor\": " << (uring_mbps / posix_mbps) << ",\n"
             << "  \"learn_throughput\": " << learn_throughput << ",\n"
             << "  \"uring_active\": " << (gateway.is_active() ? "true" : "false") << "\n"
             << "}\n";
    out_json.close();

    return 0;
}
