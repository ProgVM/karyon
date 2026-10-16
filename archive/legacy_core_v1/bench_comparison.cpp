#include "karyon_native_core.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

int main() {
    std::cout << "==================================================================\n";
    std::cout << "  KARYON-CoRE BENCHMARK: Pure C++20 vs Rust-Accelerated Substrate \n";
    std::cout << "  Author: Bazilevs (ProgVM) | Standard: KEP v16.0 Sovereign Master\n";
    std::cout << "==================================================================\n";

    // Generate large benchmark text stream (50,000 bytes)
    std::string sample_corpus = 
        "Continuous field dynamics mind substrate independence active inference "
        "Ashby homeostatic ultrastability dissipative thinking contractive decay "
        "Hebbian delta outer product trace associative memory matrix scanning "
        "Karyon-CoRE sovereign autopoiesis zero-torch pure C++20 and Rust kernel.\n";

    std::vector<uint8_t> stream;
    while (stream.size() < 50000) {
        stream.insert(stream.end(), sample_corpus.begin(), sample_corpus.end());
    }
    stream.resize(50000);

    std::cout << "[INFO] Benchmark dataset size: " << stream.size() << " bytes.\n\n";

    // ------------------------------------------------------------------------
    // Benchmark 1: Pure C++20 Native Engine
    // ------------------------------------------------------------------------
    std::cout << ">>> Running Benchmark 1: Pure C++20 Native Kernel...\n";
    {
        karyon::NativeConfig cfg_cpp;
        cfg_cpp.use_rust_backend = false;
        karyon::SovereignNativeEngine engine_cpp(cfg_cpp);

        auto t_start = std::chrono::high_resolution_clock::now();
        float total_loss = 0.0f;
        size_t correct = 0;

        for (size_t i = 0; i < stream.size() - 1; ++i) {
            auto m = engine_cpp.step(stream[i], stream[i + 1]);
            total_loss += m.loss;
            if (m.is_correct) correct++;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
        double throughput = static_cast<double>(stream.size() - 1) / elapsed_s;

        std::cout << "    [C++20 Result] Elapsed Time : " << std::fixed << std::setprecision(3) << elapsed_s << " s\n";
        std::cout << "    [C++20 Result] Throughput   : " << std::fixed << std::setprecision(2) << throughput << " steps/sec\n";
        std::cout << "    [C++20 Result] Final Loss   : " << std::setprecision(4) << (total_loss / (stream.size() - 1)) << " nats\n";
        std::cout << "    [C++20 Result] Accuracy     : " << std::setprecision(2) << ((double)correct / (stream.size() - 1) * 100.0) << " %\n\n";
    }

    // ------------------------------------------------------------------------
    // Benchmark 2: Rust-Accelerated Kernel
    // ------------------------------------------------------------------------
    std::cout << ">>> Running Benchmark 2: Rust-Accelerated C-ABI Substrate Kernel...\n";
    {
        karyon::NativeConfig cfg_rust;
        cfg_rust.use_rust_backend = true;
        karyon::SovereignNativeEngine engine_rust(cfg_rust);

        auto t_start = std::chrono::high_resolution_clock::now();
        float total_loss = 0.0f;
        size_t correct = 0;

        for (size_t i = 0; i < stream.size() - 1; ++i) {
            auto m = engine_rust.step(stream[i], stream[i + 1]);
            total_loss += m.loss;
            if (m.is_correct) correct++;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
        double throughput = static_cast<double>(stream.size() - 1) / elapsed_s;

        std::cout << "    [Rust Result]  Elapsed Time : " << std::fixed << std::setprecision(3) << elapsed_s << " s\n";
        std::cout << "    [Rust Result]  Throughput   : " << std::fixed << std::setprecision(2) << throughput << " steps/sec\n";
        std::cout << "    [Rust Result]  Final Loss   : " << std::setprecision(4) << (total_loss / (stream.size() - 1)) << " nats\n";
        std::cout << "    [Rust Result]  Accuracy     : " << std::setprecision(2) << ((double)correct / (stream.size() - 1) * 100.0) << " %\n\n";
    }

    std::cout << "==================================================================\n";
    std::cout << "BENCHMARK COMPLETED SUCCESSFULLY!\n";
    std::cout << "==================================================================\n";

    return 0;
}
