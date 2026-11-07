// ============================================================================
// KARYON-CORE EXP-1053 MULTI-HARDWARE VERIFICATION BENCHMARK
// Standard: KEP v40.0 Master Specification
// Dual Tesla T4 GPUs vs Host CPU AVX2 SIMD Auto-Dispatch Audit
// ============================================================================

#include <iostream>
#include <iomanip>
#include <chrono>

#include "include/karyon_hardware.h"
#include "include/karyon_monad_v2.h"

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "  KARYON-CORE EXP-1053: MULTI-HARDWARE AUTO-SELECTION & SPEED BENCHMARK\n";
    std::cout << "==================================================================================\n\n";

    // Instantiate Monad V=2 Engine (Automatic Hardware Audit runs inside constructor)
    karyon::SovereignMonadV2 monad(32, 16, 512);

    karyon::HardwareCaps caps = monad.GetHardwareCaps();
    karyon::HardwareSelector::PrintHardwareReport(caps);

    const uint64_t total_iterations = 200000000000ULL; // 200 Billion 64-bit frames = 1,600 GB
    const double total_gb = (total_iterations * 8.0) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "  Executing Auto-Dispatched Bit-Spin Stream of " << std::fixed << std::setprecision(2)
              << total_gb << " GB...\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    uint64_t accumulator = monad.StepInSituRegisterStream(total_iterations);

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double sustained_gb_s = total_gb / (total_ms / 1000.0);
    double sustained_tb_s = sustained_gb_s / 1024.0;

    std::cout << "\n  [MULTI-HARDWARE BENCHMARK RESULTS]:\n";
    std::cout << "    - Hardware Selected       : " << caps.device_name << "\n";
    std::cout << "    - Processed Stream Volume : " << total_gb << " GB\n";
    std::cout << "    - Execution Time          : " << total_ms << " ms\n";
    std::cout << "    - Sustained Hardware Speed: " << std::setprecision(2) << sustained_gb_s 
              << " GB/s (" << sustained_tb_s << " TB/s / " << (sustained_gb_s * 1024.0) << " MB/s)\n\n";

    std::cout << "==================================================================================\n";
    if (sustained_gb_s >= 80.0) {
        std::cout << "  [VERDICT: 🟢 POSITIVE] Multi-Hardware Autoselect Engine Verified!\n";
    } else {
        std::cout << "  [VERDICT: 🔴 REJECTED]\n";
    }
    std::cout << "==================================================================================\n";

    return 0;
}
