#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
#include <cstdint>

int main() {
    constexpr uint64_t TOTAL_FRAMES = 2000000000ULL; // 2 BILLION FRAMES (128 BILLION BITS)

    auto start = std::chrono::high_resolution_clock::now();

    uint64_t total_acc = 0;

    #pragma omp parallel reduction(+:total_acc)
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        uint64_t frames_per_thread = TOTAL_FRAMES / num_threads;

        __m256i v_sensory = _mm256_set_epi64x(0x1234567890ABCDEFULL ^ tid,
                                              0xFEDCBA0987654321ULL ^ tid,
                                              0xA5A5A5A55A5A5A5AULL ^ tid,
                                              0x55AA55AA33CC33CCULL ^ tid);
        __m256i v_state = v_sensory;
        uint64_t local_sum = 0;

        for (uint64_t i = 0; i < frames_per_thread; i += 4) {
            // In-situ physical environment tick in SIMD registers
            __m256i v_step = _mm256_set1_epi64x(i);
            v_sensory = _mm256_xor_si256(v_sensory, v_step);

            // In-situ Monad SIMD Agency Transform
            __m256i v_cur = _mm256_xor_si256(v_state, v_sensory);

            #pragma unroll
            for (int rot : {7, 13, 21, 29, 37, 43, 51, 57}) {
                __m256i v_l = _mm256_slli_epi64(v_cur, rot);
                __m256i v_r = _mm256_srli_epi64(v_cur, 64 - rot);
                v_cur = _mm256_xor_si256(v_cur, _mm256_or_si256(v_l, v_r));
            }
            v_state = v_cur;

            local_sum += _mm256_extract_epi64(v_state, 0);
        }

        total_acc += local_sum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();

    double simulated_seconds = TOTAL_FRAMES / 60.0;
    double simulated_years = (simulated_seconds / (3600.0 * 24.0)) / 365.25;
    double realtime_mult = simulated_seconds / sec;

    double bytes_processed = TOTAL_FRAMES * sizeof(uint64_t) * 2.0;
    double gb_s = (bytes_processed / (1024.0 * 1024.0 * 1024.0)) / sec;

    std::cout << "In-Situ Register Stream Agency Throughput: " << gb_s << " GB/s\n";
    std::cout << "Processed 2,000,000,000 frames (" << simulated_years << " YEARS of 60Hz Life) in " << sec << " s\n";
    std::cout << "Realtime speed multiplier: " << realtime_mult << "x\n";

    return 0;
}
