#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>

int main() {
    constexpr size_t TILE_ELEMS = 32768; // 256 KB tile (fits inside L2 cache per core)
    constexpr size_t TOTAL_FRAMES = 500000000ULL;

    uint64_t* sensory_stream = nullptr;
    uint64_t* motor_stream = nullptr;
    posix_memalign((void**)&sensory_stream, 64, TOTAL_FRAMES * sizeof(uint64_t));
    posix_memalign((void**)&motor_stream, 64, TOTAL_FRAMES * sizeof(uint64_t));

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < TOTAL_FRAMES; ++i) {
        sensory_stream[i] = i ^ 0xAAAAAAAAAAAAAAAAULL;
    }

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        size_t chunk_size = TOTAL_FRAMES / num_threads;
        size_t start_idx = tid * chunk_size;
        size_t end_idx = (tid == num_threads - 1) ? TOTAL_FRAMES : start_idx + chunk_size;

        for (size_t t = start_idx; t < end_idx; t += TILE_ELEMS) {
            size_t count = std::min(TILE_ELEMS, end_idx - t);
            size_t simd_limit = count & ~3ULL;
            const uint64_t* __restrict__ in_ptr = sensory_stream + t;
            uint64_t* __restrict__ out_ptr = motor_stream + t;

            for (size_t i = 0; i < simd_limit; i += 4) {
                __m256i v_sensory = _mm256_load_si256((const __m256i*)(in_ptr + i));
                __m256i v_state = v_sensory;
                #pragma unroll
                for (int rot : {7, 13, 21, 29, 37, 43, 51, 57}) {
                    __m256i v_left  = _mm256_slli_epi64(v_state, rot);
                    __m256i v_right = _mm256_srli_epi64(v_state, 64 - rot);
                    __m256i v_rot   = _mm256_or_si256(v_left, v_right);
                    v_state = _mm256_xor_si256(v_state, v_rot);
                }
                _mm256_store_si256((__m256i*)(out_ptr + i), v_state);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();
    double total_bytes = TOTAL_FRAMES * sizeof(uint64_t) * 2.0;
    double gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / sec;

    std::cout << "Pipeline Cache-Tiled Throughput: " << gb_s << " GB/s in " << sec << " s\n";

    free(sensory_stream);
    free(motor_stream);
    return 0;
}
