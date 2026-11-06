#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>

int main() {
    size_t chunk_size = 512 * 1024; // 4MB per thread (fits in L2/L3 cache)
    size_t total_elements = 500000000ULL;
    
    uint64_t* in_ptr = nullptr;
    uint64_t* out_ptr = nullptr;
    posix_memalign((void**)&in_ptr, 64, chunk_size * sizeof(uint64_t) * 4);
    posix_memalign((void**)&out_ptr, 64, chunk_size * sizeof(uint64_t) * 4);

    for (size_t i = 0; i < chunk_size * 4; ++i) in_ptr[i] = i ^ 0xAAAAAAAAAAAAAAAAULL;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t* t_in = in_ptr + tid * chunk_size;
        uint64_t* t_out = out_ptr + tid * chunk_size;

        size_t reps = total_elements / (chunk_size * 4);
        for (size_t r = 0; r < reps; ++r) {
            for (size_t i = 0; i < chunk_size; i += 4) {
                __m256i v = _mm256_load_si256((const __m256i*)(t_in + i));
                __m256i v_rot = _mm256_xor_si256(v, _mm256_slli_epi64(v, 13));
                _mm256_store_si256((__m256i*)(t_out + i), v_rot);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();
    double gb = (total_elements * sizeof(uint64_t) * 2.0) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "L3 Cache Blocked Bandwidth: " << (gb / sec) << " GB/s in " << sec << " s\n";

    free(in_ptr);
    free(out_ptr);
    return 0;
}
