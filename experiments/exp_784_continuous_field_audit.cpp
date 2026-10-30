/* ============================================================================
   KARYON-CORE EXP-784: Continuous Field State vs Discrete Memory Slot Hash Audit
   ============================================================================ */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <immintrin.h>

constexpr int VOCAB_SIZE = 258;
constexpr int MAX_DIM = 256;
constexpr int MAX_LAYERS = 6;
constexpr int SPARSE_SLOTS_PER_LAYER = 65536;

inline unsigned int hash_discrete(const int* history_buf, int buf_pos, int stride, int vocab_cur) {
    int idx1 = (buf_pos - 1 + 256) % 256;
    int idx2 = (buf_pos - stride + 256) % 256;
    int idx3 = (buf_pos - 2 * stride + 256) % 256;
    int idx4 = (buf_pos - 3 * stride + 256) % 256;

    int b1 = history_buf[idx1];
    int b2 = history_buf[idx2];
    int b3 = history_buf[idx3];
    int b4 = history_buf[idx4];

    unsigned int h = (unsigned int)b1 * 1013904223u 
                   ^ (unsigned int)b2 * 1196996429u 
                   ^ (unsigned int)b3 * 3559788721u 
                   ^ (unsigned int)b4 * 2802195863u
                   ^ (unsigned int)vocab_cur * 2246822519u;
    return h % SPARSE_SLOTS_PER_LAYER;
}

inline float avx2_dot_product_256(const float* a, const float* b) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < 256; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

void generate_hadamard_256(std::vector<float>& basis) {
    basis.assign(VOCAB_SIZE * MAX_DIM, 1.0f);
    std::vector<float> H(256 * 256, 1.0f);
    for (int len = 1; len < 256; len <<= 1) {
        for (int i = 0; i < 256; i += (len << 1)) {
            for (int j = 0; j < len; ++j) {
                for (int k = 0; k < len; ++k) {
                    float u = H[(i + j) * 256 + k];
                    float v = H[(i + len + j) * 256 + k];
                    H[(i + j) * 256 + k] = u;
                    H[(i + j) * 256 + len + k] = u;
                    H[(i + len + j) * 256 + k] = v;
                    H[(i + len + j) * 256 + len + k] = -v;
                }
            }
        }
    }
    for (int v = 0; v < 256; ++v) {
        for (int d = 0; d < 256; ++d) {
            basis[v * MAX_DIM + d] = H[v * 256 + d];
        }
    }
    for (int d = 0; d < 256; ++d) {
        basis[256 * MAX_DIM + d] = 1.0f;
        basis[257 * MAX_DIM + d] = -1.0f;
    }
}

int main() {
    std::string dataset;
    std::vector<std::string> files = {"karyon_core.h", "karyon_agent_runtime/agent_core.py"};
    for (const auto& fname : files) {
        std::ifstream f(fname);
        if (f.is_open()) {
            std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            dataset += content;
        }
    }
    while (dataset.length() < 100000) dataset += dataset;
    dataset = dataset.substr(0, 100000);

    std::cout << "Dataset length: " << dataset.length() << " bytes." << std::endl;

    std::vector<int> history_ring(256, 256);
    std::vector<bool> slot_visited(SPARSE_SLOTS_PER_LAYER, false);
    int unique_slots_hit = 0;

    for (int t = 0; t < (int)dataset.length(); ++t) {
        int current_byte = (unsigned char)dataset[t];
        history_ring[t % 256] = current_byte;

        unsigned int slot = hash_discrete(history_ring.data(), t % 256, 1, current_byte);
        if (!slot_visited[slot]) {
            slot_visited[slot] = true;
            unique_slots_hit++;
        }
    }

    std::cout << "--- MEMORY CONTAINER DIAGNOSTICS ---" << std::endl;
    std::cout << "Total Allocated Slots per Layer : " << SPARSE_SLOTS_PER_LAYER << std::endl;
    std::cout << "Total Unique Slots Touched (100k stream): " << unique_slots_hit << " / " << SPARSE_SLOTS_PER_LAYER << std::endl;
    std::cout << "Memory Utilization Rate        : " << (100.0f * unique_slots_hit / SPARSE_SLOTS_PER_LAYER) << " %" << std::endl;
    std::cout << "Sparsity Overhead (Empty Slots): " << (100.0f * (SPARSE_SLOTS_PER_LAYER - unique_slots_hit) / SPARSE_SLOTS_PER_LAYER) << " %" << std::endl;

    return 0;
}
