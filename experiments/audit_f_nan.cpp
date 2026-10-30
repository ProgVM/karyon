#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <immintrin.h>

constexpr int VOCAB_SIZE = 258;
constexpr int MAX_DIM = 256;

inline float avx2_dot_product(const float* a, const float* b, int dim) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
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
    std::vector<std::string> files = {"karyon_core.h", "experiments/exp_780_multimodal_autopoietic_benchmark.cpp", "karyon_agent_runtime/agent_core.py"};
    for (const auto& fname : files) {
        std::ifstream f(fname);
        if (f.is_open()) {
            std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            dataset += content;
        }
    }
    while (dataset.length() < 100000) dataset += dataset;
    dataset = dataset.substr(0, 100000);

    int total_bytes = dataset.length();
    std::vector<float> basis(VOCAB_SIZE * MAX_DIM);
    generate_hadamard_256(basis);

    std::vector<float> Phi_memory(VOCAB_SIZE * MAX_DIM, 0.0f);
    std::vector<float> response(MAX_DIM, 0.0f);

    std::cout << "Auditing F bounds step by step:" << std::endl;
    for (int t = 0; t < 100; ++t) {
        int current_byte = (unsigned char)dataset[t];
        int target_byte = (unsigned char)dataset[t + 1];

        const float* z_targ = &basis[target_byte * MAX_DIM];
        float* phi_v = &Phi_memory[current_byte * MAX_DIM];

        for (int i = 0; i < MAX_DIM; ++i) response[i] = phi_v[i];

        float resonance = avx2_dot_product(z_targ, response.data(), MAX_DIM) / 256.0f;
        float F = 1.0f - resonance;

        if (t < 10) {
            std::cout << "t=" << t << " resonance=" << resonance << " F=" << F << std::endl;
        }

        float eta = 0.02f;
        for (int i = 0; i < MAX_DIM; ++i) {
            phi_v[i] += eta * (z_targ[i] - response[i]);
        }
    }
    return 0;
}
