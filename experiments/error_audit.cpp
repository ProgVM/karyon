/* ============================================================================
   KARYON-CORE ERROR DIAGNOSTIC: 4.56% Misprediction Analysis
   ============================================================================ */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <map>
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
    std::vector<float> energies(VOCAB_SIZE, 0.0f);

    int total_errors = 0;
    int errors_first_5000 = 0;
    int errors_after_5000 = 0;

    std::map<int, int> rank_histogram;
    std::map<std::string, int> error_transitions;

    for (int t = 0; t < total_bytes - 1; ++t) {
        int current_byte = (unsigned char)dataset[t];
        int target_byte = (unsigned char)dataset[t + 1];

        const float* z_targ = &basis[target_byte * MAX_DIM];
        float* phi_v = &Phi_memory[current_byte * MAX_DIM];

        for (int i = 0; i < MAX_DIM; ++i) response[i] = phi_v[i];

        for (int v = 0; v < VOCAB_SIZE; ++v) {
            energies[v] = avx2_dot_product(&basis[v * MAX_DIM], response.data(), MAX_DIM) / 256.0f;
        }

        int rank = 0;
        float target_energy = energies[target_byte];
        for (int v = 0; v < VOCAB_SIZE; ++v) {
            if (energies[v] > target_energy) rank++;
        }

        if (rank > 0) {
            total_errors++;
            if (t < 5000) errors_first_5000++;
            else errors_after_5000++;

            rank_histogram[rank]++;

            std::string trans = "";
            trans += (current_byte >= 32 && current_byte <= 126) ? (char)current_byte : '?';
            trans += " -> ";
            trans += (target_byte >= 32 && target_byte <= 126) ? (char)target_byte : '?';
            error_transitions[trans]++;
        }

        float eta = 0.02f;
        for (int i = 0; i < MAX_DIM; ++i) {
            phi_v[i] += eta * (z_targ[i] - response[i]);
        }
    }

    std::cout << "=================================================" << std::endl;
    std::cout << "  4.56% MISPREDICTION AUDIT & DIAGNOSTIC REPORT" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Total Mispredictions : " << total_errors << " / " << (total_bytes - 1) << " (" << (100.0f * total_errors / (total_bytes - 1)) << "%)" << std::endl;
    std::cout << "Errors in First 5000 bytes (Cold Start) : " << errors_first_5000 << " (" << (100.0f * errors_first_5000 / total_errors) << "% of all errors)" << std::endl;
    std::cout << "Errors in Remaining 95000 bytes        : " << errors_after_5000 << " (" << (100.0f * errors_after_5000 / (total_bytes - 5000)) << "% error rate in warm state)" << std::endl;
    
    std::cout << "\nTarget Rank Distribution on Mispredictions:" << std::endl;
    for (int r = 1; r <= 5; ++r) {
        std::cout << "  Rank " << r << " (Top " << (r+1) << "): " << rank_histogram[r] << " times" << std::endl;
    }
    int rank_gt_5 = 0;
    for (auto const& [r, count] : rank_histogram) {
        if (r > 5) rank_gt_5 += count;
    }
    std::cout << "  Rank > 5: " << rank_gt_5 << " times" << std::endl;

    std::cout << "\nTop 10 High-Entropy Error Transitions (Where context is fundamentally ambiguous):" << std::endl;
    std::vector<std::pair<int, std::string>> top_trans;
    for (auto const& [trans, count] : error_transitions) {
        top_trans.push_back({count, trans});
    }
    std::sort(top_trans.rbegin(), top_trans.rend());
    for (int i = 0; i < std::min(10, (int)top_trans.size()); ++i) {
        std::cout << "  [" << top_trans[i].second << "] : " << top_trans[i].first << " occurrences" << std::endl;
    }
    std::cout << "=================================================" << std::endl;

    return 0;
}
