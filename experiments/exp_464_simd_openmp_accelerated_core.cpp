// experiments/exp_464_simd_openmp_accelerated_core.cpp — EXP-464: OpenMP & SIMD AVX2 Vectorized Acceleration
// Standard: KEP v16.0 Sovereign Master & Principle 1 (C++20 & High-Performance Parallel Execution)
// Zero-Python / Zero-Torch / Pure C++20 Sovereign Architecture
//
// HYPOTHESIS:
// Vectorizing core matrix-vector multiplications via AVX2 FMA instructions (#pragma omp simd / immintrin.h)
// and parallelizing multi-head associative memory reads across OpenMP threads will accelerate C++ substrate
// throughput from ~1,400 steps/s to >10,000 steps/s on CPU without degrading prediction accuracy.

#include "../karyon_native_core.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <omp.h>

namespace karyon {

class AcceleratedSIMDEngine : public SovereignNativeEngine {
public:
    AcceleratedSIMDEngine(const NativeConfig& cfg) : SovereignNativeEngine(cfg) {}

    // AVX2 SIMD Accelerated MatVec Multiplication
    void simdfma_matvec_mul(const float* W, const float* x, float* y, size_t rows, size_t cols) {
        #pragma omp parallel for schedule(static) if(rows >= 32)
        for (size_t i = 0; i < rows; ++i) {
            const float* w_row = W + i * cols;
            __m256 acc = _mm256_setzero_ps();
            size_t j = 0;

            for (; j + 8 <= cols; j += 8) {
                __m256 w_vec = _mm256_loadu_ps(w_row + j);
                __m256 x_vec = _mm256_loadu_ps(x + j);
                acc = _mm256_fmadd_ps(w_vec, x_vec, acc);
            }

            // Horizontal sum of AVX2 register
            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

            for (; j < cols; ++j) {
                sum += w_row[j] * x[j];
            }
            y[i] = sum;
        }
    }

    // Benchmark ultra-fast SIMD step execution
    StepMetrics fast_simd_step(uint8_t byte_in, uint8_t byte_target) {
        StepMetrics metrics;
        const float* emb_in = byte_emb_.data() + byte_in * config_.dim_emb;

        std::vector<float> q(config_.num_heads * config_.dim_k);
        std::vector<float> k(config_.num_heads * config_.dim_k);
        std::vector<float> v(config_.num_heads * config_.dim_v);

        simdfma_matvec_mul(W_q_.data(), emb_in, q.data(), config_.num_heads * config_.dim_k, config_.dim_emb);
        simdfma_matvec_mul(W_k_.data(), emb_in, k.data(), config_.num_heads * config_.dim_k, config_.dim_emb);
        simdfma_matvec_mul(W_v_.data(), emb_in, v.data(), config_.num_heads * config_.dim_v, config_.dim_emb);

        normalize_heads(q.data(), config_.num_heads, config_.dim_k);
        normalize_heads(k.data(), config_.num_heads, config_.dim_k);

        // OpenMP Parallel Memory Head Read
        std::vector<float> y_read(config_.num_heads * config_.dim_v, 0.0f);

        #pragma omp parallel for schedule(static)
        for (size_t h = 0; h < config_.num_heads; ++h) {
            const float* sh = S_memory_.data() + h * config_.dim_k * config_.dim_v;
            const float* qh = q.data() + h * config_.dim_k;
            float* yh = y_read.data() + h * config_.dim_v;

            for (size_t v_idx = 0; v_idx < config_.dim_v; ++v_idx) {
                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx < config_.dim_k; ++k_idx) {
                    sum += qh[k_idx] * sh[k_idx * config_.dim_v + v_idx];
                }
                yh[v_idx] = sum;
            }
        }

        std::vector<float> final_logits(config_.dim);
        simdfma_matvec_mul(W_out_.data(), y_read.data(), final_logits.data(), config_.dim, config_.num_heads * config_.dim_v);

        float max_l = final_logits[0];
        uint8_t best_byte = 0;
        for (size_t i = 1; i < config_.dim; ++i) {
            if (final_logits[i] > max_l) {
                max_l = final_logits[i];
                best_byte = static_cast<uint8_t>(i);
            }
        }

        float sum_exp = 0.0f;
        std::vector<float> probs(config_.dim);
        for (size_t i = 0; i < config_.dim; ++i) {
            probs[i] = std::exp(final_logits[i] - max_l);
            sum_exp += probs[i];
        }
        float inv_sum = 1.0f / sum_exp;
        for (size_t i = 0; i < config_.dim; ++i) probs[i] *= inv_sum;

        float loss = -std::log(std::max(probs[byte_target], 1e-7f));

        // OpenMP SIMD Update Associative Memory
        #pragma omp parallel for schedule(static)
        for (size_t h = 0; h < config_.num_heads; ++h) {
            float* sh = S_memory_.data() + h * config_.dim_k * config_.dim_v;
            const float* kh = k.data() + h * config_.dim_k;
            const float* vh = v.data() + h * config_.dim_v;

            for (size_t k_idx = 0; k_idx < config_.dim_k; ++k_idx) {
                float q_val = kh[k_idx];
                float* srow = sh + k_idx * config_.dim_v;
                #pragma omp simd
                for (size_t v_idx = 0; v_idx < config_.dim_v; ++v_idx) {
                    srow[v_idx] += config_.lr * (q_val * vh[v_idx]);
                }
            }
        }

        metrics.loss = loss;
        metrics.pred_byte = best_byte;
        metrics.is_correct = (best_byte == byte_target);
        return metrics;
    }
};

} // namespace karyon

int main() {
    std::cout << "==================================================================\n";
    std::cout << "   KARYON-CoRE EXP-464: OPENMP & AVX2 SIMD ACCELERATION BENCHMARK\n";
    std::cout << "   Targeting >10,000 steps/s Throughput on Host CPU Substrate\n";
    std::cout << "==================================================================\n";

    karyon::NativeConfig cfg;
    cfg.dim = 258;
    cfg.num_heads = 4;
    cfg.dim_k = 16;
    cfg.dim_v = 32;
    cfg.dim_emb = 128;
    cfg.lr = 0.015f;

    karyon::AcceleratedSIMDEngine engine(cfg);

    // Build large test corpus of 20,000 bytes
    std::string chunk = "template <typename T> T simd_accelerate(T val) { return val * val; }\n";
    std::string corpus = "";
    for (int i = 0; i < 300; ++i) corpus += chunk;

    size_t total_steps = corpus.size() - 1;
    std::cout << "[SIMD STRESS] Running " << total_steps << " steps with AVX2 FMA + OpenMP...\n";

    size_t correct = 0;
    float loss_sum = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < total_steps; ++i) {
        uint8_t in_b = static_cast<uint8_t>(corpus[i]);
        uint8_t target_b = static_cast<uint8_t>(corpus[i + 1]);

        auto m = engine.fast_simd_step(in_b, target_b);
        loss_sum += m.loss;
        if (m.is_correct) correct++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double throughput = static_cast<double>(total_steps) / elapsed;

    std::cout << "\n=============================================================\n";
    std::cout << "   EXP-464 BENCHMARK TELEMETRY & DECISION VERDICT\n";
    std::cout << "=============================================================\n";
    std::cout << "Total Steps Processed      : " << total_steps << "\n";
    std::cout << "Execution Elapsed Time     : " << (elapsed * 1000.0) << " ms\n";
    std::cout << "SIMD Throughput Speed      : " << throughput << " steps/s\n";
    std::cout << "Mean Loss                  : " << (loss_sum / total_steps) << " nats\n";
    std::cout << "Accuracy                   : " << (100.0f * correct / total_steps) << " %\n";
    std::cout << "Target >10,000 steps/s Met : " << (throughput >= 10000.0 ? "YES (PASSED)" : "NO") << "\n";
    std::cout << "=============================================================\n";

    std::ofstream out_json("experiments/exp_464_telemetry.json");
    out_json << "{\n"
             << "  \"exp_id\": \"EXP-464\",\n"
             << "  \"total_steps\": " << total_steps << ",\n"
             << "  \"elapsed_ms\": " << (elapsed * 1000.0) << ",\n"
             << "  \"throughput_steps_per_sec\": " << throughput << ",\n"
             << "  \"mean_loss\": " << (loss_sum / total_steps) << ",\n"
             << "  \"accuracy\": " << (100.0f * correct / total_steps) << "\n"
             << "}\n";
    out_json.close();

    return 0;
}
