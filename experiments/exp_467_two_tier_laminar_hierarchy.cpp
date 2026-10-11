// experiments/exp_467_two_tier_laminar_hierarchy.cpp — EXP-467: 2-Tier Laminar Cortical Hierarchy (Morpho-Syntax -> Semantic Discourse)
// Standard: KEP v16.0 Sovereign Master & Principle 8 (Compositional Depth Over Flat Width)
// Zero-Python / Zero-Torch / Pure C++20 Substrate Architecture with AVX2 & OpenMP

#include "../karyon_native_core.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#include <fstream>
#include <immintrin.h>
#include <omp.h>

namespace karyon {

struct LaminarConfig {
    size_t dim = 258;
    size_t num_heads = 4;
    size_t dim_k = 16;
    size_t dim_v = 32;
    size_t dim_emb = 128;
    float lr = 0.015f;

    // Layer 1: Fast Morpho-Syntax timescale
    float alpha1 = 0.85f;
    float eta_hebb1 = 0.70f;
    float eta_delta1 = 0.10f;

    // Layer 2: Slow Semantic Discourse timescale
    float alpha2 = 0.985f;
    float eta_hebb2 = 0.50f;
    float eta_delta2 = 0.05f;

    bool enable_avx2_fma = true;
};

class TwoTierLaminarEngine {
public:
    TwoTierLaminarEngine(const LaminarConfig& cfg) : cfg_(cfg) {
        init_parameters();
    }

    void reset_state() {
        std::fill(S1_.begin(), S1_.end(), 0.0f);
        std::fill(S2_.begin(), S2_.end(), 0.0f);
    }

    // Step prediction with 2-tier laminar composition
    StepMetrics step(uint8_t byte_in, uint8_t byte_target) {
        StepMetrics m;
        const float* emb_in = byte_emb_.data() + byte_in * cfg_.dim_emb;

        size_t total_dim_k = cfg_.num_heads * cfg_.dim_k;
        size_t total_dim_v = cfg_.num_heads * cfg_.dim_v;

        // --- LAYER 1: FAST MORPHO-SYNTACTIC SHEET ---
        std::vector<float> q1(total_dim_k), k1(total_dim_k), v1(total_dim_v);
        simdfma_matvec(W_q1_.data(), emb_in, q1.data(), total_dim_k, cfg_.dim_emb);
        simdfma_matvec(W_k1_.data(), emb_in, k1.data(), total_dim_k, cfg_.dim_emb);
        simdfma_matvec(W_v1_.data(), emb_in, v1.data(), total_dim_v, cfg_.dim_emb);

        normalize_heads(q1.data(), cfg_.num_heads, cfg_.dim_k);
        normalize_heads(k1.data(), cfg_.num_heads, cfg_.dim_k);

        // Readout from S1 memory
        std::vector<float> h1(total_dim_v, 0.0f);
        #pragma omp parallel for schedule(static)
        for (size_t h = 0; h < cfg_.num_heads; ++h) {
            const float* s_head = S1_.data() + h * cfg_.dim_k * cfg_.dim_v;
            const float* q_head = q1.data() + h * cfg_.dim_k;
            float* h_head = h1.data() + h * cfg_.dim_v;
            for (size_t v_idx = 0; v_idx < cfg_.dim_v; ++v_idx) {
                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx < cfg_.dim_k; ++k_idx) {
                    sum += q_head[k_idx] * s_head[k_idx * cfg_.dim_v + v_idx];
                }
                h_head[v_idx] = sum;
            }
        }

        // --- LAYER 2: SLOW SEMANTIC DISCOURSE SHEET ---
        // Project h1 to Layer 2 Query/Key/Value with LayerNorm
        layer_norm(h1.data(), total_dim_v);

        std::vector<float> q2(total_dim_k), k2(total_dim_k), v2(total_dim_v);
        simdfma_matvec(W_q2_.data(), h1.data(), q2.data(), total_dim_k, total_dim_v);
        simdfma_matvec(W_k2_.data(), h1.data(), k2.data(), total_dim_k, total_dim_v);
        simdfma_matvec(W_v2_.data(), h1.data(), v2.data(), total_dim_v, total_dim_v);

        normalize_heads(q2.data(), cfg_.num_heads, cfg_.dim_k);
        normalize_heads(k2.data(), cfg_.num_heads, cfg_.dim_k);

        // Readout from S2 memory
        std::vector<float> h2(total_dim_v, 0.0f);
        #pragma omp parallel for schedule(static)
        for (size_t h = 0; h < cfg_.num_heads; ++h) {
            const float* s_head = S2_.data() + h * cfg_.dim_k * cfg_.dim_v;
            const float* q_head = q2.data() + h * cfg_.dim_k;
            float* h_head = h2.data() + h * cfg_.dim_v;
            for (size_t v_idx = 0; v_idx < cfg_.dim_v; ++v_idx) {
                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx < cfg_.dim_k; ++k_idx) {
                    sum += q_head[k_idx] * s_head[k_idx * cfg_.dim_v + v_idx];
                }
                h_head[v_idx] = sum;
            }
        }
        layer_norm(h2.data(), total_dim_v);

        // Dual-Tier Multi-Scale Output Readout
        std::vector<float> logits1(cfg_.dim, 0.0f);
        std::vector<float> logits2(cfg_.dim, 0.0f);
        simdfma_matvec(W_out1_.data(), h1.data(), logits1.data(), cfg_.dim, total_dim_v);
        simdfma_matvec(W_out2_.data(), h2.data(), logits2.data(), cfg_.dim, total_dim_v);

        std::vector<float> final_logits(cfg_.dim);
        float max_l = -1e9f;
        for (size_t i = 0; i < cfg_.dim; ++i) {
            final_logits[i] = logits1[i] + logits2[i]; // Multi-Scale Superposition
            if (final_logits[i] > max_l) max_l = final_logits[i];
        }

        // Softmax & Cross Entropy Loss
        float sum_exp = 0.0f;
        std::vector<float> probs(cfg_.dim);
        for (size_t i = 0; i < cfg_.dim; ++i) {
            probs[i] = std::exp(final_logits[i] - max_l);
            sum_exp += probs[i];
        }
        float inv_sum = 1.0f / sum_exp;
        uint8_t best_byte = 0;
        float max_p = 0.0f;
        for (size_t i = 0; i < cfg_.dim; ++i) {
            probs[i] *= inv_sum;
            if (probs[i] > max_p) {
                max_p = probs[i];
                best_byte = static_cast<uint8_t>(i);
            }
        }

        float loss = -std::log(std::max(probs[byte_target], 1e-7f));

        // Online Multi-Timescale Plastic Updates
        // Layer 1 Update (Fast timescale)
        #pragma omp parallel for schedule(static)
        for (size_t h = 0; h < cfg_.num_heads; ++h) {
            float* s_head = S1_.data() + h * cfg_.dim_k * cfg_.dim_v;
            const float* k_head = k1.data() + h * cfg_.dim_k;
            const float* v_head = v1.data() + h * cfg_.dim_v;
            for (size_t k_idx = 0; k_idx < cfg_.dim_k; ++k_idx) {
                float q_val = k_head[k_idx];
                float* s_row = s_head + k_idx * cfg_.dim_v;
                #pragma omp simd
                for (size_t v_idx = 0; v_idx < cfg_.dim_v; ++v_idx) {
                    float delta = -cfg_.eta_delta1 * s_row[v_idx];
                    float hebb = cfg_.eta_hebb1 * (q_val * v_head[v_idx]);
                    s_row[v_idx] = cfg_.alpha1 * s_row[v_idx] + cfg_.lr * (delta + hebb);
                }
            }
        }

        // Layer 2 Update (Slow semantic timescale)
        #pragma omp parallel for schedule(static)
        for (size_t h = 0; h < cfg_.num_heads; ++h) {
            float* s_head = S2_.data() + h * cfg_.dim_k * cfg_.dim_v;
            const float* k_head = k2.data() + h * cfg_.dim_k;
            const float* v_head = v2.data() + h * cfg_.dim_v;
            for (size_t k_idx = 0; k_idx < cfg_.dim_k; ++k_idx) {
                float q_val = k_head[k_idx];
                float* s_row = s_head + k_idx * cfg_.dim_v;
                #pragma omp simd
                for (size_t v_idx = 0; v_idx < cfg_.dim_v; ++v_idx) {
                    float delta = -cfg_.eta_delta2 * s_row[v_idx];
                    float hebb = cfg_.eta_hebb2 * (q_val * v_head[v_idx]);
                    s_row[v_idx] = cfg_.alpha2 * s_row[v_idx] + cfg_.lr * (delta + hebb);
                }
            }
        }

        m.loss = loss;
        m.pred_byte = best_byte;
        m.is_correct = (best_byte == byte_target);
        m.turns_taken = 2; // 2 cortical stages traversed
        m.s_norm = compute_l2_norm(S1_.data(), S1_.size()) + compute_l2_norm(S2_.data(), S2_.size());

        return m;
    }

private:
    LaminarConfig cfg_;

    std::vector<float> byte_emb_;
    std::vector<float> W_q1_, W_k1_, W_v1_, W_out1_;
    std::vector<float> W_q2_, W_k2_, W_v2_, W_out2_;

    std::vector<float> S1_;
    std::vector<float> S2_;

    void init_parameters() {
        std::mt19937 gen(42);
        float scale1 = 1.0f / std::sqrt(static_cast<float>(cfg_.dim_emb));
        std::uniform_real_distribution<float> dist1(-scale1, scale1);

        size_t total_dim_k = cfg_.num_heads * cfg_.dim_k;
        size_t total_dim_v = cfg_.num_heads * cfg_.dim_v;

        byte_emb_.resize(cfg_.dim * cfg_.dim_emb);
        for (auto& w : byte_emb_) w = dist1(gen);

        W_q1_.resize(total_dim_k * cfg_.dim_emb);
        W_k1_.resize(total_dim_k * cfg_.dim_emb);
        W_v1_.resize(total_dim_v * cfg_.dim_emb);
        W_out1_.resize(cfg_.dim * total_dim_v);
        for (auto& w : W_q1_) w = dist1(gen);
        for (auto& w : W_k1_) w = dist1(gen);
        for (auto& w : W_v1_) w = dist1(gen);
        for (auto& w : W_out1_) w = dist1(gen);

        float scale2 = 1.0f / std::sqrt(static_cast<float>(total_dim_v));
        std::uniform_real_distribution<float> dist2(-scale2, scale2);

        W_q2_.resize(total_dim_k * total_dim_v);
        W_k2_.resize(total_dim_k * total_dim_v);
        W_v2_.resize(total_dim_v * total_dim_v);
        W_out2_.resize(cfg_.dim * total_dim_v);
        for (auto& w : W_q2_) w = dist2(gen);
        for (auto& w : W_k2_) w = dist2(gen);
        for (auto& w : W_v2_) w = dist2(gen);
        for (auto& w : W_out2_) w = dist2(gen);

        S1_.resize(cfg_.num_heads * cfg_.dim_k * cfg_.dim_v, 0.0f);
        S2_.resize(cfg_.num_heads * cfg_.dim_k * cfg_.dim_v, 0.0f);
    }

    void simdfma_matvec(const float* W, const float* x, float* y, size_t rows, size_t cols) {
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
            alignas(32) float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            for (; j < cols; ++j) {
                sum += w_row[j] * x[j];
            }
            y[i] = sum;
        }
    }

    void layer_norm(float* ptr, size_t n) {
        float mean = 0.0f;
        for (size_t i = 0; i < n; ++i) mean += ptr[i];
        mean /= n;
        float var = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float diff = ptr[i] - mean;
            var += diff * diff;
        }
        var /= n;
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        for (size_t i = 0; i < n; ++i) {
            ptr[i] = (ptr[i] - mean) * inv_std;
        }
    }

    void normalize_heads(float* ptr, size_t num_heads, size_t head_dim) {
        for (size_t h = 0; h < num_heads; ++h) {
            float* h_ptr = ptr + h * head_dim;
            float norm = compute_l2_norm(h_ptr, head_dim);
            if (norm > 1e-6f) {
                float inv_norm = 1.0f / norm;
                for (size_t i = 0; i < head_dim; ++i) h_ptr[i] *= inv_norm;
            }
        }
    }

    float compute_l2_norm(const float* ptr, size_t n) {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) sum += ptr[i] * ptr[i];
        return std::sqrt(sum);
    }
};

} // namespace karyon

int main() {
    std::cout << "==================================================================\n";
    std::cout << "   KARYON-CoRE EXP-467: 2-TIER LAMINAR CORTICAL HIERARCHY\n";
    std::cout << "   Sheet 1 (Morpho-Syntax @ alpha=0.85) -> Sheet 2 (Discourse @ alpha=0.985)\n";
    std::cout << "==================================================================\n";

    // Ingest the real-world Linux kernel C codebase stream
    std::string test_file = "/tmp/linux_karyon_core.c";
    std::ifstream file(test_file, std::ios::binary);
    std::vector<uint8_t> stream_data;
    if (file.is_open()) {
        stream_data = std::vector<uint8_t>((std::istreambuf_iterator<char>(file)),
                                            std::istreambuf_iterator<char>());
    } else {
        std::cerr << "Failed to open stream file\n";
        return 1;
    }

    std::cout << "Stream Size: " << stream_data.size() << " bytes\n";

    karyon::LaminarConfig cfg;
    cfg.enable_avx2_fma = true;
    karyon::TwoTierLaminarEngine laminar_engine(cfg);

    float total_loss = 0.0f;
    size_t correct_preds = 0;
    size_t total_steps = stream_data.size() - 1;

    float first_half_loss = 0.0f;
    float second_half_loss = 0.0f;
    size_t half_steps = total_steps / 2;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < total_steps; ++i) {
        uint8_t in_b = stream_data[i];
        uint8_t target_b = stream_data[i + 1];

        auto m = laminar_engine.step(in_b, target_b);
        total_loss += m.loss;
        if (m.is_correct) correct_preds++;

        if (i < half_steps) {
            first_half_loss += m.loss;
        } else {
            second_half_loss += m.loss;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    float mean_loss = total_loss / total_steps;
    float l1_half = first_half_loss / half_steps;
    float l2_half = second_half_loss / (total_steps - half_steps);
    float loss_drop = l2_half - l1_half;
    float accuracy = (static_cast<float>(correct_preds) / total_steps) * 100.0f;
    double throughput = (total_steps / (elapsed_ms / 1000.0));

    std::cout << "\n=============================================================\n";
    std::cout << "   EXP-467 TELEMETRY & DECISION METRICS\n";
    std::cout << "=============================================================\n";
    std::cout << "Total Steps Processed        : " << total_steps << "\n";
    std::cout << "Elapsed Time                 : " << elapsed_ms << " ms\n";
    std::cout << "Throughput Speed             : " << throughput << " steps/s\n";
    std::cout << "Overall Mean Loss            : " << mean_loss << " nats\n";
    std::cout << "First-Half Stream Loss       : " << l1_half << " nats\n";
    std::cout << "Second-Half Stream Loss      : " << l2_half << " nats\n";
    std::cout << "Single-Pass Loss Delta       : " << loss_drop << " nats\n";
    std::cout << "Online Prediction Accuracy   : " << accuracy << " %\n";
    std::cout << "=============================================================\n";

    // Write JSON telemetry
    std::ofstream out("experiments/exp_467_telemetry.json");
    out << "{\n"
        << "  \"exp_id\": \"EXP-467\",\n"
        << "  \"total_steps\": " << total_steps << ",\n"
        << "  \"elapsed_ms\": " << elapsed_ms << ",\n"
        << "  \"throughput_steps_per_sec\": " << throughput << ",\n"
        << "  \"mean_loss\": " << mean_loss << ",\n"
        << "  \"first_half_loss\": " << l1_half << ",\n"
        << "  \"second_half_loss\": " << l2_half << ",\n"
        << "  \"loss_delta\": " << loss_drop << ",\n"
        << "  \"accuracy\": " << accuracy << "\n"
        << "}\n";

    return 0;
}
