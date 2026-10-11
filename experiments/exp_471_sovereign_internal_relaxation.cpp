// experiments/exp_471_sovereign_internal_relaxation.cpp
// EXP-471: Sovereign Internal Field Settling & Endogenous Adaptive Plasticity
// Investigates why Loss is not approaching zero.
// Eliminates:
// 1. Single-pass feed-forward dictate: Allows dynamic recurrent internal relaxation loops (k in 1..K)
//    until field free energy / residual error converges.
// 2. Fixed hyperparameter constants: Plasticity eta_t and temperature beta_t are endogenous field potentials.
// 3. Modality: Evaluates on repeating algorithmic structured sequence & C++ machine code.

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <iomanip>

struct ExperimentMetrics {
    std::string model_name;
    float initial_loss;
    float final_loss;
    float min_batch_loss;
    float accuracy;
    float avg_relaxation_steps;
    double steps_per_sec;
};

class SovereignFieldEngine {
public:
    size_t dim = 258;
    size_t dim_emb = 64;
    size_t num_heads = 4;
    size_t dim_k = 16;
    size_t dim_v = 16;

    std::vector<float> byte_emb;
    std::vector<float> S_memory;
    std::vector<float> theta_heads;

    // Endogenous potentials (not fixed constants)
    float internal_temp = 12.0f;
    float base_plasticity = 0.5f;

    SovereignFieldEngine() {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        byte_emb.resize(dim * dim_emb);
        for (size_t b = 0; b < dim; ++b) {
            float* eb = byte_emb.data() + b * dim_emb;
            for (size_t d = 0; d < dim_emb; ++d) eb[d] = dist(gen);
            normalize(eb, dim_emb);
        }
        S_memory.resize(num_heads * dim_k * dim_v, 0.0f);
        theta_heads.resize(num_heads);
        for (size_t h = 0; h < num_heads; ++h) {
            theta_heads[h] = -1.5f + (4.0f * h) / (num_heads - 1);
        }
    }

    void reset() {
        std::fill(S_memory.begin(), S_memory.end(), 0.0f);
        for (size_t h = 0; h < num_heads; ++h) {
            theta_heads[h] = -1.5f + (4.0f * h) / (num_heads - 1);
        }
    }

    void normalize(float* ptr, size_t n) {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) sum += ptr[i] * ptr[i];
        float norm = std::sqrt(sum);
        if (norm > 1e-6f) {
            float inv = 1.0f / norm;
            for (size_t i = 0; i < n; ++i) ptr[i] *= inv;
        }
    }

    // Baseline: Single-pass feedforward step (1 step, fixed beta, fixed eta)
    std::pair<float, bool> step_single_pass(uint8_t byte_in, uint8_t byte_target) {
        float* e_in = byte_emb.data() + byte_in * dim_emb;

        std::vector<float> k(num_heads * dim_k);
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < dim_k; ++i) k[h * dim_k + i] = e_in[h * dim_k + i];
            normalize(k.data() + h * dim_k, dim_k);
        }

        std::vector<float> h_field(dim_emb, 0.0f);
        for (size_t h = 0; h < num_heads; ++h) {
            const float* s_head = S_memory.data() + h * dim_k * dim_v;
            const float* kh = k.data() + h * dim_k;
            float* hh = h_field.data() + h * dim_v;
            for (size_t v_idx = 0; v_idx < dim_v; ++v_idx) {
                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx < dim_k; ++k_idx) {
                    sum += kh[k_idx] * s_head[k_idx * dim_v + v_idx];
                }
                hh[v_idx] = sum;
            }
        }

        for (size_t i = 0; i < dim_emb; ++i) h_field[i] = 0.70f * h_field[i] + 0.30f * e_in[i];

        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim_emb; ++i) sum_sq += h_field[i] * h_field[i];
        float inv_h_norm = 1.0f / (std::sqrt(sum_sq) + 1e-4f);

        std::vector<float> logits(dim, 0.0f);
        float max_l = -1e9f;
        for (size_t b = 0; b < dim; ++b) {
            const float* eb = byte_emb.data() + b * dim_emb;
            float dot = 0.0f;
            for (size_t d = 0; d < dim_emb; ++d) dot += h_field[d] * eb[d];
            float l = 12.0f * (dot * inv_h_norm);
            logits[b] = l;
            if (l > max_l) max_l = l;
        }

        float sum_exp = 0.0f;
        std::vector<float> probs(dim);
        for (size_t b = 0; b < dim; ++b) {
            probs[b] = std::exp(logits[b] - max_l);
            sum_exp += probs[b];
        }
        float inv_sum = 1.0f / sum_exp;
        uint8_t best_byte = 0;
        float max_p = 0.0f;
        for (size_t b = 0; b < dim; ++b) {
            probs[b] *= inv_sum;
            if (probs[b] > max_p) {
                max_p = probs[b];
                best_byte = static_cast<uint8_t>(b);
            }
        }

        float loss = -std::log(std::max(probs[byte_target], 1e-7f));
        bool is_correct = (best_byte == byte_target);

        // Update
        float* e_target = byte_emb.data() + byte_target * dim_emb;
        std::vector<float> error_wave(dim_emb);
        for (size_t d = 0; d < dim_emb; ++d) error_wave[d] = e_target[d] - h_field[d];

        for (size_t h = 0; h < num_heads; ++h) {
            float alpha_h = 1.0f / (1.0f + std::exp(-theta_heads[h]));
            float* s_head = S_memory.data() + h * dim_k * dim_v;
            const float* kh = k.data() + h * dim_k;
            const float* err_h = error_wave.data() + h * dim_v;

            for (size_t k_idx = 0; k_idx < dim_k; ++k_idx) {
                float kval = kh[k_idx];
                float* s_row = s_head + k_idx * dim_v;
                for (size_t v_idx = 0; v_idx < dim_v; ++v_idx) {
                    s_row[v_idx] = alpha_h * s_row[v_idx] + 0.45f * (kval * err_h[v_idx]);
                }
            }
        }

        for (size_t d = 0; d < dim_emb; ++d) e_target[d] += 0.05f * h_field[d];
        normalize(e_target, dim_emb);

        return {loss, is_correct};
    }

    // Sovereign Endogenous Step:
    // 1. Recurrent Internal Relaxation: field h_field recirculates k in 1..K_max steps
    // 2. Continuous Energy Minimization: Settling until dH/dt < tolerance or max iterations reached
    // 3. Dynamic Endogenous Plasticity: eta scales with error magnitude e_t
    std::tuple<float, bool, int> step_sovereign_relaxed(uint8_t byte_in, uint8_t byte_target, int max_relaxation = 5) {
        float* e_in = byte_emb.data() + byte_in * dim_emb;

        // Latent recurrent field initialized from input embedding
        std::vector<float> h_field(dim_emb);
        for (size_t d = 0; d < dim_emb; ++d) h_field[d] = e_in[d];

        int actual_steps = 0;
        float prev_energy = 1e9f;

        // Internal Relaxation Loop (Thinking / Attractor Settling)
        for (int k_step = 0; k_step < max_relaxation; ++k_step) {
            actual_steps++;

            // Keys derived from current state of the field
            std::vector<float> k(num_heads * dim_k);
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < dim_k; ++i) k[h * dim_k + i] = h_field[h * dim_k + i];
                normalize(k.data() + h * dim_k, dim_k);
            }

            // Associative readout from memory
            std::vector<float> h_readout(dim_emb, 0.0f);
            for (size_t h = 0; h < num_heads; ++h) {
                const float* s_head = S_memory.data() + h * dim_k * dim_v;
                const float* kh = k.data() + h * dim_k;
                float* hh = h_readout.data() + h * dim_v;
                for (size_t v_idx = 0; v_idx < dim_v; ++v_idx) {
                    float sum = 0.0f;
                    for (size_t k_idx = 0; k_idx < dim_k; ++k_idx) {
                        sum += kh[k_idx] * s_head[k_idx * dim_v + v_idx];
                    }
                    hh[v_idx] = sum;
                }
            }

            // Continuous Heun/Euler relaxation step:
            // dh/dt = -grad H = Memory(h) - h + ExternalInput
            float delta_h = 0.0f;
            for (size_t d = 0; d < dim_emb; ++d) {
                float target_val = 0.65f * h_readout[d] + 0.35f * e_in[d];
                float diff = target_val - h_field[d];
                h_field[d] += 0.50f * diff; // Continuous relaxation flow
                delta_h += diff * diff;
            }

            // Check if field has settled in an attractor basin
            if (std::abs(prev_energy - delta_h) < 0.005f) {
                break; // Equilibrium reached!
            }
            prev_energy = delta_h;
        }

        // Field normalization
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim_emb; ++i) sum_sq += h_field[i] * h_field[i];
        float inv_h_norm = 1.0f / (std::sqrt(sum_sq) + 1e-4f);

        // Project onto byte manifold
        std::vector<float> logits(dim, 0.0f);
        float max_l = -1e9f;
        for (size_t b = 0; b < dim; ++b) {
            const float* eb = byte_emb.data() + b * dim_emb;
            float dot = 0.0f;
            for (size_t d = 0; d < dim_emb; ++d) dot += h_field[d] * eb[d];
            float l = internal_temp * (dot * inv_h_norm);
            logits[b] = l;
            if (l > max_l) max_l = l;
        }

        float sum_exp = 0.0f;
        std::vector<float> probs(dim);
        for (size_t b = 0; b < dim; ++b) {
            probs[b] = std::exp(logits[b] - max_l);
            sum_exp += probs[b];
        }
        float inv_sum = 1.0f / sum_exp;
        uint8_t best_byte = 0;
        float max_p = 0.0f;
        for (size_t b = 0; b < dim; ++b) {
            probs[b] *= inv_sum;
            if (probs[b] > max_p) {
                max_p = probs[b];
                best_byte = static_cast<uint8_t>(b);
            }
        }

        float loss = -std::log(std::max(probs[byte_target], 1e-7f));
        bool is_correct = (best_byte == byte_target);

        // Endogenous Surprise & Plasticity:
        // Free energy error wave
        float* e_target = byte_emb.data() + byte_target * dim_emb;
        std::vector<float> error_wave(dim_emb);
        float error_norm = 0.0f;
        for (size_t d = 0; d < dim_emb; ++d) {
            error_wave[d] = e_target[d] - h_field[d];
            error_norm += error_wave[d] * error_wave[d];
        }
        error_norm = std::sqrt(error_norm);

        // Sovereign Endogenous Plasticity Rate:
        // High surprise -> high conductivity/learning; Near-zero surprise -> conservation of memory
        float dynamic_eta = std::clamp(base_plasticity * (1.0f + error_norm), 0.1f, 1.2f);

        // Adaptive temperature sharpening: If prediction is sharp, sharpen field resonance
        internal_temp = std::clamp(internal_temp + (is_correct ? 0.01f : -0.02f), 8.0f, 24.0f);

        // Update Keys
        std::vector<float> k_final(num_heads * dim_k);
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < dim_k; ++i) k_final[h * dim_k + i] = h_field[h * dim_k + i];
            normalize(k_final.data() + h * dim_k, dim_k);
        }

        // Memory Update
        for (size_t h = 0; h < num_heads; ++h) {
            float alpha_h = 1.0f / (1.0f + std::exp(-theta_heads[h]));
            float* s_head = S_memory.data() + h * dim_k * dim_v;
            const float* kh = k_final.data() + h * dim_k;
            const float* err_h = error_wave.data() + h * dim_v;

            for (size_t k_idx = 0; k_idx < dim_k; ++k_idx) {
                float kval = kh[k_idx];
                float* s_row = s_head + k_idx * dim_v;
                for (size_t v_idx = 0; v_idx < dim_v; ++v_idx) {
                    s_row[v_idx] = alpha_h * s_row[v_idx] + dynamic_eta * (kval * err_h[v_idx]);
                }
            }
        }

        // Manifold update
        float dynamic_mu = 0.05f * (1.0f + 0.5f * error_norm);
        for (size_t d = 0; d < dim_emb; ++d) e_target[d] += dynamic_mu * h_field[d];
        normalize(e_target, dim_emb);

        return {loss, is_correct, actual_steps};
    }
};

int main() {
    std::cout << "=== EXP-471: Sovereign Field Settling & Endogenous Plasticity Investigation ===" << std::endl;

    // Load structured real data (C++ Core)
    std::ifstream f("karyon_native_core.cpp", std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    f.close();

    std::cout << "Evaluating single-pass stream on " << data.size() << " bytes..." << std::endl;

    SovereignFieldEngine engine_baseline;
    SovereignFieldEngine engine_relaxed;

    size_t n = data.size() - 1;
    size_t q_size = n / 4;

    // 1. Evaluate Baseline (Rigid Single-Pass Feedforward)
    float base_q1_loss = 0.0f, base_q4_loss = 0.0f;
    size_t base_correct = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i) {
        auto [l, c] = engine_baseline.step_single_pass(data[i], data[i+1]);
        if (i < q_size) base_q1_loss += l;
        if (i >= n - q_size) base_q4_loss += l;
        if (c) base_correct++;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double base_time = std::chrono::duration<double>(t1 - t0).count();
    base_q1_loss /= q_size;
    base_q4_loss /= (n - (n - q_size));

    // 2. Evaluate Sovereign Field Settling (k in 1..5 dynamic loops)
    float rel_q1_loss = 0.0f, rel_q4_loss = 0.0f;
    size_t rel_correct = 0;
    size_t total_relaxation_steps = 0;
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i) {
        auto [l, c, steps] = engine_relaxed.step_sovereign_relaxed(data[i], data[i+1], 5);
        if (i < q_size) rel_q1_loss += l;
        if (i >= n - q_size) rel_q4_loss += l;
        if (c) rel_correct++;
        total_relaxation_steps += steps;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double rel_time = std::chrono::duration<double>(t1 - t0).count();
    rel_q1_loss /= q_size;
    rel_q4_loss /= (n - (n - q_size));

    std::cout << "\n=== COMPARATIVE TELEMETRY (EXP-471) ===\n";
    std::cout << std::left << std::setw(32) << "Configuration"
              << std::setw(14) << "Q1 Loss"
              << std::setw(14) << "Q4 Loss"
              << std::setw(14) << "Loss Delta"
              << std::setw(14) << "Accuracy"
              << std::setw(18) << "Avg Relax Steps"
              << std::setw(14) << "Steps/Sec" << "\n";
    std::cout << std::string(120, '-') << "\n";

    printf("%-32s | %10.4f | %10.4f | %10.4f | %9.2f%% | %14.1f | %10.1f\n",
           "Rigid Single-Pass (k=1)", base_q1_loss, base_q4_loss, base_q4_loss - base_q1_loss,
           (100.0f * base_correct) / n, 1.0f, n / base_time);

    printf("%-32s | %10.4f | %10.4f | %10.4f | %9.2f%% | %14.2f | %10.1f\n",
           "Sovereign Relaxed Settling", rel_q1_loss, rel_q4_loss, rel_q4_loss - rel_q1_loss,
           (100.0f * rel_correct) / n, (float)total_relaxation_steps / n, n / rel_time);

    std::cout << "\nFinal Endogenous Resonance Temperature beta: " << engine_relaxed.internal_temp << "\n";

    return 0;
}
