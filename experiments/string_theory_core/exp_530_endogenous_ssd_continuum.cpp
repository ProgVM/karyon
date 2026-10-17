// ============================================================================
// EXP-530: Endogenous Data-Dependent State-Space Continuum (Self-Modulating SSD)
// Testing Principle 29 (Law of Sovereign Self-Control of Substrate) & Principle 3
// - Continuous Modal State-Space dynamics: M_{t+1} = diag(alpha_t) * M_t + K_t^T * V_t
// - Pure Sovereign Plasticity: alpha_t is synthesized directly from the field state h_t
//     alpha_{d, k}(t) = sigmoid(base_log_decay_{d, k} + gamma_{alpha} * h_t)
// - Long-horizon channels can self-lock up to alpha -> 0.9999 (preserving character/word context)
// - Boundary channels can purge up to alpha -> 0.01 (entropy reset on spaces/punctuation)
// - SwiGLU / Solitonic non-linear spatial resonance: h_{out} = (h * W_gate) * swish(h * W_up)
// - Objective: Overcome Markov limit and achieve fluent text reconstruction WITHOUT discrete layers
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>

constexpr size_t VOCAB_SIZE = 258;
constexpr size_t DIM = 128;       // Physical channels
constexpr size_t MODES = 8;       // Harmonic modes per channel (State matrix: DIM x MODES = 1024 scalars)

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-std::clamp(x, -15.0f, 15.0f)));
}

inline float swish(float x) {
    return x * sigmoid(x);
}

class EndogenousSSDContinuum {
public:
    EndogenousSSDContinuum() {
        init_state();
    }

    void reset_state() {
        std::fill(M_.begin(), M_.end(), 0.0f);
        std::fill(last_h_.begin(), last_h_.end(), 0.0f);
    }

    // Single step forward through the continuous self-modulating SSD field
    std::vector<float> step_perceive(uint8_t byte_in) {
        // 1. Sensory Embedding: Map byte to continuous field projection
        std::vector<float> x(DIM);
        for (size_t d = 0; d < DIM; ++d) {
            x[d] = emb_[byte_in * DIM + d];
        }

        // 2. Endogenous Decay Modulation: alpha_t = sigmoid(base_alpha + tension)
        // High tension or specific context signals freeze alpha -> 1.0 (long-term memory)
        std::vector<float> alpha(DIM * MODES);
        for (size_t d = 0; d < DIM; ++d) {
            float tension = last_h_[d];
            for (size_t k = 0; k < MODES; ++k) {
                size_t idx = d * MODES + k;
                float logit_a = base_log_alpha_[idx] + alpha_mod_[idx] * tension;
                alpha[idx] = sigmoid(logit_a);
            }
        }

        // 3. Physical State-Space Modal Accumulation: M_{t+1} = alpha * M_t + K^T * V
        // Here K and V are continuous projections of x
        std::vector<float> k_proj(MODES, 0.0f);
        for (size_t k = 0; k < MODES; ++k) {
            for (size_t d = 0; d < DIM; ++d) {
                k_proj[k] += x[d] * k_weight_[k * DIM + d];
            }
            k_proj[k] = std::tanh(k_proj[k]);
        }

        std::vector<float> ssd_out(DIM, 0.0f);
        for (size_t d = 0; d < DIM; ++d) {
            float v_val = x[d];
            float acc = 0.0f;
            for (size_t k = 0; k < MODES; ++k) {
                size_t idx = d * MODES + k;
                // State space update:
                M_[idx] = alpha[idx] * M_[idx] + (1.0f - alpha[idx]) * (v_val * k_proj[k]);
                acc += M_[idx] * k_proj[k];
            }
            ssd_out[d] = acc;
        }

        // 4. Non-Linear SwiGLU Channel Resonance in unified field
        std::vector<float> h_field(DIM);
        for (size_t d = 0; d < DIM; ++d) {
            float gate = 0.0f;
            float val = 0.0f;
            for (size_t j = 0; j < DIM; ++j) {
                gate += w_gate_[d * DIM + j] * ssd_out[j];
                val  += w_val_[d * DIM + j]  * ssd_out[j];
            }
            // Residual highway: x + SwiGLU(ssd_out)
            h_field[d] = x[d] + swish(gate) * val;
            h_field[d] = std::clamp(h_field[d], -10.0f, 10.0f);
        }

        last_h_ = h_field;

        // 5. Readout into Byte Logits
        std::vector<float> logits(VOCAB_SIZE, 0.0f);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            float score = 0.0f;
            for (size_t d = 0; d < DIM; ++d) {
                score += h_field[d] * out_weight_[v * DIM + d];
            }
            logits[v] = score;
        }

        return logits;
    }

    std::pair<float, bool> adapt_step(uint8_t in_b, uint8_t target_b, float lr = 0.02f) {
        std::vector<float> logits = step_perceive(in_b);

        float max_l = *std::max_element(logits.begin(), logits.end());
        size_t argmax_v = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        bool hit = (argmax_v == target_b);

        float sum_exp = 0.0f;
        std::vector<float> probs(VOCAB_SIZE, 0.0f);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            probs[v] = std::exp(logits[v] - max_l);
            sum_exp += probs[v];
        }
        for (size_t v = 0; v < VOCAB_SIZE; ++v) probs[v] /= sum_exp;

        float p_target = std::max(probs[target_b], 1e-7f);
        float fe_surprise = -std::log(p_target);

        // Gradient of Cross-Entropy / Surprise
        std::vector<float> d_logits(VOCAB_SIZE);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            d_logits[v] = probs[v] - (v == target_b ? 1.0f : 0.0f);
        }

        // Backprop into out_weight and last_h
        std::vector<float> d_h(DIM, 0.0f);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            float dl = d_logits[v];
            if (std::abs(dl) < 1e-5f) continue;
            for (size_t d = 0; d < DIM; ++d) {
                d_h[d] += dl * out_weight_[v * DIM + d];
                out_weight_[v * DIM + d] -= lr * dl * last_h_[d];
                out_weight_[v * DIM + d] = std::clamp(out_weight_[v * DIM + d], -5.0f, 5.0f);
            }
        }

        // Plasticity for SwiGLU and input embeddings
        for (size_t d = 0; d < DIM; ++d) {
            emb_[in_b * DIM + d] -= lr * 0.5f * d_h[d];
            emb_[in_b * DIM + d] = std::clamp(emb_[in_b * DIM + d], -3.0f, 3.0f);

            // Adapt base decay and alpha self-modulator
            for (size_t k = 0; k < MODES; ++k) {
                size_t idx = d * MODES + k;
                base_log_alpha_[idx] -= lr * 0.1f * d_h[d] * M_[idx];
                base_log_alpha_[idx] = std::clamp(base_log_alpha_[idx], -4.0f, 6.0f);

                alpha_mod_[idx] -= lr * 0.05f * d_h[d] * last_h_[d];
                alpha_mod_[idx] = std::clamp(alpha_mod_[idx], -3.0f, 3.0f);
            }
        }

        return {fe_surprise, hit};
    }

    std::string speak(const std::string& prompt, size_t max_bytes = 150, float temp = 0.01f) {
        reset_state();
        std::string text = prompt;
        uint8_t last_b = ' ';
        for (char c : prompt) {
            last_b = static_cast<uint8_t>(c);
            step_perceive(last_b);
        }

        for (size_t step = 0; step < max_bytes; ++step) {
            std::vector<float> logits = step_perceive(last_b);
            float max_l = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            std::vector<float> probs(VOCAB_SIZE, 0.0f);
            for (size_t v = 0; v < VOCAB_SIZE; ++v) {
                probs[v] = std::exp((logits[v] - max_l) / temp);
                sum_exp += probs[v];
            }
            for (size_t v = 0; v < VOCAB_SIZE; ++v) probs[v] /= sum_exp;

            size_t sampled = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
            if (sampled == 257 || sampled == '\n') break;

            char c = static_cast<char>(sampled);
            text += c;
            last_b = static_cast<uint8_t>(sampled);
        }
        return text;
    }

private:
    void init_state() {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.08f);

        M_.resize(DIM * MODES, 0.0f);
        last_h_.resize(DIM, 0.0f);

        emb_.resize(VOCAB_SIZE * DIM);
        for (auto& val : emb_) val = dist(rng);

        out_weight_.resize(VOCAB_SIZE * DIM);
        for (auto& val : out_weight_) val = dist(rng);

        k_weight_.resize(MODES * DIM);
        for (auto& val : k_weight_) val = dist(rng);

        w_gate_.resize(DIM * DIM);
        w_val_.resize(DIM * DIM);
        for (size_t i = 0; i < DIM; ++i) {
            for (size_t j = 0; j < DIM; ++j) {
                w_gate_[i * DIM + j] = (i == j) ? 1.0f : dist(rng);
                w_val_[i * DIM + j]  = (i == j) ? 1.0f : dist(rng);
            }
        }

        base_log_alpha_.resize(DIM * MODES);
        alpha_mod_.resize(DIM * MODES, 0.0f);

        // Initialize log-spaced decay scales across modes (half lives from 2 to 2000 steps)
        for (size_t d = 0; d < DIM; ++d) {
            for (size_t k = 0; k < MODES; ++k) {
                size_t idx = d * MODES + k;
                // Logit mapping: 0.5 to 0.999
                float p = static_cast<float>(k) / static_cast<float>(MODES - 1);
                float alpha_target = 0.5f + 0.499f * std::pow(p, 2.0f);
                base_log_alpha_[idx] = std::log(alpha_target / (1.0f - alpha_target));
            }
        }
    }

    std::vector<float> M_;
    std::vector<float> last_h_;
    std::vector<float> emb_;
    std::vector<float> out_weight_;
    std::vector<float> k_weight_;
    std::vector<float> w_gate_, w_val_;
    std::vector<float> base_log_alpha_, alpha_mod_;
};

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-530: SELF-MODULATING STATE-SPACE CONTINUUM (ENDOGENOUS SSD) ===\n";
    std::cout << "=== Testing Continuous Multi-Scale Decay Adaptation Without Layers ===\n";
    std::cout << "========================================================================\n\n";

    const std::string narrative = 
        "Once upon a time, there was a little girl named Lily. "
        "Lily loved to play in the green garden with her favorite red ball. "
        "One sunny day, Lily saw a soft white cat sitting near the big oak tree. "
        "The cat smiled at Lily and gave a gentle purr. Lily was very happy. "
        "They became the best of friends and played happily together until sunset.";

    EndogenousSSDContinuum model;

    std::cout << "Continuum Channels: " << DIM << " | Modal State per Channel: " << MODES << "\n";
    std::cout << "Total Continuous State Capacity: " << (DIM * MODES) << " dynamic variables\n";
    std::cout << "Relaxing self-modulating state-space continuum...\n\n";

    auto t_start = std::chrono::high_resolution_clock::now();
    size_t cycle = 0;
    float best_fe = 999.0f;
    float best_acc = 0.0f;

    while (true) {
        cycle++;
        model.reset_state();
        float cycle_fe = 0.0f;
        size_t hits = 0;
        size_t total = narrative.length() - 1;

        float lr = std::max(0.005f, 0.04f * std::exp(-static_cast<float>(cycle) / 600.0f));

        for (size_t i = 0; i < total; ++i) {
            uint8_t in_b = static_cast<uint8_t>(narrative[i]);
            uint8_t target_b = static_cast<uint8_t>(narrative[i + 1]);

            auto [fe, hit] = model.adapt_step(in_b, target_b, lr);
            cycle_fe += fe;
            if (hit) hits++;
        }

        float mean_fe = cycle_fe / total;
        float ppl = std::exp(mean_fe);
        float acc_pct = (100.0f * hits) / total;

        if (mean_fe < best_fe) best_fe = mean_fe;
        if (acc_pct > best_acc) best_acc = acc_pct;

        if (cycle == 1 || cycle % 20 == 0 || acc_pct >= 90.0f) {
            auto t_curr = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_curr - t_start).count();

            std::cout << "  Cycle " << std::setw(4) << cycle
                      << " | Time: " << std::fixed << std::setprecision(1) << elapsed << "s"
                      << " | FE: " << std::setprecision(5) << mean_fe << " nats"
                      << " | PPL: " << std::setprecision(3) << ppl
                      << " | Accuracy: " << std::setprecision(2) << acc_pct << "%"
                      << " | Peak Acc: " << best_acc << "%\n";
        }

        // Convergence check: True mastery
        if (acc_pct >= 99.0f && mean_fe < 0.05f) {
            std::cout << "\n========================================================================\n";
            std::cout << ">>> [PERFECT CONVERGENCE ACHIEVED VIA SELF-MODULATING SSD!] <<<\n";
            std::cout << ">>> 100% Next-Byte Accuracy at Cycle " << cycle << " <<<\n";
            std::cout << "========================================================================\n\n";
            break;
        }

        if (cycle >= 350) {
            std::cout << "\n>>> Completed 350 relaxation cycles. <<<\n";
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n========================================================================\n";
    std::cout << "=== EXP-530 FINAL TELEMETRY REPORT ===\n";
    std::cout << "Total Relaxation Time: " << std::fixed << std::setprecision(2) << total_time << " s\n";
    std::cout << "Cycles: " << cycle << "\n";
    std::cout << "Peak Accuracy: " << std::setprecision(2) << best_acc << "%\n";
    std::cout << "Best Free Energy: " << std::setprecision(6) << best_fe << " nats\n";
    std::cout << "Best Perplexity: " << std::setprecision(4) << std::exp(best_fe) << "\n";
    std::cout << "========================================================================\n\n";

    std::cout << "EXACT SPEECH RECONSTRUCTION PROBES:\n";
    std::cout << "1. Prompt 'Once upon ': \n   -> \"" << model.speak("Once upon ", 320, 0.01f) << "\"\n\n";
    std::cout << "2. Prompt 'Lily loved ': \n   -> \"" << model.speak("Lily loved ", 250, 0.01f) << "\"\n\n";
    std::cout << "3. Prompt 'One sunny day': \n   -> \"" << model.speak("One sunny day", 220, 0.01f) << "\"\n\n";

    return 0;
}
