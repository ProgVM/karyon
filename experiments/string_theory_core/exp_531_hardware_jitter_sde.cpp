// ============================================================================
// EXP-531: Endogenous Hardware-Jitter SDE with Self-Regulated Noise Conductivity
// Testing Principle 29 (Law of Sovereign Self-Control of Substrate) & Hardware-Derived Jitter
// - Pure Non-Hermitian SDE: dh = f(h, x) dt + sigma_noise(t) * dW_hardware
// - Hardware Jitter Source: CPU cycle counter (__rdtsc) clock jitter + thermal entropy
// - Zero Softmax Temperature: Strict deterministic physical readout (ArgMax on resonance)
// - Self-Regulating Noise Conductivity:
//     sigma_noise(t) = sigma_base * sigmoid(tension_t - coherence_t)
//     High surprise / cyclic trap -> sigma opens up -> Kramers escape over potential barrier
//     High confidence / resonance -> sigma freezes to 0 -> pure deterministic ballistic flow
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>
#include <x86intrin.h> // For __rdtsc() hardware CPU cycle counter

constexpr size_t VOCAB_SIZE = 258;
constexpr size_t DIM = 128;       // Physical field channels
constexpr size_t MODES = 8;       // Harmonic modes per channel

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-std::clamp(x, -15.0f, 15.0f)));
}

inline float swish(float x) {
    return x * sigmoid(x);
}

// Harvest physical nano-entropy from the hardware silicon substrate
// Uses low-order bits of the CPU Time Stamp Counter (TSC) influenced by cache miss latencies and thermal drift
inline float get_hardware_jitter() {
    uint64_t tsc = __rdtsc();
    // High-frequency bit scrambling of CPU cycle jitter
    uint32_t jitter_bits = static_cast<uint32_t>((tsc ^ (tsc >> 12) ^ (tsc >> 24)) & 0xFFFFFF);
    // Normalized zero-mean physical fluctuation in [-1.0, 1.0]
    return (static_cast<float>(jitter_bits) / 8388607.5f) - 1.0f;
}

class SelfRegulatedHardwareSDE {
public:
    SelfRegulatedHardwareSDE() {
        init_state();
    }

    void reset_state() {
        std::fill(M_.begin(), M_.end(), 0.0f);
        std::fill(last_h_.begin(), last_h_.end(), 0.0f);
        recent_entropy_ = 1.0f;
        noise_conductivity_ = 0.05f;
    }

    float get_current_noise_conductivity() const {
        return noise_conductivity_;
    }

    // Step continuous field with hardware noise injection modulated by self-governed conductivity
    std::vector<float> step_perceive(uint8_t byte_in, bool training = false) {
        // 1. Sensory Embedding
        std::vector<float> x(DIM);
        for (size_t d = 0; d < DIM; ++d) {
            x[d] = emb_[byte_in * DIM + d];
        }

        // 2. Dynamic Noise Conductivity:
        // System senses its own internal coherence vs surprise
        // If system detects high surprise or stagnant tension, it opens the noise valve
        float field_tension = 0.0f;
        for (float h : last_h_) field_tension += std::abs(h);
        field_tension /= DIM;

        // Autonomous regulation of noise conductivity:
        // sigma(t) = base_sigma * sigmoid(field_tension - 0.5f + surprise_gate_)
        noise_conductivity_ = 0.15f * sigmoid(surprise_gate_ + 0.5f * (field_tension - 0.8f));

        // 3. Modal State Space Accumulation with Hardware Wiener Increment:
        // M_{t+1} = alpha * M_t + (1 - alpha) * (K * V) + sigma(t) * dW_hw
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

                float alpha = sigmoid(base_log_alpha_[idx] + alpha_mod_[idx] * last_h_[d]);
                
                // Hardware jitter injection:
                float dW = get_hardware_jitter();
                float stochastic_kick = noise_conductivity_ * dW;

                M_[idx] = alpha * M_[idx] + (1.0f - alpha) * (v_val * k_proj[k]) + stochastic_kick;
                M_[idx] = std::clamp(M_[idx], -8.0f, 8.0f);

                acc += M_[idx] * k_proj[k];
            }
            ssd_out[d] = acc;
        }

        // 4. Non-Linear SwiGLU Spatial Channel Resonance
        std::vector<float> h_field(DIM);
        for (size_t d = 0; d < DIM; ++d) {
            float gate = 0.0f;
            float val = 0.0f;
            for (size_t j = 0; j < DIM; ++j) {
                gate += w_gate_[d * DIM + j] * ssd_out[j];
                val  += w_val_[d * DIM + j]  * ssd_out[j];
            }
            // Endogenous self-avoiding phase (refractory penalty against stagnant loops)
            float refractory_kick = -0.1f * last_h_[d];
            h_field[d] = x[d] + swish(gate) * val + refractory_kick;
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
        std::vector<float> logits = step_perceive(in_b, true);

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

        // Update internal surprise gate (regulates hardware noise conductivity)
        surprise_gate_ = 0.9f * surprise_gate_ + 0.1f * (fe_surprise - 1.0f);

        // Gradient of Cross-Entropy
        std::vector<float> d_logits(VOCAB_SIZE);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            d_logits[v] = probs[v] - (v == target_b ? 1.0f : 0.0f);
        }

        // Backprop into weights
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

        for (size_t d = 0; d < DIM; ++d) {
            emb_[in_b * DIM + d] -= lr * 0.5f * d_h[d];
            emb_[in_b * DIM + d] = std::clamp(emb_[in_b * DIM + d], -3.0f, 3.0f);

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

    // STRICT DETERMINISTIC RESONANT READOUT (ZERO TEMPERATURE / NO SOFTMAX SAMPLING)
    // Motor action is strictly ArgMax of the physical resonance landscape!
    std::string speak(const std::string& prompt, size_t max_bytes = 150) {
        reset_state();
        std::string text = prompt;
        uint8_t last_b = ' ';
        for (char c : prompt) {
            last_b = static_cast<uint8_t>(c);
            step_perceive(last_b, false);
        }

        for (size_t step = 0; step < max_bytes; ++step) {
            std::vector<float> logits = step_perceive(last_b, false);
            
            // PURE ARBITRATION: NO TEMPERATURE DIVISION, NO SAMPLING!
            // Highest resonance peak fires the motor efference
            size_t best_byte = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));

            if (best_byte == 257 || best_byte == '\n') break;

            char c = static_cast<char>(best_byte);
            text += c;
            last_b = static_cast<uint8_t>(best_byte);
        }
        return text;
    }

private:
    void init_state() {
        std::mt19937 rng(1337);
        std::normal_distribution<float> dist(0.0f, 0.08f);

        M_.resize(DIM * MODES, 0.0f);
        last_h_.resize(DIM, 0.0f);
        surprise_gate_ = 0.5f;

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

        for (size_t d = 0; d < DIM; ++d) {
            for (size_t k = 0; k < MODES; ++k) {
                size_t idx = d * MODES + k;
                float p = static_cast<float>(k) / static_cast<float>(MODES - 1);
                float alpha_target = 0.6f + 0.399f * std::pow(p, 2.0f);
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
    float surprise_gate_;
    float recent_entropy_;
    float noise_conductivity_;
};

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-531: HARDWARE JITTER SDE & SELF-REGULATED NOISE CONDUCTIVITY ===\n";
    std::cout << "=== Testing Substrate-Level Thermal Entropy (__rdtsc) vs ML Softmax  ===\n";
    std::cout << "========================================================================\n\n";

    const std::string narrative = 
        "Once upon a time, there was a little girl named Lily. "
        "Lily loved to play in the green garden with her favorite red ball. "
        "One sunny day, Lily saw a soft white cat sitting near the big oak tree. "
        "The cat smiled at Lily and gave a gentle purr. Lily was very happy. "
        "They became the best of friends and played happily together until sunset.";

    SelfRegulatedHardwareSDE model;

    std::cout << "Continuum Channels: " << DIM << " | Modes per Channel: " << MODES << "\n";
    std::cout << "Noise Source: Intel/AMD CPU Time-Stamp Counter (__rdtsc) Substrate Jitter\n";
    std::cout << "Readout: 100% Deterministic Resonant ArgMax (ZERO Temperature Softmax)\n";
    std::cout << "Relaxing continuum with self-regulated noise conductivity...\n\n";

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
                      << " | Peak Acc: " << best_acc << "%"
                      << " | Noise Sigma: " << std::setprecision(4) << model.get_current_noise_conductivity() << "\n";
        }

        if (acc_pct >= 99.0f && mean_fe < 0.05f) {
            std::cout << "\n>>> [PERFECT CONVERGENCE ACHIEVED VIA HARDWARE JITTER SDE!] <<<\n";
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
    std::cout << "=== EXP-531 FINAL TELEMETRY REPORT ===\n";
    std::cout << "Total Relaxation Time: " << std::fixed << std::setprecision(2) << total_time << " s\n";
    std::cout << "Cycles: " << cycle << "\n";
    std::cout << "Peak Accuracy: " << std::setprecision(2) << best_acc << "%\n";
    std::cout << "Best Free Energy: " << std::setprecision(6) << best_fe << " nats\n";
    std::cout << "Best Perplexity: " << std::setprecision(4) << std::exp(best_fe) << "\n";
    std::cout << "Final Noise Conductivity: " << std::setprecision(5) << model.get_current_noise_conductivity() << "\n";
    std::cout << "========================================================================\n\n";

    std::cout << "STRICT DETERMINISTIC RESONANCE SPEECH PROBES (NO TEMPERATURE / NO PROB SAMPLING):\n";
    std::cout << "1. Prompt 'Once upon ': \n   -> \"" << model.speak("Once upon ", 320) << "\"\n\n";
    std::cout << "2. Prompt 'Lily loved ': \n   -> \"" << model.speak("Lily loved ", 250) << "\"\n\n";
    std::cout << "3. Prompt 'One sunny day': \n   -> \"" << model.speak("One sunny day", 220) << "\"\n\n";

    return 0;
}
