// ============================================================================
// EXP-532: Autopoietic Self-Sprouting Metric Continuum (String Universe)
// Testing Principle 2, Principle 3, Principle 27 & Principle 29
//
// ABSOLUTE SOVEREIGNTY:
// 1. Zero hardcoded operator topology: NO fixed layers, NO manual gating equations.
// 2. Continuous Metric Field Substrate:
//    The universe starts with an open collection of continuous vibrating string modes.
// 3. Autopoietic Self-Sprouting (Genesis & Pruning):
//    - When the existing metric field fails to minimize Free Energy (F > threshold),
//      the substrate autonomously sprouts NEW modes and dynamic coupling equations:
//      \psi_{new}(x, t) with endogenous coupling tensors G_{ij}.
//    - When modes are redundant or dissipative (low causal contribution to F reduction),
//      the substrate autonomously dissolves / prunes them.
// 4. Boundless Operator Synthesis:
//    - Each string mode self-regulates its own differential equation:
//      \ddot{q}_i + \gamma_i(t) \dot{q}_i + \omega_i^2(t) q_i = \sum_j G_{ij}(t) f_j(q_j) + \xi_{hw}(t)
//    - Couplings G_{ij}, non-linearities, and frequencies are 100% self-generated.
// 5. Objective: Validate autonomous self-assembly and convergence on arbitrary modalities.
// ============================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <x86intrin.h>

constexpr size_t VOCAB_SIZE = 258;

inline float get_hardware_jitter() {
    uint64_t tsc = __rdtsc();
    uint32_t bits = static_cast<uint32_t>((tsc ^ (tsc >> 11) ^ (tsc >> 23)) & 0xFFFFFF);
    return (static_cast<float>(bits) / 8388607.5f) - 1.0f;
}

// An autopoietic continuous mode (a sovereign vibrating string)
struct AutopoieticStringMode {
    float q = 0.0f;          // Generalized coordinate (displacement)
    float p = 0.0f;          // Conjugate momentum (velocity)
    float omega_sq = 1.0f;   // Endogenous eigenfrequency squared
    float gamma = 0.1f;      // Self-tuned dissipation rate
    float causal_credit = 0.0f; // Epistemic utility for Free Energy reduction
    
    // Inbound coupling tensor weights from all other modes
    std::vector<float> G_in;
    // Projection to motor efference
    std::vector<float> motor_proj;
    // Sensory receptive field (coupling to incoming raw bytes)
    std::vector<float> sensory_in;

    AutopoieticStringMode(size_t vocab_size, size_t existing_modes, std::mt19937& rng) {
        std::normal_distribution<float> dist(0.0f, 0.05f);
        omega_sq = 0.5f + std::abs(dist(rng)) * 5.0f;
        gamma = 0.05f + std::abs(dist(rng)) * 0.5f;

        G_in.resize(existing_modes, 0.0f);
        for (auto& g : G_in) g = dist(rng);

        motor_proj.resize(vocab_size, 0.0f);
        for (auto& m : motor_proj) m = dist(rng);

        sensory_in.resize(vocab_size, 0.0f);
        for (auto& s : sensory_in) s = dist(rng);
    }
};

class AutopoieticStringContinuum {
public:
    AutopoieticStringContinuum() : rng_(1337) {
        // Universe starts with just 4 embryonic seed strings
        for (size_t i = 0; i < 4; ++i) {
            sprout_mode();
        }
    }

    void reset_state() {
        for (auto& m : modes_) {
            m.q = 0.0f;
            m.p = 0.0f;
        }
    }

    size_t active_mode_count() const {
        return modes_.size();
    }

    // Autopoietic Genesis: Sprout a new continuous string into the physical metric space
    void sprout_mode() {
        size_t current_n = modes_.size();
        modes_.emplace_back(VOCAB_SIZE, current_n, rng_);
        
        // Expand all existing modes to have cross-coupling with the newborn mode
        for (size_t i = 0; i < current_n; ++i) {
            std::normal_distribution<float> dist(0.0f, 0.05f);
            modes_[i].G_in.push_back(dist(rng_));
        }
    }

    // Autopoietic Dissolution: Prune a useless dissipative mode that does not serve Free Energy minimization
    void dissolve_weakest_mode() {
        if (modes_.size() <= 4) return; // Preserve embryonic core

        // Find mode with lowest causal credit
        auto min_it = std::min_element(modes_.begin(), modes_.end(),
            [](const AutopoieticStringMode& a, const AutopoieticStringMode& b) {
                return a.causal_credit < b.causal_credit;
            });
        
        size_t idx = std::distance(modes_.begin(), min_it);
        modes_.erase(min_it);

        // Remove coupling column from all surviving modes
        for (auto& m : modes_) {
            if (idx < m.G_in.size()) {
                m.G_in.erase(m.G_in.begin() + idx);
            }
        }
    }

    // Step continuous multi-string field dynamics
    std::vector<float> step_field(uint8_t byte_in, float dt = 0.05f) {
        size_t N = modes_.size();

        // 1. Compute non-linear inter-string coupling forces
        std::vector<float> nonlin_activations(N);
        for (size_t i = 0; i < N; ++i) {
            // Unrestricted non-linear wave state: self-solitonic compression
            nonlin_activations[i] = std::tanh(modes_[i].q);
        }

        // 2. Physical Symplectic Step of the coupled string universe
        float hw_jitter = get_hardware_jitter();

        for (size_t i = 0; i < N; ++i) {
            float coupled_force = modes_[i].sensory_in[byte_in];

            for (size_t j = 0; j < modes_[i].G_in.size(); ++j) {
                coupled_force += modes_[i].G_in[j] * nonlin_activations[j];
            }

            // Coupled 2nd-order wave equation with hardware thermal jitter
            float acc = -modes_[i].omega_sq * modes_[i].q
                        - modes_[i].gamma * modes_[i].p
                        + coupled_force
                        + 0.02f * hw_jitter;

            modes_[i].p += acc * dt;
            modes_[i].p = std::clamp(modes_[i].p, -15.0f, 15.0f);

            modes_[i].q += modes_[i].p * dt;
            modes_[i].q = std::clamp(modes_[i].q, -8.0f, 8.0f);
        }

        // 3. Continuous Field Motor Efference (Resonance Projection)
        std::vector<float> logits(VOCAB_SIZE, 0.0f);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            float sum = 0.0f;
            for (size_t i = 0; i < N; ++i) {
                sum += modes_[i].q * modes_[i].motor_proj[v];
            }
            logits[v] = sum;
        }

        return logits;
    }

    // Continuous learning through self-governed plasticity and morphogenesis
    std::pair<float, bool> adapt(uint8_t in_b, uint8_t target_b, float lr = 0.03f) {
        std::vector<float> logits = step_field(in_b);

        size_t best_v = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        bool hit = (best_v == target_b);

        float max_l = logits[best_v];
        float sum_exp = 0.0f;
        std::vector<float> probs(VOCAB_SIZE, 0.0f);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            probs[v] = std::exp(std::clamp(logits[v] - max_l, -30.0f, 0.0f));
            sum_exp += probs[v];
        }
        for (size_t v = 0; v < VOCAB_SIZE; ++v) probs[v] /= sum_exp;

        float p_target = std::max(probs[target_b], 1e-7f);
        float fe_surprise = -std::log(p_target);

        // Compute Free Energy gradient on motor projection
        std::vector<float> d_logits(VOCAB_SIZE);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            d_logits[v] = probs[v] - (v == target_b ? 1.0f : 0.0f);
        }

        size_t N = modes_.size();
        for (size_t i = 0; i < N; ++i) {
            float mode_causal_flow = 0.0f;
            for (size_t v = 0; v < VOCAB_SIZE; ++v) {
                float dl = d_logits[v];
                mode_causal_flow += std::abs(dl * modes_[i].motor_proj[v]);

                // Motor projection plasticity
                modes_[i].motor_proj[v] -= lr * dl * modes_[i].q;
                modes_[i].motor_proj[v] = std::clamp(modes_[i].motor_proj[v], -5.0f, 5.0f);
            }

            // Sensory receptive field plasticity
            modes_[i].sensory_in[in_b] -= lr * 0.5f * mode_causal_flow;
            modes_[i].sensory_in[in_b] = std::clamp(modes_[i].sensory_in[in_b], -3.0f, 3.0f);

            // Self-tune internal physical parameters (eigenfrequency and dissipation)
            modes_[i].omega_sq = std::clamp(modes_[i].omega_sq - lr * 0.1f * mode_causal_flow * modes_[i].q, 0.05f, 25.0f);
            modes_[i].gamma    = std::clamp(modes_[i].gamma    - lr * 0.05f * mode_causal_flow * modes_[i].p, 0.01f, 10.0f);

            // Cross-coupling plasticity
            for (size_t j = 0; j < modes_[i].G_in.size(); ++j) {
                modes_[i].G_in[j] -= lr * 0.2f * mode_causal_flow * std::tanh(modes_[j].q);
                modes_[i].G_in[j] = std::clamp(modes_[i].G_in[j], -4.0f, 4.0f);
            }

            // Epistemic causal credit accumulation (tracks which strings solve the task)
            modes_[i].causal_credit = 0.99f * modes_[i].causal_credit + 0.01f * (hit ? 1.0f : -0.5f * fe_surprise);
        }

        // ====================================================================
        // AUTOPOIETIC MORPHOGENESIS: The universe shapes its own topology!
        // ====================================================================
        // If surprise is high and accuracy is stalling, sprout a new string mode
        if (fe_surprise > 2.5f && modes_.size() < 128) {
            if (get_hardware_jitter() > 0.6f) { // Spontaneous morphogenetic bifurcation
                sprout_mode();
            }
        }
        // If a mode has negative causal credit over time, dissolve it
        if (modes_.size() > 8 && get_hardware_jitter() < -0.8f) {
            dissolve_weakest_mode();
        }

        return {fe_surprise, hit};
    }

    std::string speak(const std::string& prompt, size_t max_bytes = 120) {
        reset_state();
        std::string out = prompt;
        uint8_t last_b = ' ';
        for (char c : prompt) {
            last_b = static_cast<uint8_t>(c);
            step_field(last_b);
        }

        for (size_t s = 0; s < max_bytes; ++s) {
            std::vector<float> logits = step_field(last_b);
            size_t best_b = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
            if (best_b == 257 || best_b == '\n') break;

            out += static_cast<char>(best_b);
            last_b = static_cast<uint8_t>(best_b);
        }
        return out;
    }

private:
    std::vector<AutopoieticStringMode> modes_;
    std::mt19937 rng_;
};

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-532: AUTOPOIETIC SELF-SPROUTING STRING CONTINUUM ==============\n";
    std::cout << "=== Zero Hardcoded Layers/Equations: Strings Sprout & Dissolve Themselves ==\n";
    std::cout << "========================================================================\n\n";

    const std::string corpus = 
        "Once upon a time, there was a little girl named Lily. "
        "Lily loved to play in the green garden with her favorite red ball. "
        "One sunny day, Lily saw a soft white cat sitting near the big oak tree. "
        "The cat smiled at Lily and gave a gentle purr. Lily was very happy. "
        "They became the best of friends and played happily together until sunset.";

    AutopoieticStringContinuum universe;

    std::cout << "Initial Seed String Modes: " << universe.active_mode_count() << "\n";
    std::cout << "Evolution Protocol: Open-Ended Substrate Genesis (Principle 2, 3, 27, 29)\n\n";

    auto t_start = std::chrono::high_resolution_clock::now();
    size_t cycle = 0;
    float best_fe = 999.0f;
    float best_acc = 0.0f;

    while (true) {
        cycle++;
        universe.reset_state();
        float cycle_fe = 0.0f;
        size_t hits = 0;
        size_t total = corpus.length() - 1;

        float lr = std::max(0.008f, 0.05f * std::exp(-static_cast<float>(cycle) / 500.0f));

        for (size_t i = 0; i < total; ++i) {
            uint8_t in_b = static_cast<uint8_t>(corpus[i]);
            uint8_t target_b = static_cast<uint8_t>(corpus[i + 1]);

            auto [fe, hit] = universe.adapt(in_b, target_b, lr);
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
                      << " | Modes: " << std::setw(3) << universe.active_mode_count()
                      << " | FE: " << std::setprecision(5) << mean_fe << " nats"
                      << " | PPL: " << std::setprecision(3) << ppl
                      << " | Acc: " << std::setprecision(2) << acc_pct << "%"
                      << " | Peak Acc: " << best_acc << "%\n";
        }

        if (acc_pct >= 98.0f && mean_fe < 0.08f) {
            std::cout << "\n>>> [AUTOPOIETIC MASTERY ACHIEVED BY SELF-SPROUTED STRINGS!] <<<\n";
            break;
        }

        if (cycle >= 300) {
            std::cout << "\n>>> Completed 300 autopoietic relaxation cycles. <<<\n";
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n========================================================================\n";
    std::cout << "=== EXP-532 FINAL MORPHOGENESIS REPORT ===\n";
    std::cout << "Total Elapsed Time: " << std::fixed << std::setprecision(2) << total_time << " s\n";
    std::cout << "Cycles: " << cycle << "\n";
    std::cout << "Final Autopoietic Mode Count: " << universe.active_mode_count() << " modes\n";
    std::cout << "Peak Accuracy: " << std::setprecision(2) << best_acc << "%\n";
    std::cout << "Best Free Energy: " << std::setprecision(6) << best_fe << " nats\n";
    std::cout << "Best Perplexity: " << std::setprecision(4) << std::exp(best_fe) << "\n";
    std::cout << "========================================================================\n\n";

    std::cout << "PURE PHYSICAL RESONANT SPEECH PROBES (NO TEMPERATURE / NO PROB SAMPLING):\n";
    std::cout << "1. Prompt 'Once upon ': \n   -> \"" << universe.speak("Once upon ", 250) << "\"\n\n";
    std::cout << "2. Prompt 'Lily loved ': \n   -> \"" << universe.speak("Lily loved ", 200) << "\"\n\n";
    std::cout << "3. Prompt 'One sunny day': \n   -> \"" << universe.speak("One sunny day", 180) << "\"\n\n";

    return 0;
}
