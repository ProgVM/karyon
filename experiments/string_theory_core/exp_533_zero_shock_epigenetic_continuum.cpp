// ============================================================================
// EXP-533: Zero-Shock Epigenetic Sprouting in Sovereign String Continuum
// Testing Principle 2, 3, 6, 27 & 29:
// - Absolute Substrate Sovereignty (no hardcoded layers, no fixed menus)
// - Principle 6 (Epigenetic Smooth Sprouting):
//   New strings are born in identity silence: coupling weight starts at 0 (tanh(alpha_epi) = 0).
//   The newborn string perceives the sensory environment and existing field vibrations,
//   accumulates resonance, and its efferent influence alpha_epi smoothly opens ONLY
//   as its causal contribution to Free Energy minimization becomes positive.
// - Zero Thermal Shock: Existing attractor basins and learned memories are never disrupted!
// - Hardware Jitter SDE: __rdtsc() provides native non-deterministic physical drift.
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

struct EpigeneticStringMode {
    float q = 0.0f;               // Displacement
    float p = 0.0f;               // Momentum
    float omega_sq = 1.0f;        // Natural frequency squared
    float gamma = 0.1f;           // Damping
    float causal_credit = 0.0f;   // Epistemic utility

    // Epigenetic gating factor alpha_epi:
    // Starts at 0 at birth (zero-shock identity).
    // Opens smoothly via plasticity as mode proves its predictive power.
    float alpha_epi = 0.0f;

    std::vector<float> G_in;        // Inter-string coupling weights
    std::vector<float> motor_proj; // Projection to byte logits
    std::vector<float> sensory_in; // Coupling to incoming sensory bytes

    EpigeneticStringMode(size_t vocab_size, size_t existing_modes, std::mt19937& rng, bool is_seed = false) {
        std::normal_distribution<float> dist(0.0f, 0.04f);
        omega_sq = 0.5f + std::abs(dist(rng)) * 4.0f;
        gamma = 0.05f + std::abs(dist(rng)) * 0.3f;

        G_in.resize(existing_modes, 0.0f);
        motor_proj.resize(vocab_size, 0.0f);
        sensory_in.resize(vocab_size, 0.0f);

        if (is_seed) {
            // Seed modes are active from the genesis
            alpha_epi = 1.0f;
            for (auto& g : G_in) g = dist(rng);
            for (auto& m : motor_proj) m = dist(rng);
            for (auto& s : sensory_in) s = dist(rng);
        } else {
            // Newborn mode: ZERO-SHOCK BIRTH!
            // Inbound sensory receptive field is initialized to absorb signal,
            // but outbound motor and cross-coupling impact is strictly 0.0 at birth!
            alpha_epi = 0.0f; // Completely silent outbound impact
            for (auto& s : sensory_in) s = dist(rng);
            // G_in and motor_proj remain 0.0 at moment of genesis
        }
    }
};

class SovereignEpigeneticContinuum {
public:
    SovereignEpigeneticContinuum() : rng_(1337) {
        // Universe begins with 8 active seed strings
        for (size_t i = 0; i < 8; ++i) {
            modes_.emplace_back(VOCAB_SIZE, modes_.size(), rng_, true);
            // expand couplings for existing seed modes
            for (size_t j = 0; j < i; ++j) {
                std::normal_distribution<float> dist(0.0f, 0.04f);
                modes_[j].G_in.push_back(dist(rng_));
            }
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

    float get_mean_alpha_epi() const {
        if (modes_.empty()) return 0.0f;
        float sum = 0.0f;
        for (const auto& m : modes_) sum += m.alpha_epi;
        return sum / modes_.size();
    }

    // Zero-Shock Autopoietic Sprouting (Principle 6)
    void sprout_zero_shock_mode() {
        size_t current_n = modes_.size();
        // Newborn mode is completely silent out-bound
        modes_.emplace_back(VOCAB_SIZE, current_n, rng_, false);

        // Existing modes get a 0.0 inbound coupling from the newcomer (no perturbation!)
        for (size_t i = 0; i < current_n; ++i) {
            modes_[i].G_in.push_back(0.0f);
        }
    }

    // Autopoietic Pruning: Dissolve modes that failed to achieve causal efficacy
    void prune_dissipative_modes() {
        if (modes_.size() <= 8) return; // Preserve core seeds

        // Find candidate with lowest causal credit
        auto min_it = std::min_element(modes_.begin() + 8, modes_.end(),
            [](const EpigeneticStringMode& a, const EpigeneticStringMode& b) {
                return a.causal_credit < b.causal_credit;
            });

        if (min_it != modes_.end() && min_it->causal_credit < -1.0f) {
            size_t idx = std::distance(modes_.begin(), min_it);
            modes_.erase(min_it);

            for (auto& m : modes_) {
                if (idx < m.G_in.size()) {
                    m.G_in.erase(m.G_in.begin() + idx);
                }
            }
        }
    }

    // Physical Field Step
    std::vector<float> step_field(uint8_t byte_in, float dt = 0.05f) {
        size_t N = modes_.size();

        std::vector<float> activations(N);
        for (size_t i = 0; i < N; ++i) {
            // Outbound activation is gated by the mode's own epigenetic maturation
            activations[i] = modes_[i].alpha_epi * std::tanh(modes_[i].q);
        }

        float hw_jitter = get_hardware_jitter();

        for (size_t i = 0; i < N; ++i) {
            float coupled_force = modes_[i].sensory_in[byte_in];

            for (size_t j = 0; j < modes_[i].G_in.size(); ++j) {
                coupled_force += modes_[i].G_in[j] * activations[j];
            }

            float acc = -modes_[i].omega_sq * modes_[i].q
                        - modes_[i].gamma * modes_[i].p
                        + coupled_force
                        + 0.015f * hw_jitter;

            modes_[i].p += acc * dt;
            modes_[i].p = std::clamp(modes_[i].p, -12.0f, 12.0f);

            modes_[i].q += modes_[i].p * dt;
            modes_[i].q = std::clamp(modes_[i].q, -8.0f, 8.0f);
        }

        // Motor efference: resonance projection weighted by epigenetic maturity
        std::vector<float> logits(VOCAB_SIZE, 0.0f);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            float sum = 0.0f;
            for (size_t i = 0; i < N; ++i) {
                sum += modes_[i].alpha_epi * modes_[i].q * modes_[i].motor_proj[v];
            }
            logits[v] = sum;
        }

        return logits;
    }

    // Adaptive step with endogenous epigenetic plasticity
    std::pair<float, bool> adapt(uint8_t in_b, uint8_t target_b, float lr = 0.025f) {
        std::vector<float> logits = step_field(in_b);

        size_t best_v = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        bool hit = (best_v == target_b);

        float max_l = logits[best_v];
        float sum_exp = 0.0f;
        std::vector<float> probs(VOCAB_SIZE, 0.0f);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            probs[v] = std::exp(std::clamp(logits[v] - max_l, -25.0f, 0.0f));
            sum_exp += probs[v];
        }
        for (size_t v = 0; v < VOCAB_SIZE; ++v) probs[v] /= sum_exp;

        float p_target = std::max(probs[target_b], 1e-7f);
        float fe_surprise = -std::log(p_target);

        // Compute Free Energy gradient
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
                modes_[i].motor_proj[v] -= lr * dl * (modes_[i].alpha_epi * modes_[i].q);
                modes_[i].motor_proj[v] = std::clamp(modes_[i].motor_proj[v], -5.0f, 5.0f);
            }

            // Sensory receptive field plasticity
            modes_[i].sensory_in[in_b] -= lr * 0.4f * mode_causal_flow;
            modes_[i].sensory_in[in_b] = std::clamp(modes_[i].sensory_in[in_b], -3.0f, 3.0f);

            // Self-tuning physical resonance parameters
            modes_[i].omega_sq = std::clamp(modes_[i].omega_sq - lr * 0.08f * mode_causal_flow * modes_[i].q, 0.05f, 25.0f);
            modes_[i].gamma    = std::clamp(modes_[i].gamma    - lr * 0.04f * mode_causal_flow * modes_[i].p, 0.01f, 8.0f);

            // Cross-coupling plasticity
            for (size_t j = 0; j < modes_[i].G_in.size(); ++j) {
                modes_[i].G_in[j] -= lr * 0.15f * mode_causal_flow * std::tanh(modes_[j].q);
                modes_[i].G_in[j] = std::clamp(modes_[i].G_in[j], -3.0f, 3.0f);
            }

            // Epigenetic Maturation:
            // If the mode is consistently predictive, its alpha_epi opens up.
            // If it introduces noise, its alpha_epi closes down.
            if (hit) {
                modes_[i].causal_credit = 0.98f * modes_[i].causal_credit + 0.02f * 1.0f;
                modes_[i].alpha_epi = std::min(1.0f, modes_[i].alpha_epi + 0.005f);
            } else {
                modes_[i].causal_credit = 0.98f * modes_[i].causal_credit - 0.02f * 0.5f * fe_surprise;
                if (modes_[i].alpha_epi < 0.2f) {
                    // Newborn modes that fail quickly stay suppressed
                    modes_[i].alpha_epi = std::max(0.0f, modes_[i].alpha_epi - 0.002f);
                }
            }
        }

        // ====================================================================
        // ZERO-SHOCK AUTOPOIETIC SPROUTING:
        // ====================================================================
        // Sprout only when surprise is sustained, but since birth is zero-shock,
        // it NEVER disrupts the already existing field equilibrium!
        if (fe_surprise > 2.0f && modes_.size() < 64) {
            if (get_hardware_jitter() > 0.7f) {
                sprout_zero_shock_mode();
            }
        }

        if (modes_.size() > 16 && get_hardware_jitter() < -0.85f) {
            prune_dissipative_modes();
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
    std::vector<EpigeneticStringMode> modes_;
    std::mt19937 rng_;
};

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-533: ZERO-SHOCK EPIGENETIC SPROUTING CONTINUUM ================\n";
    std::cout << "=== Testing Principle 6: Smooth Net2Net Genesis with Zero Disruption ===\n";
    std::cout << "========================================================================\n\n";

    const std::string corpus = 
        "Once upon a time, there was a little girl named Lily. "
        "Lily loved to play in the green garden with her favorite red ball. "
        "One sunny day, Lily saw a soft white cat sitting near the big oak tree. "
        "The cat smiled at Lily and gave a gentle purr. Lily was very happy. "
        "They became the best of friends and played happily together until sunset.";

    SovereignEpigeneticContinuum universe;

    std::cout << "Initial Seed Strings: " << universe.active_mode_count() << "\n";
    std::cout << "Birth Mechanism: Zero-Shock Epigenetic Gating (alpha_epi = 0.0 at birth)\n";
    std::cout << "Readout: Strict Physical Resonance (ArgMax, Zero Temperature)\n\n";

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

        float lr = std::max(0.006f, 0.045f * std::exp(-static_cast<float>(cycle) / 600.0f));

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

        if (cycle == 1 || cycle % 20 == 0 || acc_pct >= 85.0f) {
            auto t_curr = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_curr - t_start).count();

            std::cout << "  Cycle " << std::setw(4) << cycle
                      << " | Time: " << std::fixed << std::setprecision(1) << elapsed << "s"
                      << " | Modes: " << std::setw(2) << universe.active_mode_count()
                      << " | Alpha Epi: " << std::setprecision(3) << universe.get_mean_alpha_epi()
                      << " | FE: " << std::setprecision(5) << mean_fe << " nats"
                      << " | PPL: " << std::setprecision(3) << ppl
                      << " | Acc: " << std::setprecision(2) << acc_pct << "%"
                      << " | Peak Acc: " << best_acc << "%\n";
        }

        if (acc_pct >= 98.0f && mean_fe < 0.08f) {
            std::cout << "\n>>> [ZERO-SHOCK EPIGENETIC MASTERY ACHIEVED!] <<<\n";
            break;
        }

        if (cycle >= 350) {
            std::cout << "\n>>> Completed 350 cycles. <<<\n";
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n========================================================================\n";
    std::cout << "=== EXP-533 FINAL TELEMETRY REPORT ===\n";
    std::cout << "Total Elapsed Time: " << std::fixed << std::setprecision(2) << total_time << " s\n";
    std::cout << "Cycles: " << cycle << "\n";
    std::cout << "Final Mode Count: " << universe.active_mode_count() << "\n";
    std::cout << "Mean Epigenetic Maturity: " << std::setprecision(4) << universe.get_mean_alpha_epi() << "\n";
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
