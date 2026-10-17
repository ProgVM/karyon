// ============================================================================
// EXP-528: Self-Modulating Sovereign String Continuum (W = \emptyset)
// Testing Principle 29 (Law of Sovereign Self-Control of Internal Substrate)
// "Strings dynamically synthesize their own damping Gamma_k(t) and frequencies omega_k(t)"
// - Dynamic Damping Modulation: Gamma_k(t) -> 0 (solitonic memory locking) or Gamma_k(t) >> 1 (entropy purge)
// - Dynamic Stiffness/Frequency Tuning: omega_k^2(t) parametric resonance
// - Zero MLP / Zero external overseers: self-modulation is endogenous to field state
// - Target: Break through 1st-order Markov barrier and observe memory phase-locking
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
constexpr size_t MODES = 8;

class SelfModulatingSovereignContinuum {
public:
    SelfModulatingSovereignContinuum(size_t dim = 128) : dim_(dim) {
        init_state();
    }

    size_t get_dim() const { return dim_; }

    void get_byte_excitation(uint8_t byte_val, std::vector<float>& out_x) const {
        out_x.resize(dim_);
        float base_phase = (2.0f * M_PI * static_cast<float>(byte_val)) / static_cast<float>(VOCAB_SIZE);
        for (size_t d = 0; d < dim_; ++d) {
            float spatial_freq = static_cast<float>(d + 1);
            out_x[d] = std::sin(base_phase * spatial_freq) / std::sqrt(spatial_freq);
        }
    }

    std::vector<float> step_perceive(uint8_t byte_in, float dt = 0.02f) {
        std::vector<float> x;
        get_byte_excitation(byte_in, x);

        // 1. Dynamic Endogenous Self-Modulation of Damping Gamma and Frequency Omega
        // Computed purely from internal field tension without external MLPs
        std::vector<float> h_raw(dim_, 0.0f);
        
        for (size_t d = 0; d < dim_; ++d) {
            float sum_q = 0.0f;
            float local_tension = last_h_[d]; // feedback from previous field state

            for (size_t k = 0; k < MODES; ++k) {
                size_t idx = d * MODES + k;

                // Endogenous parametric frequency modulation:
                // omega_eff^2 = omega_base^2 * exp(gamma_omega * local_tension)
                float freq_mod = std::exp(std::clamp(omega_mod_coupling_[idx] * local_tension, -1.5f, 1.5f));
                float w_sq = base_stiffness_[idx] * freq_mod;

                // Endogenous damping modulation:
                // When field is resonant, damping can collapse to 0 (undamped standing wave memory)
                // When field detects surprise/entropy reset, damping increases for swift dissipation
                float damp_mod = 1.0f / (1.0f + std::exp(-(base_damping_[idx] + damp_mod_coupling_[idx] * local_tension)));
                float gamma_eff = damp_mod * 2.0f; // range [0.0, 2.0]

                float force = x[d] / k_floats_[k];
                float q_star = force / (w_sq + 1e-5f);
                float decay = std::exp(-gamma_eff * dt);

                q_[idx] = decay * (q_[idx] - q_star) + q_star;
                sum_q += k_floats_[k] * q_[idx];
            }
            h_raw[d] = sum_q;
        }

        // 2. Spatial Cross-Induction Tensor Field & Non-Linear Cubic Soliton
        std::vector<float> h_field(dim_, 0.0f);
        for (size_t i = 0; i < dim_; ++i) {
            float induction = 0.0f;
            size_t row_off = i * dim_;
            for (size_t j = 0; j < dim_; ++j) {
                induction += coupling_[row_off + j] * h_raw[j];
            }
            float h_total = h_raw[i] + induction;
            float soliton = h_total + 0.25f * (h_total * h_total * h_total);
            h_field[i] = std::tanh(soliton);
        }

        // 3. Weightless Dual Metric Readout Resonator
        std::vector<float> logits(VOCAB_SIZE, 0.0f);
        std::vector<float> dual_x(dim_);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            get_byte_excitation(static_cast<uint8_t>(v), dual_x);
            float overlap = 0.0f;
            for (size_t d = 0; d < dim_; ++d) {
                overlap += h_field[d] * dual_x[d] * readout_metric_[d];
            }
            logits[v] = overlap * 6.0f;
        }

        last_h_ = h_field;
        return logits;
    }

    std::pair<float, bool> adapt_step(uint8_t in_b, uint8_t target_b, float lr = 0.04f) {
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

        // Compute Resonant Error Field
        std::vector<float> error_field(dim_, 0.0f);
        std::vector<float> dual_x(dim_);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            float err_v = probs[v] - (v == target_b ? 1.0f : 0.0f);
            if (std::abs(err_v) < 1e-5f) continue;
            get_byte_excitation(static_cast<uint8_t>(v), dual_x);
            for (size_t d = 0; d < dim_; ++d) {
                error_field[d] += err_v * dual_x[d];
            }
        }

        // Sovereign Self-Tuning Plasticity: Metric, Coupling, and Self-Modulation Tensor
        for (size_t d = 0; d < dim_; ++d) {
            readout_metric_[d] -= lr * error_field[d] * last_h_[d];
            readout_metric_[d] = std::clamp(readout_metric_[d], -20.0f, 20.0f);

            size_t row_off = d * dim_;
            for (size_t j = 0; j < dim_; ++j) {
                coupling_[row_off + j] -= (lr * 0.5f) * error_field[d] * last_h_[j];
                coupling_[row_off + j] = std::clamp(coupling_[row_off + j], -10.0f, 10.0f);
            }

            for (size_t k = 0; k < MODES; ++k) {
                size_t idx = d * MODES + k;
                // Plasticity on frequency and damping self-modulators
                omega_mod_coupling_[idx] -= (lr * 0.2f) * error_field[d] * last_h_[d];
                omega_mod_coupling_[idx] = std::clamp(omega_mod_coupling_[idx], -5.0f, 5.0f);

                damp_mod_coupling_[idx] -= (lr * 0.2f) * error_field[d] * last_h_[d];
                damp_mod_coupling_[idx] = std::clamp(damp_mod_coupling_[idx], -5.0f, 5.0f);

                base_stiffness_[idx] -= (lr * 0.1f) * error_field[d] * q_[idx];
                base_stiffness_[idx] = std::clamp(base_stiffness_[idx], 0.01f, 50.0f);
            }
        }

        return {fe_surprise, hit};
    }

    void reset_wave_state() {
        std::fill(q_.begin(), q_.end(), 0.0f);
        std::fill(p_.begin(), p_.end(), 0.0f);
        std::fill(last_h_.begin(), last_h_.end(), 0.0f);
    }

    std::string speak(const std::string& prompt, size_t max_bytes = 150, float temp = 0.01f) {
        reset_wave_state();
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
        q_.resize(dim_ * MODES, 0.0f);
        p_.resize(dim_ * MODES, 0.0f);
        last_h_.resize(dim_, 0.0f);
        readout_metric_.resize(dim_, 1.0f);
        base_stiffness_.resize(dim_ * MODES);
        base_damping_.resize(dim_ * MODES, 0.0f);
        omega_mod_coupling_.resize(dim_ * MODES, 0.0f);
        damp_mod_coupling_.resize(dim_ * MODES, 0.0f);

        for (size_t d = 0; d < dim_; ++d) {
            for (size_t k = 0; k < MODES; ++k) {
                size_t idx = d * MODES + k;
                base_stiffness_[idx] = 1.0f + 0.5f * static_cast<float>(k);
            }
        }
        coupling_.resize(dim_ * dim_, 0.0f);
        for (size_t k = 0; k < MODES; ++k) k_floats_[k] = static_cast<float>(k + 1);
    }

    size_t dim_;
    std::vector<float> q_, p_;
    std::vector<float> last_h_;
    std::vector<float> readout_metric_;
    std::vector<float> base_stiffness_, base_damping_;
    std::vector<float> omega_mod_coupling_, damp_mod_coupling_;
    std::vector<float> coupling_;
    float k_floats_[MODES];
};

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-528: SELF-MODULATING STRINGS (ACTIVE NON-HERMITIAN FIELD) ===\n";
    std::cout << "=== Autonomous Frequency Omega(t) & Damping Gamma(t) Self-Control ===\n";
    std::cout << "========================================================================\n\n";

    const std::string narrative = 
        "Once upon a time, there was a little girl named Lily. "
        "Lily loved to play in the green garden with her favorite red ball. "
        "One sunny day, Lily saw a soft white cat sitting near the big oak tree. "
        "The cat smiled at Lily and gave a gentle purr. Lily was very happy. "
        "They became the best of friends and played happily together until sunset.";

    SelfModulatingSovereignContinuum model(128);

    std::cout << "Continuum Channels: " << model.get_dim() << " | Modes per Channel: " << MODES << "\n";
    std::cout << "Relaxing self-modulating field...\n\n";

    auto t_start = std::chrono::high_resolution_clock::now();
    size_t cycle = 0;
    float best_fe = 999.0f;
    float best_acc = 0.0f;

    while (true) {
        cycle++;
        model.reset_wave_state();
        float cycle_fe = 0.0f;
        size_t hits = 0;
        size_t total = narrative.length() - 1;

        float lr = std::max(0.005f, 0.05f * std::exp(-static_cast<float>(cycle) / 1000.0f));

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

        if (cycle == 1 || cycle % 25 == 0 || acc_pct >= 90.0f) {
            auto t_curr = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_curr - t_start).count();

            std::cout << "  Cycle " << std::setw(4) << cycle
                      << " | Time: " << std::fixed << std::setprecision(1) << elapsed << "s"
                      << " | FE: " << std::setprecision(5) << mean_fe << " nats"
                      << " | PPL: " << std::setprecision(3) << ppl
                      << " | Accuracy: " << std::setprecision(2) << acc_pct << "%"
                      << " | Peak Acc: " << best_acc << "%\n";
        }

        if (acc_pct >= 99.0f && mean_fe < 0.05f) {
            std::cout << "\n>>> [IDEAL CONVERGENCE ACHIEVED VIA SELF-MODULATION!] <<<\n";
            break;
        }

        if (cycle >= 500) {
            std::cout << "\n>>> Reached benchmark cycle limit (500). <<<\n";
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n========================================================================\n";
    std::cout << "=== EXP-528 TELEMETRY REPORT ===\n";
    std::cout << "Total Relaxation Time: " << std::fixed << std::setprecision(2) << total_time << " s\n";
    std::cout << "Cycles: " << cycle << "\n";
    std::cout << "Peak Accuracy: " << std::setprecision(2) << best_acc << "%\n";
    std::cout << "Best Free Energy: " << std::setprecision(6) << best_fe << " nats\n";
    std::cout << "Best Perplexity: " << std::setprecision(4) << std::exp(best_fe) << "\n";
    std::cout << "========================================================================\n\n";

    std::cout << "RECONSTRUCTION SPEECH PROBES (Deterministic MAP):\n";
    std::cout << "1. Prompt 'Once upon ': \n   -> \"" << model.speak("Once upon ", 250, 0.01f) << "\"\n\n";
    std::cout << "2. Prompt 'Lily loved ': \n   -> \"" << model.speak("Lily loved ", 200, 0.01f) << "\"\n\n";
    std::cout << "3. Prompt 'One sunny day': \n   -> \"" << model.speak("One sunny day", 180, 0.01f) << "\"\n\n";

    return 0;
}
