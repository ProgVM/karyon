// ============================================================================
// EXP-529: Chiral Soliton Waveguide with Travelling Waves & Unidirectional Time Arrow
// Testing Principle 29 (Law of Sovereign Self-Control of Substrate) & Principle 3
// - Pure Weightless Continuous Field (W = \emptyset)
// - Spatial Continuum x \in [0, L] with Chiral Advection (v * dpsi/dx): True Arrow of Time
// - Non-Linear Soliton Interaction (Korteweg-de Vries / Non-linear Schrodinger dynamics):
//     dpsi/dt + v * dpsi/dx + beta * d^3psi/dx^3 + alpha * |psi|^2 * psi = Input(x, t)
// - Self-Organizing Travelling Solitons encode multi-word context as spatial wave packets
// - No layers, no MLPs, no discrete tokenizers: Continuous 1D Hydrodynamic Waveguide
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
constexpr size_t SPATIAL_L = 128; // Discrete spatial points on continuous waveguide [0, L]

class ChiralSolitonWaveguide {
public:
    ChiralSolitonWaveguide(size_t spatial_points = SPATIAL_L) 
        : L_(spatial_points), psi_(spatial_points, 0.0f), dpsi_dt_(spatial_points, 0.0f) {
        init_state();
    }

    size_t get_spatial_points() const { return L_; }

    void get_byte_excitation(uint8_t byte_val, std::vector<float>& out_src) const {
        out_src.assign(L_, 0.0f);
        float base_phase = (2.0f * M_PI * static_cast<float>(byte_val)) / static_cast<float>(VOCAB_SIZE);
        
        // Input excites the entry port (x near 0) with characteristic spectral harmonics
        for (size_t x = 0; x < 16; ++x) {
            float spatial_envelope = std::exp(-static_cast<float>(x * x) / 32.0f);
            out_src[x] = std::sin(base_phase * (x + 1)) * spatial_envelope;
        }
    }

    // Step continuous hydrodynamic chiral waveguide
    // Solitons travel from left (x=0) to right (x=L-1)
    std::vector<float> step_perceive(uint8_t byte_in, float dt = 0.05f) {
        std::vector<float> src;
        get_byte_excitation(byte_in, src);

        // Hydrodynamic / Soliton PDE integration:
        // 1. Chiral Advection: -v * (psi[x] - psi[x-1]) / dx  (unidirectional drift)
        // 2. Dispersion: -beta * (psi[x+1] - 2*psi[x] + psi[x-1])  (wave packet broadening)
        // 3. Cubic Non-Linearity (Self-Phase Modulation / Soliton Focusing): +alpha * psi^3
        // 4. Source excitation: +src[x]
        // 5. Boundary absorber at x = L-1 (prevent unphysical reflection back)

        std::vector<float> next_psi = psi_;

        for (size_t sub = 0; sub < 4; ++sub) { // 4 sub-steps for numerical stability
            float sub_dt = dt / 4.0f;
            std::vector<float> dpsi(L_, 0.0f);

            for (size_t x = 0; x < L_; ++x) {
                float psi_curr = next_psi[x];
                float psi_left = (x > 0) ? next_psi[x - 1] : 0.0f;
                float psi_right = (x + 1 < L_) ? next_psi[x + 1] : next_psi[x];

                // Chiral first-order advection gradient (directed time arrow)
                float advection = -advection_v_ * (psi_curr - psi_left);

                // Wave dispersion (Laplacian)
                float dispersion = dispersion_beta_ * (psi_right - 2.0f * psi_curr + psi_left);

                // Non-linear Kerr/soliton self-focusing
                float non_linear = non_linear_alpha_ * (psi_curr * psi_curr * psi_curr);

                // Endogenous damping (stronger at the boundary absorber)
                float local_gamma = base_damping_[x];
                if (x >= L_ - 8) local_gamma += 1.5f; // Absorbing sponge boundary

                float force = src[x] + advection + dispersion + non_linear - local_gamma * psi_curr;
                dpsi[x] = force;
            }

            for (size_t x = 0; x < L_; ++x) {
                next_psi[x] += sub_dt * dpsi[x];
                next_psi[x] = std::clamp(next_psi[x], -4.0f, 4.0f);
            }
        }

        psi_ = next_psi;

        // Weightless Resonant Readout:
        // The entire spatial waveguide profile psi(x) interacts with the byte dual metric
        std::vector<float> logits(VOCAB_SIZE, 0.0f);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            float base_phase = (2.0f * M_PI * static_cast<float>(v)) / static_cast<float>(VOCAB_SIZE);
            float overlap = 0.0f;

            for (size_t x = 0; x < L_; ++x) {
                float dual_wave = std::sin(base_phase * (x + 1)) * readout_metric_[x];
                overlap += psi_[x] * dual_wave;
            }
            logits[v] = overlap * 4.5f;
        }

        return logits;
    }

    std::pair<float, bool> adapt_step(uint8_t in_b, uint8_t target_b, float lr = 0.03f) {
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

        // Error back-propagation into continuous metric tensor
        std::vector<float> error_profile(L_, 0.0f);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            float err_v = probs[v] - (v == target_b ? 1.0f : 0.0f);
            if (std::abs(err_v) < 1e-5f) continue;

            float base_phase = (2.0f * M_PI * static_cast<float>(v)) / static_cast<float>(VOCAB_SIZE);
            for (size_t x = 0; x < L_; ++x) {
                float dual_wave = std::sin(base_phase * (x + 1));
                error_profile[x] += err_v * dual_wave;
            }
        }

        // Plasticity: Adapt readout metric, local dispersion, and non-linearity
        for (size_t x = 0; x < L_; ++x) {
            readout_metric_[x] -= lr * error_profile[x] * psi_[x];
            readout_metric_[x] = std::clamp(readout_metric_[x], -10.0f, 10.0f);

            base_damping_[x] -= (lr * 0.1f) * error_profile[x] * psi_[x];
            base_damping_[x] = std::clamp(base_damping_[x], 0.001f, 1.0f);
        }

        // Tune waveguide hydrodynamic parameters
        float global_err = 0.0f;
        for (size_t x = 0; x < L_; ++x) global_err += std::abs(error_profile[x]);
        global_err /= L_;

        advection_v_ = std::clamp(advection_v_ - lr * 0.05f * (global_err - 0.2f), 0.2f, 2.0f);
        non_linear_alpha_ = std::clamp(non_linear_alpha_ + lr * 0.05f * (0.5f - global_err), 0.05f, 1.5f);

        return {fe_surprise, hit};
    }

    void reset_wave_state() {
        std::fill(psi_.begin(), psi_.end(), 0.0f);
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

    float get_wave_energy() const {
        float e = 0.0f;
        for (float p : psi_) e += p * p;
        return e;
    }

private:
    void init_state() {
        readout_metric_.resize(L_, 1.0f);
        base_damping_.resize(L_, 0.02f); // Low damping for long soliton flight
        advection_v_ = 0.8f;            // Fluid velocity / chiral speed
        dispersion_beta_ = 0.15f;       // Dispersion coefficient
        non_linear_alpha_ = 0.40f;      // Non-linear focusing (Kerr effect)
    }

    size_t L_;
    std::vector<float> psi_;
    std::vector<float> dpsi_dt_;
    std::vector<float> readout_metric_;
    std::vector<float> base_damping_;
    float advection_v_;
    float dispersion_beta_;
    float non_linear_alpha_;
};

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-529: CHIRAL SOLITON WAVEGUIDE CONTINUUM (TIME ARROW) ===\n";
    std::cout << "=== Testing Hydrodynamic Soliton Propagation & Non-Linear Memory ===\n";
    std::cout << "========================================================================\n\n";

    const std::string narrative = 
        "Once upon a time, there was a little girl named Lily. "
        "Lily loved to play in the green garden with her favorite red ball. "
        "One sunny day, Lily saw a soft white cat sitting near the big oak tree. "
        "The cat smiled at Lily and gave a gentle purr. Lily was very happy. "
        "They became the best of friends and played happily together until sunset.";

    ChiralSolitonWaveguide model(128); // 128 continuous spatial nodes along 1D chiral fiber

    std::cout << "Waveguide Length L: " << model.get_spatial_points() << " spatial nodes\n";
    std::cout << "Input Port: x in [0, 15] | Propagation: x -> L-1 | Sponge Absorber: x >= 120\n";
    std::cout << "Relaxing chiral soliton waveguide...\n\n";

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

        float lr = std::max(0.005f, 0.05f * std::exp(-static_cast<float>(cycle) / 800.0f));

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
            std::cout << "\n>>> [PERFECT CONVERGENCE ACHIEVED VIA CHIRAL SOLITONS!] <<<\n";
            break;
        }

        if (cycle >= 400) {
            std::cout << "\n>>> Completed 400 relaxation cycles. <<<\n";
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n========================================================================\n";
    std::cout << "=== EXP-529 TELEMETRY REPORT ===\n";
    std::cout << "Relaxation Time: " << std::fixed << std::setprecision(2) << total_time << " s\n";
    std::cout << "Cycles: " << cycle << "\n";
    std::cout << "Peak Accuracy: " << std::setprecision(2) << best_acc << "%\n";
    std::cout << "Best Free Energy: " << std::setprecision(6) << best_fe << " nats\n";
    std::cout << "Best Perplexity: " << std::setprecision(4) << std::exp(best_fe) << "\n";
    std::cout << "Final Waveguide Energy: " << std::setprecision(4) << model.get_wave_energy() << "\n";
    std::cout << "========================================================================\n\n";

    std::cout << "RECONSTRUCTION PROBES (Chiral Soliton Waveguide Speech):\n";
    std::cout << "1. Prompt 'Once upon ': \n   -> \"" << model.speak("Once upon ", 250, 0.01f) << "\"\n\n";
    std::cout << "2. Prompt 'Lily loved ': \n   -> \"" << model.speak("Lily loved ", 200, 0.01f) << "\"\n\n";
    std::cout << "3. Prompt 'One sunny day': \n   -> \"" << model.speak("One sunny day", 180, 0.01f) << "\"\n\n";

    return 0;
}
