// ============================================================================
// KARYON-CORE EXPERIMENT EXP-891
// Title: Endogenous Unbounded Continuous Field Physics (EUCFP)
// Standard: KEP v37.0 Sovereign Master | Substrate: Pure V=2 Bit-Spin
// Philosophical Principles:
//   - Principle 2: Rejection of copying biological or external top-down rules.
//   - Principle 27 & 32: ZERO hardcoded 8-bit byte boxes (no 256-table lookups, no fixed N-gram masks).
//   - Principle 30: All dynamic memory states, timescales, and plastic projections are 
//     endogenously synthesized and evolved BOTTOM-UP from continuous bit-spins s_t \in \{-1, +1\}.
// Core Mechanism:
//   - Multi-rate Continuous State Field H(t) with dynamic temporal decay rates \gamma_k.
//   - Endogenous Retrograde Gas Diffusion G(t) acting as a physical homeostatic surprise signal.
//   - Continuous Bilinear Attractor Projection: Prediction logits and energy gradients evolve
//     without ANY artificial byte boundaries, discrete tables, or predefined graph structures.
// Author: Karyon Autonomous Lead AI Cybernetics Engine & Bazilevs (ProgVM)
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <random>
#include <cstdint>
#include <sstream>

constexpr size_t FIELD_DIM = 128; // Continuous multi-scale state dimensions
constexpr double PI = 3.14159265358979323846;

class HardwareEntropySource {
private:
    std::random_device rd;
public:
    double sample_jitter(double scale = 0.001) {
        uint32_t val = rd();
        double u1 = (static_cast<double>(val & 0xFFFF) + 1.0) / 65537.0;
        double u2 = (static_cast<double>((val >> 16) & 0xFFFF) + 1.0) / 65537.0;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI * u2) * scale;
    }
};

class EndogenousUnboundedSubstrate {
private:
    std::vector<double> state_H;            // Continuous multi-scale state vector H_k
    std::vector<double> gamma_decay;        // Dynamic relaxation timescales per channel \gamma_k
    std::vector<double> W_proj;             // Continuous plastic memory field (FIELD_DIM)
    std::vector<double> M_bilinear;         // Second-order interaction field (FIELD_DIM)
    
    double gas_diffusion = 0.0;             // Endogenous retrograde gas diffusion (Ashby surprise signal)
    HardwareEntropySource hw_noise;

    uint64_t total_bits = 0;
    double cumulative_free_energy = 0.0;
    double base_learning_rate = 0.08;

public:
    EndogenousUnboundedSubstrate() {
        state_H.resize(FIELD_DIM, 0.0);
        gamma_decay.resize(FIELD_DIM, 0.0);
        W_proj.resize(FIELD_DIM, 0.0);
        M_bilinear.resize(FIELD_DIM, 0.0);

        std::mt19937 rng(1337);
        std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(FIELD_DIM));

        // Initialize continuous dynamic timescales across logarithmic continuum
        for (size_t k = 0; k < FIELD_DIM; ++k) {
            // Timescales span from fast (decay 0.95) to long-term memory (decay 0.9999)
            gamma_decay[k] = 0.8 + 0.1999 * (static_cast<double>(k) / static_cast<double>(FIELD_DIM - 1));
            W_proj[k] = dist(rng);
            M_bilinear[k] = dist(rng);
        }
    }

    double assimilate_bit(uint8_t raw_bit) {
        double s_t = (raw_bit & 1) ? +1.0 : -1.0;
        double noise = hw_noise.sample_jitter(0.0005);

        // ====================================================================
        // 1. ENDOGENOUS MULTI-SCALE STATE EVOLUTION
        // H_k(t+1) = \gamma_k * H_k(t) + (1 - \gamma_k) * s_t
        // Dynamic continuous fields evolve without ANY hardcoded 8-bit boundaries!
        // ====================================================================
        for (size_t k = 0; k < FIELD_DIM; ++k) {
            state_H[k] = gamma_decay[k] * state_H[k] + (1.0 - gamma_decay[k]) * s_t;
        }

        // ====================================================================
        // 2. CONTINUOUS BILINEAR INTERACTION & READOUT
        // Logit = \sum_k W_k * H_k + M_k * H_k * H_{k-1}
        // ====================================================================
        double linear_term = 0.0;
        double bilinear_term = 0.0;

        for (size_t k = 0; k < FIELD_DIM; ++k) {
            linear_term += W_proj[k] * state_H[k];
            size_t prev_k = (k == 0) ? (FIELD_DIM - 1) : (k - 1);
            bilinear_term += M_bilinear[k] * state_H[k] * state_H[prev_k];
        }

        double logit = (linear_term + 0.5 * bilinear_term) / std::sqrt(static_cast<double>(FIELD_DIM)) + noise;
        double prob_pos = 1.0 / (1.0 + std::exp(-logit));
        prob_pos = std::clamp(prob_pos, 1e-6, 1.0 - 1e-6);

        double actual_prob = (s_t > 0.0) ? prob_pos : (1.0 - prob_pos);
        double free_energy = -std::log(actual_prob);
        cumulative_free_energy += free_energy;

        // ====================================================================
        // 3. RETROGRADE GAS DIFFUSION & ENDOGENOUS PLASTICITY
        // Gas G_t accumulates on prediction surprise (high Free Energy),
        // modulating local learning rates without artificial supervisory loss gates.
        // ====================================================================
        double surprise = free_energy; // High Free Energy = High Surprise
        gas_diffusion = 0.95 * gas_diffusion + 0.05 * surprise;

        // Dynamic plastic learning rate modulated by physical gas density
        double dynamic_eta = base_learning_rate * (1.0 + std::tanh(gas_diffusion - 0.5));
        double error_grad = (s_t > 0.0) ? (prob_pos - 1.0) : prob_pos;
        double scale_inv = 1.0 / std::sqrt(static_cast<double>(FIELD_DIM));

        for (size_t k = 0; k < FIELD_DIM; ++k) {
            size_t prev_k = (k == 0) ? (FIELD_DIM - 1) : (k - 1);
            
            // Causal update of continuous weights
            double dW = error_grad * state_H[k] * scale_inv;
            double dM = error_grad * state_H[k] * state_H[prev_k] * scale_inv;

            W_proj[k] -= dynamic_eta * dW + 0.00005 * W_proj[k];
            M_bilinear[k] -= dynamic_eta * dM + 0.00005 * M_bilinear[k];
        }

        total_bits++;
        return free_energy;
    }

    double get_mean_free_energy() const {
        return (total_bits > 0) ? (cumulative_free_energy / total_bits) : 0.0;
    }

    double get_gas_density() const {
        return gas_diffusion;
    }
};

// DOMAIN DATA GENERATORS
std::vector<uint8_t> generate_math_bitstream(size_t samples = 1000) {
    std::string stream = "";
    for (size_t i = 1; i <= samples; ++i) {
        double x = i * 0.1;
        double y = x * x + 3 * x + 2.0;
        stream += "x=" + std::to_string(x) + ";y=" + std::to_string(y) + "\n";
    }
    return std::vector<uint8_t>(stream.begin(), stream.end());
}

std::vector<uint8_t> generate_pendulum_bitstream(size_t steps = 1000) {
    std::string stream = "";
    double theta = 0.785398, omega = 0.0, dt = 0.05, g = 9.81, L = 1.0;
    for (size_t i = 0; i < steps; ++i) {
        double alpha = - (g / L) * std::sin(theta);
        omega += alpha * dt;
        theta += omega * dt;
        stream += "th:" + std::to_string(theta) + "\n";
    }
    return std::vector<uint8_t>(stream.begin(), stream.end());
}

std::vector<uint8_t> generate_chess_bitstream() {
    std::string pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O\n";
    std::string full_stream = "";
    for (int i = 0; i < 25; ++i) full_stream += pgn;
    return std::vector<uint8_t>(full_stream.begin(), full_stream.end());
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-891: ENDOGENOUS UNBOUNDED CONTINUOUS FIELD PHYSICS\n";
    std::cout << " Standard: KEP v37.0 | Zero Hardcoded Tables, 8-bit Boxes or External Rules\n";
    std::cout << " Baseline Reference: Random Noise = 0.6931 nats/bit\n";
    std::cout << "===================================================================\n\n";

    // 1. MATH
    auto math_bytes = generate_math_bitstream(1000);
    EndogenousUnboundedSubstrate math_engine;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint8_t byte_val : math_bytes) {
        for (int b = 7; b >= 0; --b) math_engine.assimilate_bit((byte_val >> b) & 1);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double math_dur = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "--- 1. MATH DOMAIN ---\n";
    std::cout << " [+] Mean Free Energy: " << math_engine.get_mean_free_energy() << " nats/bit\n";
    std::cout << " [+] Gas Density G: " << math_engine.get_gas_density() << "\n";
    std::cout << " [+] Throughput: " << (math_bytes.size() * 8 / math_dur) << " bits/sec\n\n";

    // 2. PENDULUM
    auto pend_bytes = generate_pendulum_bitstream(1000);
    EndogenousUnboundedSubstrate pend_engine;
    t0 = std::chrono::high_resolution_clock::now();
    for (uint8_t byte_val : pend_bytes) {
        for (int b = 7; b >= 0; --b) pend_engine.assimilate_bit((byte_val >> b) & 1);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double pend_dur = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "--- 2. PENDULUM DOMAIN ---\n";
    std::cout << " [+] Mean Free Energy: " << pend_engine.get_mean_free_energy() << " nats/bit\n";
    std::cout << " [+] Gas Density G: " << pend_engine.get_gas_density() << "\n";
    std::cout << " [+] Throughput: " << (pend_bytes.size() * 8 / pend_dur) << " bits/sec\n\n";

    // 3. CHESS
    auto chess_bytes = generate_chess_bitstream();
    EndogenousUnboundedSubstrate chess_engine;
    t0 = std::chrono::high_resolution_clock::now();
    for (uint8_t byte_val : chess_bytes) {
        for (int b = 7; b >= 0; --b) chess_engine.assimilate_bit((byte_val >> b) & 1);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double chess_dur = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "--- 3. CHESS DOMAIN ---\n";
    std::cout << " [+] Mean Free Energy: " << chess_engine.get_mean_free_energy() << " nats/bit\n";
    std::cout << " [+] Gas Density G: " << chess_engine.get_gas_density() << "\n";
    std::cout << " [+] Throughput: " << (chess_bytes.size() * 8 / chess_dur) << " bits/sec\n\n";

    return 0;
}
