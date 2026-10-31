// ============================================================================
// KARYON-CORE EXPERIMENT EXP-888
// Title: Endogenous Homeostatic Free-Energy Attractor Substrate (EH-FEAS)
// Standard: KEP v37.0 Sovereign Master | Substrate: Pure V=2 Bit-Spin
// Philosophical Principles: KEP Rules #1-#6, Principles 2, 4, 7, 27, 28, 29, 30, 31 & 32.
// Core Breakthrough Idea:
//   - The environment possesses an intrinsic Ashby Homeostatic Drive toward low Free Energy (Stability).
//   - Analytical Gradient \nabla_H F(H, s_t) drives rapid continuous relaxation of state H into 
//     deep low-entropy attractor basins.
//   - Zero top-down hardcoded equations or discrete N-gram tables.
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

constexpr size_t FIELD_DIM = 64;
constexpr size_t RELAXATION_STEPS = 4; // Physical relaxation steps
constexpr double PI = 3.14159265358979323846;

class HardwareEntropySource {
private:
    std::random_device rd;
public:
    double sample_jitter(double scale = 0.005) {
        uint32_t val = rd();
        double u1 = (static_cast<double>(val & 0xFFFF) + 1.0) / 65537.0;
        double u2 = (static_cast<double>((val >> 16) & 0xFFFF) + 1.0) / 65537.0;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI * u2) * scale;
    }
};

class EndogenousHomeostaticSubstrate {
private:
    std::vector<double> state_h;             // Continuous state vector H_i
    std::vector<double> W_field;             // Continuous plastic field transformation matrix (FIELD_DIM x FIELD_DIM)
    std::vector<double> V_potential;         // Endogenous potential bias vector (FIELD_DIM)
    HardwareEntropySource hw_noise;

    uint64_t total_bits = 0;
    double cumulative_free_energy = 0.0;

    double eta_state = 0.2;                  // Homeostatic relaxation rate
    double eta_law = 0.05;                   // Plasticity rate

public:
    EndogenousHomeostaticSubstrate() {
        state_h.resize(FIELD_DIM, 0.0);
        W_field.resize(FIELD_DIM * FIELD_DIM, 0.0);
        V_potential.resize(FIELD_DIM, 0.0);

        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(FIELD_DIM));

        for (size_t i = 0; i < FIELD_DIM * FIELD_DIM; ++i) {
            W_field[i] = dist(rng);
        }
        for (size_t i = 0; i < FIELD_DIM; ++i) {
            state_h[i] = dist(rng);
            V_potential[i] = 0.0;
        }
    }

    double assimilate_bit(uint8_t raw_bit) {
        double s_t = (raw_bit & 1) ? +1.0 : -1.0;
        
        // ====================================================================
        // 1. ENDOGENOUS HOMEOSTATIC RELAXATION WITH ANALYTICAL GRADIENT
        // Substrate state H is drawn toward low Free Energy basins (Ashby Drive)
        // ====================================================================
        double logit = 0.0;
        double prob_pos = 0.5;

        for (size_t step = 0; step < RELAXATION_STEPS; ++step) {
            // Forward pass: compute linear field activation A_i = V_i + \sum W_ij H_j
            std::vector<double> act(FIELD_DIM, 0.0);
            std::vector<double> tanh_act(FIELD_DIM, 0.0);
            double alignment = 0.0;

            for (size_t i = 0; i < FIELD_DIM; ++i) {
                double a = V_potential[i];
                size_t row_off = i * FIELD_DIM;
                for (size_t j = 0; j < FIELD_DIM; ++j) {
                    a += W_field[row_off + j] * state_h[j];
                }
                act[i] = a;
                tanh_act[i] = std::tanh(a);
                alignment += state_h[i] * tanh_act[i];
            }

            logit = alignment / std::sqrt(static_cast<double>(FIELD_DIM));
            prob_pos = 1.0 / (1.0 + std::exp(-logit));
            prob_pos = std::clamp(prob_pos, 1e-6, 1.0 - 1e-6);

            // Error gradient w.r.t logit: dF / dLogit
            double dF_dlogit = (s_t > 0.0) ? (prob_pos - 1.0) : prob_pos;
            double inv_norm = 1.0 / std::sqrt(static_cast<double>(FIELD_DIM));

            // Analytical Gradient \nabla_H F(H)
            std::vector<double> grad_h(FIELD_DIM, 0.0);
            for (size_t i = 0; i < FIELD_DIM; ++i) {
                // Direct contribution to alignment: tanh(A_i)
                double g = tanh_act[i];
                // Indirect contribution through W_ji:
                for (size_t j = 0; j < FIELD_DIM; ++j) {
                    double dtanh = 1.0 - tanh_act[j] * tanh_act[j];
                    g += state_h[j] * dtanh * W_field[j * FIELD_DIM + i];
                }
                grad_h[i] = dF_dlogit * g * inv_norm + 0.02 * state_h[i]; // + quadratic state bound
            }

            // Slide state H down Free Energy gradient towards homeostatic basin
            for (size_t i = 0; i < FIELD_DIM; ++i) {
                double noise = hw_noise.sample_jitter(0.002);
                state_h[i] -= eta_state * grad_h[i] + noise;
                state_h[i] = std::clamp(state_h[i], -3.0, 3.0);
            }
        }

        // Final Free Energy Evaluation
        double actual_prob = (s_t > 0.0) ? prob_pos : (1.0 - prob_pos);
        double bit_free_energy = -std::log(actual_prob);
        cumulative_free_energy += bit_free_energy;

        // ====================================================================
        // 2. PLASTICITY (Carving low-energy attractors into laws W)
        // ====================================================================
        double dF_dlogit = (s_t > 0.0) ? (prob_pos - 1.0) : prob_pos;
        double inv_norm = 1.0 / std::sqrt(static_cast<double>(FIELD_DIM));

        for (size_t i = 0; i < FIELD_DIM; ++i) {
            size_t row_off = i * FIELD_DIM;
            double h_i = state_h[i];
            
            V_potential[i] -= eta_law * dF_dlogit * h_i * inv_norm;

            for (size_t j = 0; j < FIELD_DIM; ++j) {
                double delta_W = eta_law * dF_dlogit * h_i * state_h[j] * inv_norm;
                W_field[row_off + j] -= delta_W + 0.0001 * W_field[row_off + j];
            }
        }

        total_bits++;
        return bit_free_energy;
    }

    double get_mean_free_energy() const {
        return (total_bits > 0) ? (cumulative_free_energy / total_bits) : 0.0;
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
    std::cout << " KARYON-CORE EXP-888: ENDOGENOUS HOMEOSTATIC FREE-ENERGY ATTRACTOR\n";
    std::cout << " Standard: KEP v37.0 | Ashby Homeostatic Drive (Analytical FE Relaxation)\n";
    std::cout << " Baseline Reference: Random Noise = 0.6931 nats/bit\n";
    std::cout << "===================================================================\n\n";

    // 1. MATH
    auto math_bytes = generate_math_bitstream(1000);
    EndogenousHomeostaticSubstrate math_engine;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint8_t byte_val : math_bytes) {
        for (int b = 7; b >= 0; --b) math_engine.assimilate_bit((byte_val >> b) & 1);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double math_dur = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "--- 1. MATH DOMAIN ---\n";
    std::cout << " [+] Mean Free Energy: " << math_engine.get_mean_free_energy() << " nats/bit\n";
    std::cout << " [+] Throughput: " << (math_bytes.size() * 8 / math_dur) << " bits/sec\n\n";

    // 2. PENDULUM
    auto pend_bytes = generate_pendulum_bitstream(1000);
    EndogenousHomeostaticSubstrate pend_engine;
    t0 = std::chrono::high_resolution_clock::now();
    for (uint8_t byte_val : pend_bytes) {
        for (int b = 7; b >= 0; --b) pend_engine.assimilate_bit((byte_val >> b) & 1);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double pend_dur = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "--- 2. PENDULUM DOMAIN ---\n";
    std::cout << " [+] Mean Free Energy: " << pend_engine.get_mean_free_energy() << " nats/bit\n";
    std::cout << " [+] Throughput: " << (pend_bytes.size() * 8 / pend_dur) << " bits/sec\n\n";

    // 3. CHESS
    auto chess_bytes = generate_chess_bitstream();
    EndogenousHomeostaticSubstrate chess_engine;
    t0 = std::chrono::high_resolution_clock::now();
    for (uint8_t byte_val : chess_bytes) {
        for (int b = 7; b >= 0; --b) chess_engine.assimilate_bit((byte_val >> b) & 1);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double chess_dur = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "--- 3. CHESS DOMAIN ---\n";
    std::cout << " [+] Mean Free Energy: " << chess_engine.get_mean_free_energy() << " nats/bit\n";
    std::cout << " [+] Throughput: " << (chess_bytes.size() * 8 / chess_dur) << " bits/sec\n\n";

    return 0;
}
