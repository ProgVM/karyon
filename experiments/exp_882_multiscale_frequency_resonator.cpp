// ============================================================================
// KARYON-CORE EXPERIMENT EXP-882
// Title: Dynamic Multi-Scale Frequency Resonance & Hebbian PAC Phase (DMFR-PAC)
// Standard: KEP v37.0 Sovereign Master | Substrate: Pure V=2 Bit-Spin
// Hypothesis: Combining dynamic adaptive frequency tuning w_i with multi-layer
//             bit-spin aggregation reduces Free Energy from ~6.2 nats/bit down below 
//             the random guessing baseline (0.693 nats/bit) on complex non-linguistic data.
// Author: Karyon Autonomous Lead AI Cybernetics Engine & Bazilevs (ProgVM)
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <random>
#include <cstdint>
#include <sstream>

constexpr size_t FIELD_DIM = 128;
constexpr double PI = 3.14159265358979323846;

using Complex = std::complex<double>;

class MultiScaleResonatorEngine {
private:
    std::vector<Complex> wavefield;
    std::vector<double> metric_tensor;
    std::vector<double> frequencies;
    std::vector<double> phase_accumulator;

    uint64_t total_bits = 0;
    double cumulative_free_energy = 0.0;

public:
    MultiScaleResonatorEngine() {
        wavefield.resize(FIELD_DIM);
        metric_tensor.resize(FIELD_DIM * FIELD_DIM);
        frequencies.resize(FIELD_DIM);
        phase_accumulator.resize(FIELD_DIM, 0.0);

        for (size_t i = 0; i < FIELD_DIM; ++i) {
            double phase = (static_cast<double>(i) / static_cast<double>(FIELD_DIM)) * 2.0 * PI;
            wavefield[i] = std::polar(0.5, phase);
            frequencies[i] = 0.01 + 0.1 * (static_cast<double>(i) / static_cast<double>(FIELD_DIM));
        }

        for (size_t i = 0; i < FIELD_DIM; ++i) {
            for (size_t j = 0; j < FIELD_DIM; ++j) {
                size_t idx = i * FIELD_DIM + j;
                metric_tensor[idx] = (i == j) ? 1.0 : 0.05 * std::cos(static_cast<double>(i - j));
            }
        }
    }

    double assimilate_bit(uint8_t raw_bit) {
        double bit_spin = (raw_bit & 1) ? +1.0 : -1.0;

        // Compute Phase Interference Energy
        double total_real_coupling = 0.0;
        for (size_t i = 0; i < FIELD_DIM; ++i) {
            double local_coupling = 0.0;
            size_t row_offset = i * FIELD_DIM;
            for (size_t j = 0; j < FIELD_DIM; ++j) {
                local_coupling += metric_tensor[row_offset + j] * std::real(wavefield[j]);
            }
            total_real_coupling += local_coupling;
        }

        double normalized_logit = total_real_coupling / static_cast<double>(FIELD_DIM);
        double prob_pos = 1.0 / (1.0 + std::exp(-2.0 * normalized_logit));
        prob_pos = std::clamp(prob_pos, 1e-6, 1.0 - 1e-6);

        double actual_prob = (bit_spin > 0.0) ? prob_pos : (1.0 - prob_pos);
        double free_energy = -std::log(actual_prob);
        cumulative_free_energy += free_energy;

        // Adaptive Phase Update & Hebbian Frequency Tuning
        for (size_t i = 0; i < FIELD_DIM; ++i) {
            size_t row_offset = i * FIELD_DIM;
            double coupling = 0.0;
            for (size_t j = 0; j < FIELD_DIM; ++j) {
                coupling += metric_tensor[row_offset + j] * std::real(wavefield[j]);
            }

            // Phase error drives frequency resonance
            double error = bit_spin - std::tanh(coupling);
            frequencies[i] += 0.01 * error * std::sin(phase_accumulator[i]);
            frequencies[i] = std::clamp(frequencies[i], 0.001, 2.0 * PI);

            phase_accumulator[i] += frequencies[i];
            if (phase_accumulator[i] > 2.0 * PI) phase_accumulator[i] -= 2.0 * PI;

            Complex rot = std::polar(1.0, phase_accumulator[i]);
            wavefield[i] = 0.9 * wavefield[i] + 0.1 * rot * bit_spin;

            // Hebbian Metric Learning
            for (size_t j = 0; j < FIELD_DIM; ++j) {
                size_t idx = row_offset + j;
                metric_tensor[idx] += 0.005 * (bit_spin * std::real(wavefield[i]) * std::real(wavefield[j]) - 0.01 * metric_tensor[idx]);
            }
        }

        total_bits++;
        return free_energy;
    }

    double get_mean_free_energy() const {
        return (total_bits > 0) ? (cumulative_free_energy / total_bits) : 0.0;
    }
};

// DATA GENERATORS
std::vector<uint8_t> generate_math_bitstream(size_t samples = 1200) {
    std::string stream = "";
    for (size_t i = 1; i <= samples; ++i) {
        double x = i * 0.1;
        double y = x * x + 3 * x + 2.0;
        stream += "x=" + std::to_string(x) + ";y=" + std::to_string(y) + "\n";
    }
    return std::vector<uint8_t>(stream.begin(), stream.end());
}

std::vector<uint8_t> generate_pendulum_bitstream(size_t steps = 1200) {
    std::string stream = "";
    double theta = PI / 4.0, omega = 0.0, dt = 0.05, g = 9.81, L = 1.0;
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
    for (int i = 0; i < 30; ++i) full_stream += pgn;
    return std::vector<uint8_t>(full_stream.begin(), full_stream.end());
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-882: MULTI-SCALE FREQUENCY RESONANCE (DMFR-PAC)\n";
    std::cout << " Baseline Target: Free Energy < 0.693 nats/bit (Beating Random Guesses)\n";
    std::cout << "===================================================================\n\n";

    // MATH
    auto math_bytes = generate_math_bitstream(1200);
    MultiScaleResonatorEngine math_engine;
    for (uint8_t byte_val : math_bytes) {
        for (int b = 7; b >= 0; --b) math_engine.assimilate_bit((byte_val >> b) & 1);
    }
    std::cout << "--- 1. MATH DOMAIN ---\n";
    std::cout << " [+] Mean Free Energy: " << math_engine.get_mean_free_energy() << " nats/bit\n\n";

    // PENDULUM
    auto pend_bytes = generate_pendulum_bitstream(1200);
    MultiScaleResonatorEngine pend_engine;
    for (uint8_t byte_val : pend_bytes) {
        for (int b = 7; b >= 0; --b) pend_engine.assimilate_bit((byte_val >> b) & 1);
    }
    std::cout << "--- 2. PENDULUM DOMAIN ---\n";
    std::cout << " [+] Mean Free Energy: " << pend_engine.get_mean_free_energy() << " nats/bit\n\n";

    // CHESS
    auto chess_bytes = generate_chess_bitstream();
    MultiScaleResonatorEngine chess_engine;
    for (uint8_t byte_val : chess_bytes) {
        for (int b = 7; b >= 0; --b) chess_engine.assimilate_bit((byte_val >> b) & 1);
    }
    std::cout << "--- 3. CHESS DOMAIN ---\n";
    std::cout << " [+] Mean Free Energy: " << chess_engine.get_mean_free_energy() << " nats/bit\n\n";

    return 0;
}
