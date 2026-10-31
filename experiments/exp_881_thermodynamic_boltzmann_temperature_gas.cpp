// ============================================================================
// KARYON-CORE EXPERIMENT EXP-881
// Title: Thermodynamic Boltzmann Temperature Gas & Phase Attractor Coupling (TBT-PAC)
// Standard: KEP v37.0 Sovereign Master | Substrate: Pure V=2 Bit-Spin
// Hypothesis: Fixing the inverted thermal feedback loop by treating Entropy-Gas G_i 
//             as physical Boltzmann Temperature T_i in denominator (P = sigmoid(Delta E / T)) 
//             prevents overconfidence shocks (FE <= ln 2 during exploration) and enables 
//             true multi-domain attractor condensation (Math, Pendulum, Chess).
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

class HardwareEntropySource {
private:
    std::random_device rd;
public:
    double sample_gaussian_noise(double stddev = 0.015) {
        uint32_t val = rd();
        double u1 = (static_cast<double>(val & 0xFFFF) + 1.0) / 65537.0;
        double u2 = (static_cast<double>((val >> 16) & 0xFFFF) + 1.0) / 65537.0;
        double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI * u2);
        return z0 * stddev;
    }
};

class CorrectedTEGUWS_Engine {
private:
    std::vector<Complex> wavefield;
    std::vector<double> metric_tensor;
    std::vector<double> temperature_gas; // T_i (Entropy-Gas = Physical Temperature)
    std::vector<double> frequencies;
    HardwareEntropySource hw_noise;

    uint64_t total_bits_assimilated = 0;
    double cumulative_free_energy = 0.0;
    double mean_temp_accumulator = 0.0;

public:
    CorrectedTEGUWS_Engine() {
        wavefield.resize(FIELD_DIM);
        metric_tensor.resize(FIELD_DIM * FIELD_DIM);
        temperature_gas.resize(FIELD_DIM, 1.0); // Initial Temperature T = 1.0 (High Plasticity)
        frequencies.resize(FIELD_DIM);

        for (size_t i = 0; i < FIELD_DIM; ++i) {
            double phase = (static_cast<double>(i) / static_cast<double>(FIELD_DIM)) * 2.0 * PI;
            wavefield[i] = std::polar(0.2, phase);
            frequencies[i] = 0.05 + 0.01 * (i % 16);
        }

        for (size_t i = 0; i < FIELD_DIM; ++i) {
            for (size_t j = 0; j < FIELD_DIM; ++j) {
                size_t idx = i * FIELD_DIM + j;
                metric_tensor[idx] = (i == j) ? 1.0 : (0.02 * std::sin(static_cast<double>(i + j)));
            }
        }
    }

    double assimilate_bit(uint8_t raw_bit) {
        double bit_spin = (raw_bit & 1) ? +1.0 : -1.0;
        double hw_jitter = hw_noise.sample_gaussian_noise(0.01);

        // Compute Wavefield Energy Difference
        double delta_E = 0.0;
        double mean_T = 0.0;

        for (size_t i = 0; i < FIELD_DIM; ++i) {
            double phase = std::arg(wavefield[i]);
            // Energy difference between positive spin (+pi/4) and negative spin (-pi/4)
            delta_E += std::cos(phase - 0.25 * PI) - std::cos(phase + 0.25 * PI);
            mean_T += temperature_gas[i];
        }
        mean_T /= static_cast<double>(FIELD_DIM);
        mean_T = std::max(0.05, mean_T); // Minimum thermal floor

        // Correct Physical Boltzmann Distribution: P(pos) = 1 / (1 + exp(-Delta_E / T))
        double scaled_logits = delta_E / mean_T;
        double prob_pos = 1.0 / (1.0 + std::exp(-scaled_logits));
        prob_pos = std::clamp(prob_pos, 1e-7, 1.0 - 1e-7);

        double actual_prob = (bit_spin > 0.0) ? prob_pos : (1.0 - prob_pos);
        double instant_free_energy = -std::log(actual_prob);
        cumulative_free_energy += instant_free_energy;

        // Correct Thermodynamic Temperature Evolution:
        // Free Energy (surprise) releases heat -> increases Temperature T_i
        // Dissipation (cooling) lowers Temperature T_i towards thermal equilibrium
        for (size_t i = 0; i < FIELD_DIM; ++i) {
            temperature_gas[i] += 0.01 * (instant_free_energy - 0.2); // Heat injection
        }

        // Laplacian Heat Transport across metric tensor G_ij
        std::vector<double> next_temp(FIELD_DIM, 0.0);
        for (size_t i = 0; i < FIELD_DIM; ++i) {
            double diffusion_sum = 0.0;
            size_t row_offset = i * FIELD_DIM;
            for (size_t j = 0; j < FIELD_DIM; ++j) {
                diffusion_sum += std::abs(metric_tensor[row_offset + j]) * (temperature_gas[j] - temperature_gas[i]);
            }
            // Heat equation with natural cooling
            double updated_T = temperature_gas[i] + 0.05 * diffusion_sum - 0.02 * (temperature_gas[i] - 0.1);
            next_temp[i] = std::clamp(updated_T, 0.05, 5.0);
        }
        temperature_gas = std::move(next_temp);
        mean_temp_accumulator += mean_T;

        // Wavefield State Evolution under Thermal Coupling
        std::vector<Complex> next_wavefield(FIELD_DIM);
        for (size_t i = 0; i < FIELD_DIM; ++i) {
            Complex coupling_sum(0.0, 0.0);
            size_t row_offset = i * FIELD_DIM;
            for (size_t j = 0; j < FIELD_DIM; ++j) {
                coupling_sum += metric_tensor[row_offset + j] * wavefield[j];
            }

            // Frequency modulation driven by prediction error & spin
            frequencies[i] += 0.001 * bit_spin * std::real(coupling_sum);

            double phase_shift = frequencies[i] + hw_jitter + 0.01 * temperature_gas[i];
            Complex rot = coupling_sum * std::polar(1.0, phase_shift);
            Complex perturbed = rot + Complex(0.2 * bit_spin, 0.1 * bit_spin);

            double mag = std::abs(perturbed);
            if (mag > 1e-9) perturbed = (perturbed / mag) * std::tanh(mag);
            next_wavefield[i] = perturbed;
        }
        wavefield = std::move(next_wavefield);

        // Plasticity Annealing over Metric Tensor G_ij:
        // High Temperature = High Plasticity (Melting bonds)
        // Low Temperature = Frozen Invariants (Crystalline Attractor)
        for (size_t i = 0; i < FIELD_DIM; ++i) {
            double re_i = std::real(wavefield[i]);
            double local_plasticity = 0.01 * temperature_gas[i];
            size_t row_offset = i * FIELD_DIM;
            for (size_t j = 0; j < FIELD_DIM; ++j) {
                double re_j = std::real(wavefield[j]);
                size_t idx = row_offset + j;
                // Hebbian phase-lock update with metric regularization
                metric_tensor[idx] += local_plasticity * (bit_spin * re_i * re_j - 0.01 * metric_tensor[idx]);
            }
        }

        total_bits_assimilated++;
        return instant_free_energy;
    }

    double get_mean_free_energy() const {
        return (total_bits_assimilated > 0) ? (cumulative_free_energy / total_bits_assimilated) : 0.0;
    }

    double get_mean_temperature() const {
        return (total_bits_assimilated > 0) ? (mean_temp_accumulator / total_bits_assimilated) : 0.0;
    }
};

// DOMAIN GENERATORS
std::vector<uint8_t> generate_math_bitstream(size_t samples = 1200) {
    std::string stream = "";
    for (size_t i = 1; i <= samples; ++i) {
        double x = i * 0.1;
        double y = x * x + 3 * x + 2.0;
        double dist_sq = x * x + y * y;
        stream += "x=" + std::to_string(x) + ";y=" + std::to_string(y) + ";r2=" + std::to_string(dist_sq) + "\n";
    }
    return std::vector<uint8_t>(stream.begin(), stream.end());
}

std::vector<uint8_t> generate_pendulum_bitstream(size_t steps = 1200) {
    std::string stream = "";
    double theta = PI / 4.0, omega = 0.0, dt = 0.05, g = 9.81, L = 1.0, b = 0.1;

    for (size_t i = 0; i < steps; ++i) {
        auto derivatives = [&](double t_, double o_) { return - (g / L) * std::sin(t_) - b * o_; };
        double k1_o = derivatives(theta, omega), k1_t = omega;
        double k2_o = derivatives(theta + 0.5 * dt * k1_t, omega + 0.5 * dt * k1_o), k2_t = omega + 0.5 * dt * k1_o;
        double k3_o = derivatives(theta + 0.5 * dt * k2_t, omega + 0.5 * dt * k2_o), k3_t = omega + 0.5 * dt * k2_o;
        double k4_o = derivatives(theta + dt * k3_t, omega + dt * k3_o), k4_t = omega + dt * k3_o;

        theta += (dt / 6.0) * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t);
        omega += (dt / 6.0) * (k1_o + 2.0 * k2_o + 2.0 * k3_o + k4_o);
        stream += "th:" + std::to_string(theta) + ",om:" + std::to_string(omega) + "\n";
    }
    return std::vector<uint8_t>(stream.begin(), stream.end());
}

std::vector<uint8_t> generate_chess_bitstream() {
    std::string pgn = 
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 "
        "11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 "
        "20. Nbd2 Nxd6 21. Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5\n";
    std::string full_stream = "";
    for (int i = 0; i < 20; ++i) full_stream += pgn;
    return std::vector<uint8_t>(full_stream.begin(), full_stream.end());
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-881: BOLTZMANN TEMPERATURE GAS BENCHMARK (FIXED)\n";
    std::cout << " Hypothesis: T in Denominator eliminates thermal explosion loop\n";
    std::cout << "===================================================================\n\n";

    // 1. MATH
    auto math_bytes = generate_math_bitstream(1200);
    CorrectedTEGUWS_Engine math_engine;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint8_t byte_val : math_bytes) {
        for (int b = 7; b >= 0; --b) math_engine.assimilate_bit((byte_val >> b) & 1);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double math_dur = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "--- 1. MATH DOMAIN ---\n";
    std::cout << " [+] Mean Free Energy: " << math_engine.get_mean_free_energy() << " nats/bit\n";
    std::cout << " [+] Mean Temperature T: " << math_engine.get_mean_temperature() << "\n";
    std::cout << " [+] Throughput: " << (math_bytes.size() * 8 / math_dur) << " bits/sec\n\n";

    // 2. PENDULUM
    auto pend_bytes = generate_pendulum_bitstream(1200);
    CorrectedTEGUWS_Engine pend_engine;
    t0 = std::chrono::high_resolution_clock::now();
    for (uint8_t byte_val : pend_bytes) {
        for (int b = 7; b >= 0; --b) pend_engine.assimilate_bit((byte_val >> b) & 1);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double pend_dur = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "--- 2. PENDULUM DOMAIN ---\n";
    std::cout << " [+] Mean Free Energy: " << pend_engine.get_mean_free_energy() << " nats/bit\n";
    std::cout << " [+] Mean Temperature T: " << pend_engine.get_mean_temperature() << "\n";
    std::cout << " [+] Throughput: " << (pend_bytes.size() * 8 / pend_dur) << " bits/sec\n\n";

    // 3. CHESS
    auto chess_bytes = generate_chess_bitstream();
    CorrectedTEGUWS_Engine chess_engine;
    t0 = std::chrono::high_resolution_clock::now();
    for (uint8_t byte_val : chess_bytes) {
        for (int b = 7; b >= 0; --b) chess_engine.assimilate_bit((byte_val >> b) & 1);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double chess_dur = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "--- 3. CHESS DOMAIN ---\n";
    std::cout << " [+] Mean Free Energy: " << chess_engine.get_mean_free_energy() << " nats/bit\n";
    std::cout << " [+] Mean Temperature T: " << chess_engine.get_mean_temperature() << "\n";
    std::cout << " [+] Throughput: " << (chess_bytes.size() * 8 / chess_dur) << " bits/sec\n\n";

    return 0;
}
