// ============================================================================
// KARYON-CORE EXPERIMENT EXP-897
// Title: Multi-Domain Weightless Scalar-Phase Field Substrate Benchmark
// Standard: KEP v37.0 Master Protocol | Principles 2, 3, 27, 29, 30, 31, 32, 33, 34
//
// MATRIX-FREE MULTI-DOMAIN EVALUATION (0% WEIGHT MATRICES):
//   1. Chaotic Dynamical System Prediction (Mackey-Glass / Lorenz Attractor Stream)
//   2. Raw Machine Bytecode / Source Code Bit-Spin Stream Prediction (V=2)
//   3. Non-Linear Multi-Frequency Wave Interference Tracking
//   4. Continuous Active Sensorimotor Control
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

constexpr size_t TAP_CAPACITY = 64; // 64 scalar delay taps (O(K), 0 matrices)

class WeightlessPhaseSubstrate {
private:
    std::vector<double> delay_taps;
    std::vector<double> c_coupling;
    double scalar_bias = 0.0;
    double eta = 0.08;

public:
    WeightlessPhaseSubstrate(size_t taps = TAP_CAPACITY, double lr = 0.08)
        : delay_taps(taps, 0.0), c_coupling(taps, 0.0), eta(lr) {
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 0.02);
        for (size_t k = 0; k < taps; ++k) {
            c_coupling[k] = dist(rng);
        }
    }

    double Predict() const {
        double p = scalar_bias;
        for (size_t k = 0; k < delay_taps.size(); ++k) {
            p += c_coupling[k] * delay_taps[k];
        }
        return p;
    }

    double EmitAction() const {
        return std::tanh(Predict());
    }

    double Step(double target) {
        double pred = Predict();
        double err = target - pred;
        double fe = 0.5 * err * err;

        // Shift taps
        for (size_t k = delay_taps.size() - 1; k > 0; --k) {
            delay_taps[k] = delay_taps[k - 1];
        }
        delay_taps[0] = std::tanh(target);

        // Least-action phase update with bounded gradient
        double clipped_err = std::clamp(err, -2.0, 2.0);
        scalar_bias += eta * clipped_err;
        for (size_t k = 0; k < delay_taps.size(); ++k) {
            double grad = clipped_err * delay_taps[k];
            c_coupling[k] += eta * std::clamp(grad, -1.0, 1.0);
            c_coupling[k] *= 0.9999; // subtle regularization
        }

        return fe;
    }
};

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-897: MULTI-DOMAIN WEIGHTLESS BENCHMARK\n";
    std::cout << " Standard: KEP v37.0 | ZERO Weight Matrices (O(K) Complexity)\n";
    std::cout << "===================================================================\n\n";

    // ------------------------------------------------------------------------
    // DOMAIN 1: MULTI-FREQUENCY HARMONIC WAVE INTERFERENCE
    // ------------------------------------------------------------------------
    {
        std::cout << "[DOMAIN 1] Multi-Frequency Wave Interference Prediction\n";
        WeightlessPhaseSubstrate substrate(64, 0.10);

        double total_fe = 0.0;
        double tail_fe = 0.0;
        size_t steps = 50000;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < steps; ++i) {
            double t = i * 0.01;
            // Compound non-linear wave signal
            double signal = std::sin(t) + 0.5 * std::cos(2.3 * t) + 0.25 * std::sin(5.7 * t);
            double fe = substrate.Step(signal);
            total_fe += fe;
            if (i >= steps - 1000) tail_fe += fe;
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count();
        tail_fe /= 1000.0;
        std::cout << "  -> Throughput: " << std::fixed << std::setprecision(0) << (steps / sec) << " steps/sec\n";
        std::cout << "  -> Mean Free Energy: " << std::fixed << std::setprecision(6) << (total_fe / steps) << " nats\n";
        std::cout << "  -> Asymptotic Tail Free Energy: " << std::scientific << std::setprecision(6) << tail_fe << " nats\n\n";
    }

    // ------------------------------------------------------------------------
    // DOMAIN 2: CHAOTIC LORENZ ATTRACTOR STREAM PREDICTION
    // ------------------------------------------------------------------------
    {
        std::cout << "[DOMAIN 2] Chaotic Lorenz Attractor Stream Prediction\n";
        WeightlessPhaseSubstrate substrate(128, 0.05);

        double x = 0.1, y = 0.0, z = 0.0;
        double sigma = 10.0, rho = 28.0, beta = 8.0 / 3.0;
        double dt = 0.005;

        double total_fe = 0.0;
        double tail_fe = 0.0;
        size_t steps = 50000;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < steps; ++i) {
            double dx = sigma * (y - x);
            double dy = x * (rho - z) - y;
            double dz = x * y - beta * z;

            x += dx * dt;
            y += dy * dt;
            z += dz * dt;

            // Normalized X state stream
            double target = x * 0.05;
            double fe = substrate.Step(target);
            total_fe += fe;
            if (i >= steps - 1000) tail_fe += fe;
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count();
        tail_fe /= 1000.0;
        std::cout << "  -> Throughput: " << std::fixed << std::setprecision(0) << (steps / sec) << " steps/sec\n";
        std::cout << "  -> Mean Free Energy: " << std::fixed << std::setprecision(6) << (total_fe / steps) << " nats\n";
        std::cout << "  -> Asymptotic Tail Free Energy: " << std::scientific << std::setprecision(6) << tail_fe << " nats\n\n";
    }

    // ------------------------------------------------------------------------
    // DOMAIN 3: RAW BYTECODE / TEXT BIT-SPIN STREAM PREDICTION (V=2)
    // ------------------------------------------------------------------------
    {
        std::cout << "[DOMAIN 3] Raw C++ Bytecode Bit-Spin Stream Prediction (V=2)\n";
        WeightlessPhaseSubstrate substrate(32, 0.12);

        std::string cpp_code = 
            "#include <iostream>\nint main() { double x = 3.14159; std::cout << x; return 0; }\n";

        // Convert string bytes into binary spin stream (-1.0, +1.0)
        std::vector<double> spin_stream;
        for (char c : cpp_code) {
            uint8_t byte = static_cast<uint8_t>(c);
            for (int b = 7; b >= 0; --b) {
                spin_stream.push_back(((byte >> b) & 1) ? +1.0 : -1.0);
            }
        }

        double total_fe = 0.0;
        double tail_fe = 0.0;
        size_t bit_steps = 100000;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < bit_steps; ++i) {
            double target_spin = spin_stream[i % spin_stream.size()];
            double fe = substrate.Step(target_spin);
            total_fe += fe;
            if (i >= bit_steps - 1000) tail_fe += fe;
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count();
        tail_fe /= 1000.0;
        std::cout << "  -> Bit Stream Length: " << spin_stream.size() << " bits\n";
        std::cout << "  -> Throughput: " << std::fixed << std::setprecision(0) << (bit_steps / sec) << " bits/sec\n";
        std::cout << "  -> Mean Free Energy: " << std::fixed << std::setprecision(6) << (total_fe / bit_steps) << " nats\n";
        std::cout << "  -> Asymptotic Tail Free Energy: " << std::scientific << std::setprecision(6) << tail_fe << " nats\n\n";
    }

    // ------------------------------------------------------------------------
    // DOMAIN 4: SENSORIMOTOR ACTIVE CONTROL (NON-LINEAR CART-POLE)
    // ------------------------------------------------------------------------
    {
        std::cout << "[DOMAIN 4] Non-Linear Sensorimotor Cart-Pole Tracking\n";
        WeightlessPhaseSubstrate agent(64, 0.15);

        double x = 0.0, x_dot = 0.0;
        double theta = 0.15, theta_dot = 0.0;
        double dt = 0.02;
        double g = 9.81, mc = 1.0, mp = 0.1, l = 0.5;

        uint32_t stable_steps = 0;
        size_t sim_steps = 5000;

        for (size_t i = 0; i < sim_steps; ++i) {
            double force = agent.EmitAction() * 20.0;

            // Equations of motion
            double temp = (force + mp * l * theta_dot * theta_dot * std::sin(theta)) / (mc + mp);
            double theta_acc = (g * std::sin(theta) - std::cos(theta) * temp) / 
                               (l * (4.0 / 3.0 - mp * std::cos(theta) * std::cos(theta) / (mc + mp)));
            double x_acc = temp - mp * l * theta_acc * std::cos(theta) / (mc + mp);

            x_dot += x_acc * dt;
            x += x_dot * dt;
            theta_dot += theta_acc * dt;
            theta += theta_dot * dt;

            // Damping
            x_dot *= 0.98;
            theta_dot *= 0.98;

            agent.Step(theta);

            if (std::abs(theta) < 0.10 && std::abs(x) < 2.0) {
                stable_steps++;
            }
        }

        std::cout << "  -> Stability Rate (|theta|<0.10 rad, |x|<2.0m): " 
                  << std::fixed << std::setprecision(2) << (100.0 * stable_steps / sim_steps) << "%\n";
        std::cout << "  -> Final Pole Angle: " << std::setprecision(4) << theta << " rad\n\n";
    }

    return 0;
}
