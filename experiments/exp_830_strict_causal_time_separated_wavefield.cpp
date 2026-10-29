// ============================================================================
// KARYON-CORE EXP-830: Strict Causal Time-Separated Physical Wavefield Engine
// KEP v30.0 Master | Absolute Zero Leakage | Time Step T Updates Before Prediction T+1
// Author: Bazilevs (ProgVM) & Lead AI Cyberneticist
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <fstream>

static uint64_t prng_seed = 0x987654321ABCDEF0ULL;
inline uint64_t xorshift64() {
    prng_seed ^= prng_seed << 13;
    prng_seed ^= prng_seed >> 7;
    prng_seed ^= prng_seed << 17;
    return prng_seed;
}

// Causal Time-Separated Wave Lattice Engine
class CausalWaveLattice {
private:
    int N; // Number of scalar spatial points (64)
    std::vector<float> psi;   // Physical wave amplitude vector at step t
    std::vector<float> theta; // Input coupling vector
    std::vector<float> mu;    // Emission coupling vector
    float alpha, beta, gamma; // Continuous spatial PDE wave diffusion parameters
    float bias;
    float lr;                 // Learning rate

public:
    CausalWaveLattice(int size = 64, float learning_rate = 0.01f)
        : N(size), lr(learning_rate), bias(0.0f) {
        psi.assign(N, 0.0f);
        theta.assign(N, 0.0f);
        mu.assign(N, 0.0f);

        alpha = 0.3f;
        beta  = 0.4f;
        gamma = 0.3f;

        for (int i = 0; i < N; ++i) {
            theta[i] = std::sin(static_cast<float>(i + 1) * 0.1f) * 0.05f;
            mu[i]    = std::cos(static_cast<float>(i + 1) * 0.1f) * 0.05f;
        }
    }

    void reset() {
        std::fill(psi.begin(), psi.end(), 0.0f);
        bias = 0.0f;
        alpha = 0.3f; beta = 0.4f; gamma = 0.3f;
        for (int i = 0; i < N; ++i) {
            theta[i] = std::sin(static_cast<float>(i + 1) * 0.1f) * 0.05f;
            mu[i]    = std::cos(static_cast<float>(i + 1) * 0.1f) * 0.05f;
        }
    }

    struct StepEval {
        float free_energy;
        bool is_correct;
        float pred_spin;
    };

    // STRICT CAUSAL TWO-STAGE TIMESTEP:
    // Stage 1: PREDICT next spin x_{t+1} using strictly state psi(t) before observing x_{t+1}.
    // Stage 2: OBSERVE x_{t+1}, measure error, and update psi state and parameters.

    float predict_next() {
        float sum = bias;
        for (int i = 0; i < N; ++i) {
            sum += mu[i] * psi[i];
        }
        return std::tanh(sum); // Predicted spin for t+1
    }

    StepEval update_with_actual(float pred_spin, float actual_spin) {
        // Measure Variational Free Energy: F = ln(1 + exp(-2 * actual_spin * pred_spin))
        float spin_product = actual_spin * pred_spin;
        float free_energy = std::log(1.0f + std::exp(-2.0f * spin_product));

        bool is_correct = ((pred_spin >= 0.0f && actual_spin >= 0.0f) || (pred_spin < 0.0f && actual_spin < 0.0f));

        // Gradient flow of Free Energy dF / d_pred_spin
        float dF_dpred = -2.0f * actual_spin / (1.0f + std::exp(2.0f * spin_product));
        float dtanh_action = 1.0f - pred_spin * pred_spin;
        float delta_action = dF_dpred * dtanh_action;

        // Parameter update
        bias -= lr * delta_action;
        for (int i = 0; i < N; ++i) {
            mu[i] -= lr * delta_action * psi[i];
        }

        // Advance wave state psi to t+1 using observed actual_spin
        std::vector<float> psi_next(N, 0.0f);
        for (int i = 0; i < N; ++i) {
            int left_idx  = (i == 0) ? (N - 1) : (i - 1);
            int right_idx = (i == N - 1) ? 0 : (i + 1);

            float local_wave_force = alpha * psi[left_idx] + beta * psi[i] + gamma * psi[right_idx] + theta[i] * actual_spin;
            psi_next[i] = std::tanh(local_wave_force);
        }

        // Gradient update for spatial coupling theta and diffusion PDE parameters
        for (int i = 0; i < N; ++i) {
            int left_idx  = (i == 0) ? (N - 1) : (i - 1);
            int right_idx = (i == N - 1) ? 0 : (i + 1);

            float dpsi_i = delta_action * mu[i];
            float dtanh_psi = 1.0f - psi_next[i] * psi_next[i];
            float delta_psi = dpsi_i * dtanh_psi;

            theta[i] -= lr * delta_psi * actual_spin;
            alpha    -= lr * 0.01f * delta_psi * psi[left_idx];
            beta     -= lr * 0.01f * delta_psi * psi[i];
            gamma    -= lr * 0.01f * delta_psi * psi[right_idx];
        }

        psi = psi_next;

        return {free_energy, is_correct, pred_spin};
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-830: STRICT CAUSAL TIME-SEPARATED WAVEFIELD\n";
    std::cout << "  KEP v30.0 Master | Absolute Zero Leakage | Stage 1 Predict, Stage 2 Observe\n";
    std::cout << "=================================================================\n\n";

    CausalWaveLattice engine(64, 0.01f);

    // --- TEST 1: Absolute Random Noise Benchmark (Expected: 50.00% Accuracy, F = 0.6931 nats) ---
    std::cout << "[TEST 1] Pure Unpredictable Random Noise Control (100,000 spins):\n";
    engine.reset();

    size_t correct_rand = 0;
    double f_rand_sum = 0.0;
    const size_t TEST_LEN = 100000;

    for (size_t i = 0; i < TEST_LEN; ++i) {
        float actual_spin = ((xorshift64() & 1) == 1) ? +1.0f : -1.0f;

        // STAGE 1: Predict BEFORE observing actual_spin
        float pred_spin = engine.predict_next();

        // STAGE 2: Observe actual_spin and compute telemetry/update
        auto eval = engine.update_with_actual(pred_spin, actual_spin);

        if (eval.is_correct) correct_rand++;
        f_rand_sum += eval.free_energy;
    }

    double acc_rand = (static_cast<double>(correct_rand) / TEST_LEN) * 100.0;
    double f_rand_avg = f_rand_sum / TEST_LEN;

    std::cout << "  -> Random Stream Accuracy : " << std::fixed << std::setprecision(4) << acc_rand << " % (Expected: ~50.00%)\n";
    std::cout << "  -> Random Free Energy (F) : " << f_rand_avg << " nats (Expected: ln(2) = 0.6931)\n\n";

    // --- TEST 2: Real Non-Repeating C++ Source Stream ---
    std::cout << "[TEST 2] Non-Repeating Real Source Stream (Reading C++ code, Causal Separation):\n";
    engine.reset();

    std::ifstream file("experiments/exp_825_autonomous_recursive_composition.cpp", std::ios::binary);
    std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<float> raw_spins;
    for (char c : file_content) {
        for (int b = 7; b >= 0; --b) {
            int bit = (c >> b) & 1;
            raw_spins.push_back((bit == 1) ? +1.0f : -1.0f);
        }
    }

    size_t quarter_len = raw_spins.size() / 4;

    std::cout << " Quarter | Processed Spins | Accuracy (%) | Free Energy F (nats) | Speed (spins/sec)\n";
    std::cout << "---------|-----------------|--------------|----------------------|------------------\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t q = 0; q < 4; ++q) {
        size_t correct = 0;
        double f_sum = 0.0;

        auto q_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < quarter_len; ++i) {
            size_t idx = q * quarter_len + i;
            float actual_spin = raw_spins[idx];

            // STAGE 1: Predict strictly BEFORE observing actual_spin
            float pred_spin = engine.predict_next();

            // STAGE 2: Observe actual_spin and evaluate/update
            auto eval = engine.update_with_actual(pred_spin, actual_spin);

            if (eval.is_correct) correct++;
            f_sum += eval.free_energy;
        }

        auto q_end = std::chrono::high_resolution_clock::now();
        double q_sec = std::chrono::duration<double>(q_end - q_start).count();

        double acc = (static_cast<double>(correct) / quarter_len) * 100.0;
        double avg_f = f_sum / quarter_len;
        double speed = quarter_len / q_sec;

        std::cout << std::setw(8) << (q + 1) << " | "
                  << std::setw(15) << ((q + 1) * quarter_len) << " | "
                  << std::setw(12) << std::fixed << std::setprecision(4) << acc << " | "
                  << std::setw(20) << std::setprecision(6) << avg_f << " | "
                  << std::setw(17) << std::setprecision(1) << speed << "\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - start_time).count();

    std::cout << "\n=================================================================\n";
    std::cout << "  EXP-830 STRICT CAUSAL TIME-SEPARATED FINAL TELEMETRY\n";
    std::cout << "=================================================================\n";
    std::cout << "  - Random Noise Control Acc   : " << acc_rand << " % (Theoretical Limit: 50.0%)\n";
    std::cout << "  - Real Source Code Acc       : " << ((static_cast<double>(quarter_len * 4) / quarter_len) * 100.0) << " %\n";
    std::cout << "  - Total Spins Processed      : " << (quarter_len * 4) << "\n";
    std::cout << "  - Total Execution Time       : " << total_sec << " seconds\n";
    std::cout << "  - Causal Throughput          : " << ((quarter_len * 4) / total_sec) << " spins/sec\n";
    std::cout << "=================================================================\n";

    return 0;
}
