// ============================================================================
// KARYON-CORE EXP-829: Zero-MatMul Matrix-Free Scalar Wavefield Substrate
// KEP v30.0 Master | Absolute Matrix-Free Physical Wave Dynamics | O(N) Spatial PDE
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

// Zero-MatMul Matrix-Free Continuous Physical Wave Lattice
class ZeroMatMulWaveLattice {
private:
    int N; // Number of scalar lattice points (e.g., 64)
    std::vector<float> psi;       // Wave amplitude string \psi_i(t)
    std::vector<float> theta;     // Local scalar input susceptibility \theta_i
    std::vector<float> mu;        // Local scalar output coupling \mu_i
    float alpha, beta, gamma;     // Continuous spatial wave diffusion coupling parameters
    float bias_x;

    float learning_rate;

public:
    ZeroMatMulWaveLattice(int lattice_size = 64, float lr = 0.02f)
        : N(lattice_size), learning_rate(lr), bias_x(0.0f) {
        psi.assign(N, 0.0f);
        theta.assign(N, 0.0f);
        mu.assign(N, 0.0f);

        // Spatial PDE continuous wave diffusion parameters
        alpha = 0.4f; // Left neighbor coupling
        beta  = 0.2f; // Self inertia
        gamma = 0.4f; // Right neighbor coupling

        for (int i = 0; i < N; ++i) {
            theta[i] = std::sin(static_cast<float>(i + 1) * 0.2f) * 0.1f;
            mu[i]    = std::cos(static_cast<float>(i + 1) * 0.2f) * 0.1f;
        }
    }

    struct StepResult {
        float free_energy;
        bool is_correct;
        float pred_spin;
    };

    void reset() {
        std::fill(psi.begin(), psi.end(), 0.0f);
        for (int i = 0; i < N; ++i) {
            theta[i] = std::sin(static_cast<float>(i + 1) * 0.2f) * 0.1f;
            mu[i]    = std::cos(static_cast<float>(i + 1) * 0.2f) * 0.1f;
        }
        bias_x = 0.0f;
    }

    // Matrix-Free Spatial Wave PDE Update Step: O(N) operations, ZERO Matrix Multiplications!
    StepResult step(float raw_spin) {
        std::vector<float> psi_next(N, 0.0f);

        // 1. Local Spatial Wave PDE Propagation:
        // \psi_i(t+1) = \tanh( \alpha \psi_{i-1}(t) + \beta \psi_i(t) + \gamma \psi_{i+1}(t) + \theta_i \cdot x_t )
        for (int i = 0; i < N; ++i) {
            int left_idx  = (i == 0) ? (N - 1) : (i - 1); // Periodic boundary condition
            int right_idx = (i == N - 1) ? 0 : (i + 1);

            float local_wave_force = alpha * psi[left_idx] + beta * psi[i] + gamma * psi[right_idx] + theta[i] * raw_spin;
            psi_next[i] = std::tanh(local_wave_force);
        }

        // 2. Action Potential Constructive Wave Interference Emission:
        // \hat{x}_t = \tanh( \sum_i \mu_i \psi_i(t+1) + bias )
        float Interference_sum = bias_x;
        for (int i = 0; i < N; ++i) {
            Interference_sum += mu[i] * psi_next[i];
        }
        float pred_spin = std::tanh(Interference_sum);

        // 3. Exact Variational Free Energy: F = ln(1 + exp(-2 * x_t * \hat{x}_t))
        float spin_product = raw_spin * pred_spin;
        float free_energy = std::log(1.0f + std::exp(-2.0f * spin_product));

        bool is_correct = ((pred_spin >= 0.0f && raw_spin >= 0.0f) || (pred_spin < 0.0f && raw_spin < 0.0f));

        // 4. Matrix-Free Gradient Flow for Field Parameters
        float dF_dpred = -2.0f * raw_spin / (1.0f + std::exp(2.0f * spin_product));
        float dtanh_action = 1.0f - pred_spin * pred_spin;
        float delta_action = dF_dpred * dtanh_action;

        // Scalar parameter updates
        bias_x -= learning_rate * delta_action;

        for (int i = 0; i < N; ++i) {
            mu[i] -= learning_rate * delta_action * psi_next[i];

            float dpsi_i = delta_action * mu[i];
            float dtanh_psi = 1.0f - psi_next[i] * psi_next[i];
            float delta_psi = dpsi_i * dtanh_psi;

            theta[i] -= learning_rate * delta_psi * raw_spin;
        }

        // Update wave PDE parameters (alpha, beta, gamma)
        for (int i = 0; i < N; ++i) {
            int left_idx  = (i == 0) ? (N - 1) : (i - 1);
            int right_idx = (i == N - 1) ? 0 : (i + 1);

            float dpsi_i = delta_action * mu[i];
            float dtanh_psi = 1.0f - psi_next[i] * psi_next[i];
            float delta_psi = dpsi_i * dtanh_psi;

            alpha -= learning_rate * 0.01f * delta_psi * psi[left_idx];
            beta  -= learning_rate * 0.01f * delta_psi * psi[i];
            gamma -= learning_rate * 0.01f * delta_psi * psi[right_idx];
        }

        // Wave state advance
        psi = psi_next;

        return {free_energy, is_correct, pred_spin};
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-829: ZERO-MATMUL MATRIX-FREE SCALAR WAVEFIELD\n";
    std::cout << "  KEP v30.0 Master | Pure O(N) Spatial Wave PDE | Zero Matrices\n";
    std::cout << "=================================================================\n\n";

    ZeroMatMulWaveLattice lattice(64, 0.03f);

    // --- TEST 1: Control Benchmark on Pure Random Noise (Target: 50.00%, F = 0.6931 nats) ---
    std::cout << "[TEST 1] Pure Unpredictable Random Noise Control (100,000 spins):\n";
    lattice.reset();

    size_t correct_rand = 0;
    double f_rand_sum = 0.0;
    const size_t TEST_LEN = 100000;

    for (size_t i = 0; i < TEST_LEN; ++i) {
        float raw_spin = ((xorshift64() & 1) == 1) ? +1.0f : -1.0f;
        auto res = lattice.step(raw_spin);

        if (res.is_correct) correct_rand++;
        f_rand_sum += res.free_energy;
    }

    double acc_rand = (static_cast<double>(correct_rand) / TEST_LEN) * 100.0;
    double f_rand_avg = f_rand_sum / TEST_LEN;

    std::cout << "  -> Random Stream Accuracy : " << std::fixed << std::setprecision(4) << acc_rand << " % (Expected: ~50.00%)\n";
    std::cout << "  -> Random Free Energy (F) : " << f_rand_avg << " nats (Expected: ln(2) = 0.6931)\n\n";

    // --- TEST 2: Real Non-Repeating Source Stream ---
    std::cout << "[TEST 2] Non-Repeating Real Source Stream (Reading C++ code, ZERO MatMul):\n";
    lattice.reset();

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
            float spin = raw_spins[idx];

            auto res = lattice.step(spin);

            if (res.is_correct) correct++;
            f_sum += res.free_energy;
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
    std::cout << "  EXP-829 ZERO-MATMUL FINAL TELEMETRY\n";
    std::cout << "=================================================================\n";
    std::cout << "  - Random Noise Control Acc   : " << acc_rand << " % (Theoretical Limit: 50.0%)\n";
    std::cout << "  - Total Spins Processed      : " << (quarter_len * 4) << "\n";
    std::cout << "  - Total Execution Time       : " << total_sec << " seconds\n";
    std::cout << "  - Matrix-Free Throughput     : " << ((quarter_len * 4) / total_sec) << " spins/sec\n";
    std::cout << "=================================================================\n";

    return 0;
}
