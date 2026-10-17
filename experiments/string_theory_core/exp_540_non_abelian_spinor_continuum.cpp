// ============================================================================
// EXP-540: Non-Abelian SU(2) Spinor String Continuum & Topological Braiding
// Testing Principle 2, 3, 4, 27, 28, 29 (Continuous Single-Pass Stream N=1)
//
// 1. NON-ABELIAN SU(2) SPINOR MANIFOLD:
//    - Eliminate commutative U(1) phase bleed.
//    - Wave function is a 2-component spinor: \Psi = [ \psi_up, \psi_down ]^T in C^2.
//    - Pauli spin matrices: \sigma_x, \sigma_y, \sigma_z.
//    - Non-commutativity: [\sigma_a, \sigma_b] = 2i \epsilon_{abc} \sigma_c.
//    - Operator order is strictly preserved! Action of 'T' followed by 'h'
//      yields a state fundamentally distinct from 'h' followed by 'T'.
// 2. DISCRETE PHASE LOCKING / TOPOLOGICAL BRAIDING:
//    - Spin-orbit coupling: H_so = \lambda (\vec{\sigma} \times \vec{p}) \cdot \vec{E}.
//    - Deep non-linear potential wells snap continuous trajectories into
//      discrete attractor basins (combating continuous phase blur).
// 3. ZERO EPOCHS & RIGID EMPIRICAL METRICS:
//    - Single-pass stream (N=1).
//    - Real token hit rate, byte accuracy, and exact word reproduction.
//    - No Goodhart smoothing or pseudo-syllabic excuses.
// ============================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <complex>
#include <x86intrin.h>
#include <omp.h>

using ComplexF = std::complex<float>;

struct Spinor2 {
    ComplexF up;
    ComplexF down;

    Spinor2() : up(0.0f, 0.0f), down(0.0f, 0.0f) {}
    Spinor2(ComplexF u, ComplexF d) : up(u), down(d) {}

    Spinor2 operator+(const Spinor2& o) const { return Spinor2(up + o.up, down + o.down); }
    Spinor2 operator-(const Spinor2& o) const { return Spinor2(up - o.up, down - o.down); }
    Spinor2 operator*(float s) const { return Spinor2(up * s, down * s); }
    Spinor2& operator+=(const Spinor2& o) { up += o.up; down += o.down; return *this; }
    Spinor2& operator-=(const Spinor2& o) { up -= o.up; down -= o.down; return *this; }

    float norm_sq() const { return std::norm(up) + std::norm(down); }
    float norm() const { return std::sqrt(norm_sq()); }

    void normalize() {
        float n = norm();
        if (n > 1e-7f) {
            up /= n;
            down /= n;
        }
    }
};

// Pauli matrices acting on Spinor2
inline Spinor2 apply_sigma_x(const Spinor2& s) {
    return Spinor2(s.down, s.up);
}

inline Spinor2 apply_sigma_y(const Spinor2& s) {
    ComplexF i_unit(0.0f, 1.0f);
    return Spinor2(-i_unit * s.down, i_unit * s.up);
}

inline Spinor2 apply_sigma_z(const Spinor2& s) {
    return Spinor2(s.up, -s.down);
}

// SU(2) Lie algebra rotation: exp(i * theta * (n_x \sigma_x + n_y \sigma_y + n_z \sigma_z))
inline Spinor2 su2_rotate(const Spinor2& s, float theta, float nx, float ny, float nz) {
    float len = std::sqrt(nx * nx + ny * ny + nz * nz + 1e-8f);
    nx /= len; ny /= len; nz /= len;

    float half_theta = 0.5f * theta;
    float cos_t = std::cos(half_theta);
    float sin_t = std::sin(half_theta);

    ComplexF i_unit(0.0f, 1.0f);

    // Matrix elements of U = cos(t/2) * I + i * sin(t/2) * (n . sigma)
    ComplexF u00 = ComplexF(cos_t, 0.0f) + i_unit * (sin_t * nz);
    ComplexF u01 = i_unit * (sin_t * nx) + (sin_t * ny);
    ComplexF u10 = i_unit * (sin_t * nx) - (sin_t * ny);
    ComplexF u11 = ComplexF(cos_t, 0.0f) - i_unit * (sin_t * nz);

    return Spinor2(u00 * s.up + u01 * s.down, u10 * s.up + u11 * s.down);
}

inline float get_hardware_jitter() {
    uint64_t tsc = __rdtsc();
    uint32_t bits = static_cast<uint32_t>((tsc ^ (tsc >> 11) ^ (tsc >> 23)) & 0xFFFFFF);
    return (static_cast<float>(bits) / 8388607.5f) - 1.0f;
}

class NonAbelianSpinorContinuum {
public:
    NonAbelianSpinorContinuum(size_t in_dim, size_t n_spinors = 128, size_t n_modes = 64)
        : in_dim_(in_dim), n_spinors_(n_spinors), n_modes_(n_modes),
          total_modes_(n_spinors * n_modes), global_tau_(0.0f) {
        init_spinor_field();
    }

    void step_spinor_flow(const std::vector<float>& sensory_in, float dt = 0.03f) {
        global_tau_ += dt;

        // 1. Sensory D-Brane Influx projected onto SU(2) Generator Angles
        std::vector<Spinor2> boundary_drive(n_spinors_);
        for (size_t s = 0; s < n_spinors_; ++s) {
            float rot_x = 0.0f, rot_y = 0.0f, rot_z = 0.0f;
            for (size_t i = 0; i < in_dim_; ++i) {
                rot_x += sensory_in[i] * in_weight_x_[s * in_dim_ + i];
                rot_y += sensory_in[i] * in_weight_y_[s * in_dim_ + i];
                rot_z += sensory_in[i] * in_weight_z_[s * in_dim_ + i];
            }
            float angle = std::sqrt(rot_x * rot_x + rot_y * rot_y + rot_z * rot_z);
            Spinor2 base(ComplexF(1.0f, 0.0f), ComplexF(0.0f, 0.0f));
            boundary_drive[s] = su2_rotate(base, angle, rot_x, rot_y, rot_z);
        }

        float hw_j = 0.001f * get_hardware_jitter();

        // 2. Non-Abelian Spinor Evolution on Lattice Modes
        #pragma omp parallel for schedule(static)
        for (size_t s = 0; s < n_spinors_; ++s) {
            size_t base = s * n_modes_;
            Spinor2 drive = boundary_drive[s];

            for (size_t m = 0; m < n_modes_; ++m) {
                size_t idx = base + m;
                Spinor2 psi = psi_[idx];
                Spinor2 pi_s = pi_[idx];

                // Non-Abelian Gauge Laplacian with Wilson line link variables U_ij
                Spinor2 non_abelian_laplacian;
                if (m > 0 && m + 1 < n_modes_) {
                    float theta_link = gauge_link_angle_[idx];
                    Spinor2 fwd = su2_rotate(psi_[idx + 1], theta_link, 1.0f, 0.0f, 0.0f);
                    Spinor2 bwd = su2_rotate(psi_[idx - 1], -theta_link, 1.0f, 0.0f, 0.0f);
                    non_abelian_laplacian = fwd - psi * 2.0f + bwd;
                }

                // Spin-Orbit & Non-Commutative Interaction: \vec{\sigma} coupling
                Spinor2 sx = apply_sigma_x(psi);
                Spinor2 sy = apply_sigma_y(psi);
                Spinor2 sz = apply_sigma_z(psi);
                Spinor2 spin_torque = sx * 0.15f + sy * 0.15f + sz * 0.20f;

                // Deep Attractor Wells (snap into discrete non-commuting poles)
                // Minimizes continuous phase blur!
                float density = psi.norm_sq();
                float well_factor = 2.0f * (1.0f - density); // Potential V = -(|\Psi|^2 - 1)^2
                Spinor2 well_force = psi * well_factor;

                Spinor2 restore = psi * (-omega_sq_[idx]);
                Spinor2 damping = pi_s * (-gamma_[idx]);
                Spinor2 drive_term = drive * eigen_basis_[m] + Spinor2(ComplexF(hw_j, hw_j), ComplexF(hw_j, -hw_j));

                Spinor2 acc = restore + non_abelian_laplacian * 1.8f + spin_torque + well_force + damping + drive_term;

                pi_[idx] += acc * dt;
                float p_mag = pi_[idx].norm();
                if (p_mag > 4.0f) pi_[idx] = pi_[idx] * (4.0f / p_mag);

                psi_[idx] += pi_[idx] * dt;
                float q_mag = psi_[idx].norm();
                if (q_mag > 3.0f) psi_[idx] = psi_[idx] * (3.0f / q_mag);

                // Non-commutative state braid history
                braid_history_[idx] = braid_history_[idx] * decay_[idx] + psi_[idx] * (1.0f - decay_[idx]);
            }
        }
    }

    // Exact Categorical Byte Probabilities via Quantum Non-Abelian Projection
    std::vector<float> compute_byte_probabilities() {
        std::vector<float> probs(256, 0.0f);
        float sum_p = 0.0f;

        for (size_t b = 0; b < 256; ++b) {
            float total_projection = 0.0f;
            size_t stride = total_modes_ / 64;
            for (size_t k = 0; k < 64; ++k) {
                Spinor2 field_s = psi_[k * stride];
                Spinor2 basis_s = byte_spinor_basis_[b * 64 + k];

                // Quantum overlap: |<basis|field>|^2
                ComplexF overlap = std::conj(basis_s.up) * field_s.up + std::conj(basis_s.down) * field_s.down;
                total_projection += std::norm(overlap);
            }
            probs[b] = total_projection;
            sum_p += total_projection;
        }

        if (sum_p > 1e-7f) {
            for (size_t b = 0; b < 256; ++b) probs[b] /= sum_p;
        } else {
            probs[32] = 1.0f;
        }
        return probs;
    }

    // Plastic Update on Real Discreteness
    float adapt_authentic_stream(const std::vector<float>& sensory_in, uint8_t target_byte, float lr = 0.035f) {
        step_spinor_flow(sensory_in);
        std::vector<float> probs = compute_byte_probabilities();

        float target_p = std::clamp(probs[target_byte], 1e-7f, 1.0f);
        float surprisal = -std::log(target_p);

        size_t stride = total_modes_ / 64;
        for (size_t b = 0; b < 256; ++b) {
            float err = (b == target_byte) ? (probs[b] - 1.0f) : probs[b];
            if (std::abs(err) < 1e-4f) continue;

            for (size_t k = 0; k < 64; ++k) {
                Spinor2 field_s = psi_[k * stride];
                byte_spinor_basis_[b * 64 + k] -= field_s * (lr * err);
                byte_spinor_basis_[b * 64 + k].normalize();
            }
        }

        // Wilson link gauge self-tuning
        #pragma omp parallel for schedule(static)
        for (size_t s = 0; s < n_spinors_; ++s) {
            size_t base = s * n_modes_;
            for (size_t m = 0; m < n_modes_; m += 4) {
                size_t idx = base + m;
                float d_angle = std::arg(psi_[idx].up) - std::arg(braid_history_[idx].up);
                gauge_link_angle_[idx] = std::clamp(gauge_link_angle_[idx] + lr * 0.02f * std::sin(d_angle), -3.14f, 3.14f);
            }
        }

        return surprisal;
    }

    size_t total_modes() const { return total_modes_; }
    float global_tau() const { return global_tau_; }

private:
    void init_spinor_field() {
        std::mt19937 rng(4242);
        std::normal_distribution<float> dist(0.0f, 0.05f);

        psi_.resize(total_modes_);
        pi_.resize(total_modes_);
        braid_history_.resize(total_modes_);
        gauge_link_angle_.resize(total_modes_, 0.0f);
        omega_sq_.resize(total_modes_);
        gamma_.resize(total_modes_);
        decay_.resize(total_modes_);

        for (size_t s = 0; s < n_spinors_; ++s) {
            for (size_t m = 0; m < n_modes_; ++m) {
                size_t idx = s * n_modes_ + m;
                float ratio = static_cast<float>(m + 1) / static_cast<float>(n_modes_);
                omega_sq_[idx] = 0.5f + 5.0f * ratio;
                gamma_[idx] = 0.15f + 0.35f * (1.0f - ratio);
                decay_[idx] = 0.70f + 0.29f * std::pow(ratio, 2.0f);

                psi_[idx] = Spinor2(ComplexF(dist(rng), dist(rng)), ComplexF(dist(rng), dist(rng)));
                psi_[idx].normalize();
            }
        }

        eigen_basis_.resize(n_modes_);
        for (size_t m = 0; m < n_modes_; ++m) {
            eigen_basis_[m] = std::sin(static_cast<float>(m + 1) * 3.14159265f / static_cast<float>(n_modes_));
        }

        in_weight_x_.resize(n_spinors_ * in_dim_);
        in_weight_y_.resize(n_spinors_ * in_dim_);
        in_weight_z_.resize(n_spinors_ * in_dim_);
        for (auto& w : in_weight_x_) w = dist(rng);
        for (auto& w : in_weight_y_) w = dist(rng);
        for (auto& w : in_weight_z_) w = dist(rng);

        byte_spinor_basis_.resize(256 * 64);
        for (auto& bs : byte_spinor_basis_) {
            bs = Spinor2(ComplexF(dist(rng), dist(rng)), ComplexF(dist(rng), dist(rng)));
            bs.normalize();
        }
    }

    size_t in_dim_;
    size_t n_spinors_;
    size_t n_modes_;
    size_t total_modes_;
    float global_tau_;

    std::vector<Spinor2> psi_;
    std::vector<Spinor2> pi_;
    std::vector<Spinor2> braid_history_;
    std::vector<float> gauge_link_angle_;
    std::vector<float> omega_sq_;
    std::vector<float> gamma_;
    std::vector<float> decay_;
    std::vector<float> eigen_basis_;

    std::vector<float> in_weight_x_;
    std::vector<float> in_weight_y_;
    std::vector<float> in_weight_z_;
    std::vector<Spinor2> byte_spinor_basis_;
};

// ============================================================================
// BENCHMARK HARNESS: TESTING CAUSAL CONTINUATION AND WORD INTEGRITY
// ============================================================================

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-540: NON-ABELIAN SU(2) SPINOR STRING CONTINUUM ================\n";
    std::cout << "=== Eradicating Pseudo-Syllabic Noise with Non-Commuting Braiding =====\n";
    std::cout << "=== Single-Pass Continuous Stream (N=1) ================================\n";
    std::cout << "========================================================================\n\n";

    // 128 spinors * 64 modes = 8,192 spinors = 16,384 complex components (32,768 real degrees of freedom)
    NonAbelianSpinorContinuum continuum(4, 128, 64);
    std::cout << "Spinor Continuum Active Degrees of Freedom: " << continuum.total_modes() * 4 << " real scalars\n";
    std::cout << "Lie Algebra: SU(2) Pauli Braiding with Wilson Links and Deep Attractor Wells\n\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    // Natural English Corpus with repeating structural syntax
    const std::string text_corpus = 
        "The mind is not a machine of static weights. "
        "The mind is a living stream of continuous waves. "
        "The mind perceives the world and creates meaning.";

    std::cout << ">>> Stream Ingestion of Structured Narrative (" << text_corpus.length() << " bytes, Single Pass N=1)...\n";

    size_t total_tokens = text_corpus.length() - 1;
    size_t exact_hits = 0;
    float surprisal_start = 0.0f;
    float surprisal_end = 0.0f;

    for (size_t step = 0; step < total_tokens; ++step) {
        uint8_t cur_b = static_cast<uint8_t>(text_corpus[step]);
        uint8_t nxt_b = static_cast<uint8_t>(text_corpus[step + 1]);

        float a = static_cast<float>(cur_b) / 256.0f * 6.2831853f;
        std::vector<float> s_in = {std::cos(a), std::sin(a), static_cast<float>(cur_b) / 256.0f, 1.0f};

        // Evaluation BEFORE learning
        continuum.step_spinor_flow(s_in);
        std::vector<float> probs = continuum.compute_byte_probabilities();

        auto max_it = std::max_element(probs.begin(), probs.end());
        uint8_t pred_b = static_cast<uint8_t>(std::distance(probs.begin(), max_it));
        if (pred_b == nxt_b) exact_hits++;

        float nxt_p = std::clamp(probs[nxt_b], 1e-7f, 1.0f);
        float s = -std::log(nxt_p);

        if (step == 0) surprisal_start = s;
        if (step > total_tokens - 30) surprisal_end = 0.9f * surprisal_end + 0.1f * s;

        continuum.adapt_authentic_stream(s_in, nxt_b, 0.045f);

        if (step % 25 == 0 || step == total_tokens - 1) {
            std::cout << "  [Byte " << std::setw(3) << step << " ('" << text_corpus[step] << "' -> '" << text_corpus[step+1] << "')] "
                      << "tau: " << std::fixed << std::setprecision(2) << continuum.global_tau()
                      << " | Surprisal: " << std::setprecision(3) << s << " nats"
                      << " | Top-1: '" << static_cast<char>(pred_b >= 32 && pred_b < 127 ? pred_b : '?') << "'"
                      << " | Hit: " << (pred_b == nxt_b ? "[YES]" : " no ") << "\n";
        }
    }

    float hit_rate = (static_cast<float>(exact_hits) / total_tokens) * 100.0f;
    std::cout << "\nSingle-Pass Stream Finished: Hit Rate = " << hit_rate << "% | Surprisal: " 
              << surprisal_start << " -> " << surprisal_end << " nats\n\n";

    // ------------------------------------------------------------------------
    // AUTONOMOUS CONTINUATION PROBE
    // ------------------------------------------------------------------------
    std::cout << ">>> Autonomous String Braiding Readout:\n";
    std::string prompt = "The mind is ";
    std::string generated = prompt;
    uint8_t last_b = ' ';

    // Warmup prompt
    for (char c : prompt) {
        last_b = static_cast<uint8_t>(c);
        float a = static_cast<float>(last_b) / 256.0f * 6.2831853f;
        continuum.step_spinor_flow({std::cos(a), std::sin(a), last_b / 256.0f, 1.0f});
    }

    // Direct Argmax Projection (Zero Artificial Randomness, Pure Physical Attractor)
    for (size_t s = 0; s < 45; ++s) {
        float a = static_cast<float>(last_b) / 256.0f * 6.2831853f;
        continuum.step_spinor_flow({std::cos(a), std::sin(a), last_b / 256.0f, 1.0f});

        std::vector<float> probs = continuum.compute_byte_probabilities();
        auto max_it = std::max_element(probs.begin() + 32, probs.begin() + 127);
        uint8_t best_b = static_cast<uint8_t>(std::distance(probs.begin(), max_it));

        generated += static_cast<char>(best_b);
        last_b = best_b;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n========================================================================\n";
    std::cout << "=== EXP-540 TELEMETRY RESULTS ===\n";
    std::cout << "Total Wall Clock: " << std::fixed << std::setprecision(2) << total_time << " s\n";
    std::cout << "Surprisal Drop: " << surprisal_start << " -> " << surprisal_end << " nats\n";
    std::cout << "Hit Rate on Unseen Byte Stream: " << hit_rate << "%\n";
    std::cout << "Generated Attractor Readout: \"" << generated << "\"\n";
    std::cout << "========================================================================\n";

    return 0;
}
