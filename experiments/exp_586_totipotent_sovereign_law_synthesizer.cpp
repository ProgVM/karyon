// ============================================================================
// EXP-586: TOTIPOTENT SOVEREIGN LAW SYNTHESIZER (TSLS)
// Standard: KEP v30.0 (Principles 2, 3, 4, 5, 6, 8, 27, 29, 30)
//
// Author / Concept: Bazilevs & Karyon Lead Cyberneticist
// Hypothesis:
// Fixed human-devised equations (pure SDE, fixed Hopfield, hardcoded PAC)
// inherently limit convergence to ~3.5 - 4.0 nats because human designers
// cannot foresee the exact non-linear manifold of natural language.
//
// By applying KEP Principle 4 (Self-Evolution of Formulas and Dynamic Laws),
// Principle 5 (Causal Credit Assignment - Death of Blind Dice-Throwing),
// Principle 8 (Axiom of Unshackled Flow - Dimensional Budding), and
// Principle 30 (Law of Totipotent Self-Synthesis):
//
// In EXP-586, we construct an unconstrained Totipotent Continuum where:
// 1. Dynamic Law Representation via Generalized Continuous Volterra-Taylor Field:
//    dz_i / dt = sum_k c_k * Phi_k(z, x)
//    where Phi_k are freely synthesized basis terms:
//    - Linear/harmonic drift: Phi(z) = W * z
//    - Higher-order polynomial/bilinear cross-couplings: Phi(z) = (U * z) * (V * z)
//    - Fractional / geometric metric curvatures: Phi(z) = |z|^gamma * sign(z)
//    - Rotational gauge fluxes: Phi(z) = J * dz
//    - Bottom-up unconstrained sensory coupling fields.
// 2. Open-Ended Term Genesis & Pruning:
//    The organism dynamically sprouts new dynamic terms when Free Energy
//    exceeds threshold, and prunes or locks them epigenetically based on
//    continuous causal trace:
//    dCausal_k / dt = -lambda * Causal_k + (-dFE/dt) * Contribution_k
// 3. Dimensional Budding:
//    When phase trajectories collide (high local surprise), the state manifold
//    sprouts new orthogonal dimensions (z_{N+1}) via Net2Net smooth birth,
//    giving the system the exact degrees of freedom needed to collapse Free Energy to zero.
// 4. Symbolic Endoscopy:
//    Prints human-readable expressions of the mathematical laws synthesized
//    by the organism during learning.
// ============================================================================

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <immintrin.h>
#include <numbers>
#include <omp.h>
#include <map>
#include <memory>
#include <string>

#include "karyon_core.h"

namespace karyon {

// Types of dynamic field operators that can be synthesized
enum class DynamicTermType {
    LINEAR_DRIFT,           // sum_j W_ij * z_j
    ROTATIONAL_GAUGE,       // sum_j Omega_ij * dz_j
    BILINEAR_COUPLING,      // (sum_j U_rj * z_j) * (sum_k V_rk * z_k) projected via W
    POWER_METRIC,           // |z_i|^gamma * sign(z_i)
    EXPONENTIAL_ATTRACTOR,  // D_ij * tanh(beta * z_j)
    SENSORY_SHOCK           // Inflow from sensory byte
};

std::string term_type_to_string(DynamicTermType t) {
    switch (t) {
        case DynamicTermType::LINEAR_DRIFT: return "LinearDrift[W*z]";
        case DynamicTermType::ROTATIONAL_GAUGE: return "RotationalGauge[Omega*dz]";
        case DynamicTermType::BILINEAR_COUPLING: return "Bilinear[(Uz)*(Vz)]";
        case DynamicTermType::POWER_METRIC: return "PowerMetric[|z|^gamma]";
        case DynamicTermType::EXPONENTIAL_ATTRACTOR: return "Attractor[D*tanh(bz)]";
        case DynamicTermType::SENSORY_SHOCK: return "SensoryShock[In(x)]";
    }
    return "Unknown";
}

struct SynthesizedTerm {
    DynamicTermType type;
    std::string formula_str;
    float coefficient = 1.0f;
    float causal_credit = 0.0f;
    float epigenetic_lock = 0.0f; // 0.0 = completely plastic, 1.0 = immutable invariant
    std::vector<float> params;    // Operator-specific continuous parameters
    float gamma_power = 1.0f;
    float beta_scale = 1.0f;
};

struct SpeechMetrics {
    double char_entropy = 0.0;
    double whitespace_ratio = 0.0;
    double alphabet_ratio = 0.0;
    size_t unique_chars = 0;
};

SpeechMetrics evaluate_speech(const std::string& text) {
    SpeechMetrics m;
    if (text.empty()) return m;
    std::map<char, size_t> counts;
    size_t whitespaces = 0;
    size_t alphabets = 0;

    for (char ch : text) {
        counts[ch]++;
        if (std::isspace(static_cast<unsigned char>(ch))) whitespaces++;
        if (std::isalpha(static_cast<unsigned char>(ch))) alphabets++;
    }

    m.unique_chars = counts.size();
    m.whitespace_ratio = static_cast<double>(whitespaces) / text.size();
    m.alphabet_ratio = static_cast<double>(alphabets) / text.size();

    double ent = 0.0;
    for (const auto& [ch, count] : counts) {
        double p = static_cast<double>(count) / text.size();
        if (p > 0.0) ent -= p * std::log2(p);
    }
    m.char_entropy = ent;
    return m;
}

class TotipotentLawSynthesizerOrganism {
public:
    TotipotentLawSynthesizerOrganism(size_t initial_dim = 128) noexcept 
        : current_dim_(initial_dim), max_dim_(512) {
        total_steps_ = 0;
        free_energy_ = 1.0f;
        prev_free_energy_ = 1.0f;
        dt_eff_ = 0.040f;
        endogenous_beta_ = 4.0f;

        z_.assign(current_dim_, 0.0f);
        dz_.assign(current_dim_, 0.0f);
        last_term_contributions_.assign(current_dim_, 0.0f);

        // Motor readout
        motor_readout_.assign(CHANNEL_IO_DIM * current_dim_, 0.0f);
        std::mt19937 rng(5860);
        std::normal_distribution<float> init_dist(0.0f, 0.02f);
        for (auto& w : motor_readout_) w = init_dist(rng);

        anchor_momentum_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));

        // Initialize with a minimal primordial law soup (vacuum state)
        // 1. Sensory Shock
        synthesize_term(DynamicTermType::SENSORY_SHOCK, 1.0f);
        // 2. Linear Drift
        synthesize_term(DynamicTermType::LINEAR_DRIFT, 0.5f);
        // 3. Attractor
        synthesize_term(DynamicTermType::EXPONENTIAL_ATTRACTOR, 0.5f);
    }

    void synthesize_term(DynamicTermType type, float init_coeff = 0.1f) {
        SynthesizedTerm term;
        term.type = type;
        term.coefficient = init_coeff;
        term.causal_credit = 0.0f;
        term.epigenetic_lock = 0.0f;

        std::mt19937 rng(5860 + static_cast<uint32_t>(terms_.size()));
        std::normal_distribution<float> init_dist(0.0f, 0.02f);

        if (type == DynamicTermType::SENSORY_SHOCK) {
            term.params.assign(CHANNEL_IO_DIM * current_dim_, 0.0f);
            for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                float phase = 2.0f * std::numbers::pi_v<float> * static_cast<float>(c + 1) / static_cast<float>(CHANNEL_IO_DIM);
                for (size_t i = 0; i < current_dim_; ++i) {
                    term.params[c * current_dim_ + i] = std::sin(phase * static_cast<float>(i + 1)) * 0.15f + init_dist(rng);
                }
            }
            term.formula_str = "Phi_sensory(x) = Inflow_{x, i}";
        } else if (type == DynamicTermType::LINEAR_DRIFT) {
            term.params.assign(current_dim_ * current_dim_, 0.0f);
            for (auto& w : term.params) w = init_dist(rng);
            term.formula_str = "Phi_linear(z) = sum_j W_{ij} * z_j";
        } else if (type == DynamicTermType::ROTATIONAL_GAUGE) {
            term.params.assign(current_dim_ * current_dim_, 0.0f);
            for (size_t i = 0; i < current_dim_; ++i) {
                for (size_t j = i + 1; j < current_dim_; ++j) {
                    float val = init_dist(rng);
                    term.params[i * current_dim_ + j] = val;
                    term.params[j * current_dim_ + i] = -val; // Anti-symmetric
                }
            }
            term.formula_str = "Phi_gauge(dz) = sum_j Omega_{ij} * dz_j [Omega = -Omega^T]";
        } else if (type == DynamicTermType::EXPONENTIAL_ATTRACTOR) {
            term.params.assign(current_dim_ * current_dim_, 0.0f);
            for (size_t i = 0; i < current_dim_; ++i) {
                for (size_t j = i; j < current_dim_; ++j) {
                    float val = init_dist(rng);
                    term.params[i * current_dim_ + j] = val;
                    term.params[j * current_dim_ + i] = val; // Symmetric Lyapunov
                }
            }
            term.beta_scale = 2.0f;
            term.formula_str = "Phi_attractor(z) = sum_j D_{ij} * tanh(beta * z_j) [D = D^T]";
        } else if (type == DynamicTermType::BILINEAR_COUPLING) {
            // Rank 16 bilinear form: (U*z) * (V*z) projected via W
            size_t rank = 16;
            term.params.assign(2 * rank * current_dim_ + current_dim_ * rank, 0.0f);
            for (auto& w : term.params) w = init_dist(rng);
            term.formula_str = "Phi_bilinear(z) = sum_r W_{ir} * (U_r * z) * sigma(V_r * z)";
        } else if (type == DynamicTermType::POWER_METRIC) {
            term.gamma_power = 1.5f + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            term.params.assign(current_dim_, 1.0f);
            term.formula_str = "Phi_metric(z) = lambda_i * |z_i|^" + std::to_string(term.gamma_power) + " * sgn(z_i)";
        }

        terms_.push_back(std::move(term));
    }

    // Principle 8: Autonomous Dimensional Budding (Net2Net Smooth Sprouting)
    void bud_dimension() {
        if (current_dim_ >= max_dim_) return;
        size_t old_dim = current_dim_;
        size_t new_dim = old_dim + 32; // Sprout 32 new orthogonal dimensions

        std::cout << "\n[!] AUTOPOIETIC MORPHOGENESIS: Sprouting state dimensions: " 
                  << old_dim << " -> " << new_dim << " (Free Energy: " << free_energy_ << ")\n";

        z_.resize(new_dim, 0.0f);
        dz_.resize(new_dim, 0.0f);
        last_term_contributions_.resize(new_dim, 0.0f);

        // Expand motor readout with zero-padding (Net2Net identity preserving)
        std::vector<float> new_motor(CHANNEL_IO_DIM * new_dim, 0.0f);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            for (size_t i = 0; i < old_dim; ++i) {
                new_motor[c * new_dim + i] = motor_readout_[c * old_dim + i];
            }
        }
        motor_readout_ = std::move(new_motor);

        // Expand parameters of all synthesized terms
        for (auto& term : terms_) {
            if (term.type == DynamicTermType::SENSORY_SHOCK) {
                std::vector<float> new_p(CHANNEL_IO_DIM * new_dim, 0.0f);
                for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
                    for (size_t i = 0; i < old_dim; ++i) {
                        new_p[c * new_dim + i] = term.params[c * old_dim + i];
                    }
                }
                term.params = std::move(new_p);
            } else if (term.type == DynamicTermType::LINEAR_DRIFT || 
                       term.type == DynamicTermType::ROTATIONAL_GAUGE ||
                       term.type == DynamicTermType::EXPONENTIAL_ATTRACTOR) {
                std::vector<float> new_p(new_dim * new_dim, 0.0f);
                for (size_t i = 0; i < old_dim; ++i) {
                    for (size_t j = 0; j < old_dim; ++j) {
                        new_p[i * new_dim + j] = term.params[i * old_dim + j];
                    }
                }
                term.params = std::move(new_p);
            } else if (term.type == DynamicTermType::POWER_METRIC) {
                term.params.resize(new_dim, 1.0f);
            } else if (term.type == DynamicTermType::BILINEAR_COUPLING) {
                size_t rank = 16;
                term.params.assign(2 * rank * new_dim + new_dim * rank, 0.0f);
            }
        }

        current_dim_ = new_dim;
    }

    size_t step(uint8_t input_byte) noexcept {
        const size_t in_offset = static_cast<size_t>(input_byte) * current_dim_;

        // Compute total state norm for Oja regulation
        float z_sq = 0.0f;
        for (size_t i = 0; i < current_dim_; ++i) z_sq += z_[i] * z_[i];
        float mean_z_sq = z_sq / static_cast<float>(current_dim_);

        // Dynamic Time Dilation
        float total_kinetic = 0.0f;
        for (size_t i = 0; i < current_dim_; ++i) total_kinetic += dz_[i] * dz_[i];
        float mean_kinetic = total_kinetic / static_cast<float>(current_dim_);
        dt_eff_ = std::clamp(0.040f / (1.0f + 2.0f * mean_kinetic), 0.010f, 0.080f);

        // Compute total vector field acceleration from all synthesized laws:
        // dz_total = sum_k coeff_k * Phi_k(z, dz, x)
        std::vector<float> total_accel(current_dim_, 0.0f);
        std::vector<std::vector<float>> term_evals(terms_.size(), std::vector<float>(current_dim_, 0.0f));

        for (size_t k = 0; k < terms_.size(); ++k) {
            const auto& term = terms_[k];
            if (std::abs(term.coefficient) < 1e-5f) continue;

            auto& eval = term_evals[k];

            if (term.type == DynamicTermType::SENSORY_SHOCK) {
                for (size_t i = 0; i < current_dim_; ++i) {
                    eval[i] = term.params[in_offset + i];
                }
            } else if (term.type == DynamicTermType::LINEAR_DRIFT) {
                for (size_t i = 0; i < current_dim_; ++i) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < current_dim_; ++j) {
                        sum += term.params[i * current_dim_ + j] * z_[j];
                    }
                    eval[i] = sum;
                }
            } else if (term.type == DynamicTermType::ROTATIONAL_GAUGE) {
                for (size_t i = 0; i < current_dim_; ++i) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < current_dim_; ++j) {
                        sum += term.params[i * current_dim_ + j] * dz_[j];
                    }
                    eval[i] = sum;
                }
            } else if (term.type == DynamicTermType::EXPONENTIAL_ATTRACTOR) {
                std::vector<float> tz(current_dim_);
                for (size_t j = 0; j < current_dim_; ++j) tz[j] = std::tanh(term.beta_scale * z_[j]);
                for (size_t i = 0; i < current_dim_; ++i) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < current_dim_; ++j) {
                        sum += term.params[i * current_dim_ + j] * tz[j];
                    }
                    eval[i] = sum;
                }
            } else if (term.type == DynamicTermType::POWER_METRIC) {
                for (size_t i = 0; i < current_dim_; ++i) {
                    float val = z_[i];
                    float sgn = (val >= 0.0f) ? 1.0f : -1.0f;
                    eval[i] = -term.params[i] * std::pow(std::abs(val), term.gamma_power) * sgn;
                }
            } else if (term.type == DynamicTermType::BILINEAR_COUPLING) {
                size_t rank = 16;
                std::vector<float> u(rank, 0.0f), v(rank, 0.0f);
                const float* U_ptr = term.params.data();
                const float* V_ptr = term.params.data() + rank * current_dim_;
                const float* W_ptr = term.params.data() + 2 * rank * current_dim_;

                for (size_t r = 0; r < rank; ++r) {
                    for (size_t j = 0; j < current_dim_; ++j) {
                        u[r] += U_ptr[r * current_dim_ + j] * z_[j];
                        v[r] += V_ptr[r * current_dim_ + j] * z_[j];
                    }
                    v[r] = 1.0f / (1.0f + std::exp(-v[r])); // Sigmoid gate
                }
                for (size_t i = 0; i < current_dim_; ++i) {
                    float sum = 0.0f;
                    for (size_t r = 0; r < rank; ++r) {
                        sum += W_ptr[i * rank + r] * (u[r] * v[r]);
                    }
                    eval[i] = sum;
                }
            }

            // Accumulate into total acceleration with term coefficient
            for (size_t i = 0; i < current_dim_; ++i) {
                total_accel[i] += term.coefficient * eval[i];
            }
        }

        // Integrate State Dynamics: dz/dt = total_accel, z_{t+1} = z_t + dz * dt
        for (size_t i = 0; i < current_dim_; ++i) {
            dz_[i] = (dz_[i] + total_accel[i] * dt_eff_) * 0.985f;
            z_[i] = std::tanh(z_[i] + dz_[i] * dt_eff_);
        }

        // Volitional Motor Resonance Readout
        std::array<float, CHANNEL_IO_DIM> resonance{};
        float max_res = -1e9f;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float dot = 0.0f;
            const float* w_ptr = &motor_readout_[c * current_dim_];
            for (size_t i = 0; i < current_dim_; ++i) dot += w_ptr[i] * z_[i];

            float norm_dot = dot / (std::sqrt(static_cast<float>(current_dim_)) + SAFETY_EPSILON);
            float eff_res = norm_dot - anchor_momentum_[c];
            resonance[c] = eff_res;
            if (eff_res > max_res) max_res = eff_res;
        }

        // Endogenous Inverse Temperature
        float variance = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float diff = resonance[c] - max_res;
            variance += diff * diff;
        }
        variance /= static_cast<float>(CHANNEL_IO_DIM);
        endogenous_beta_ = std::clamp(1.5f + 2.5f * std::sqrt(variance), 1.0f, 16.0f);

        // Softmax
        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float p = std::exp((resonance[c] - max_res) * endogenous_beta_);
            latest_probabilities_[c] = p;
            sum_exp += p;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probabilities_[c] *= inv_sum;

        // Volitional selection
        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float cum = 0.0f;
        size_t chosen = 0;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            cum += latest_probabilities_[c];
            if (cum >= r) { chosen = c; break; }
        }

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) anchor_momentum_[c] *= 0.72f;
        anchor_momentum_[chosen] += 0.85f * (1.0f + mean_kinetic);

        return chosen;
    }

    void adapt(uint8_t input_byte, uint8_t target_byte, float lr = 0.04f) noexcept {
        total_steps_++;
        float p_target = latest_probabilities_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        prev_free_energy_ = free_energy_;
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;

        float delta_FE = free_energy_ - prev_free_energy_;
        float causal_boost = (-delta_FE) * 25.0f; // Positive when FE decreases!

        // Compute mean z norm for Oja dissipation
        float z_sq = 0.0f;
        for (size_t i = 0; i < current_dim_; ++i) z_sq += z_[i] * z_[i];
        float mean_z_sq = z_sq / static_cast<float>(current_dim_);

        // 1. Motor Readout Oja Plasticity
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probabilities_[c];
            if (std::abs(err) > 0.0005f) {
                float alpha = lr * err;
                float oja_motor = 0.02f * latest_probabilities_[c] * mean_z_sq;
                for (size_t i = 0; i < current_dim_; ++i) {
                    size_t m_idx = c * current_dim_ + i;
                    motor_readout_[m_idx] += alpha * z_[i] - oja_motor * motor_readout_[m_idx];
                }
            }
        }

        // 2. PRINCIPLE 5: Causal Credit Assignment & Dynamic Law Plasticity
        for (size_t k = 0; k < terms_.size(); ++k) {
            auto& term = terms_[k];
            // Track causal trace of the formula
            term.causal_credit = 0.98f * term.causal_credit + 0.02f * causal_boost;

            // Epigenetic protection: If a law proves persistently useful (credit > 0), lock it!
            if (term.causal_credit > 0.1f) {
                term.epigenetic_lock = std::clamp(term.epigenetic_lock + 0.001f, 0.0f, 0.95f);
            } else if (term.causal_credit < -0.1f) {
                term.epigenetic_lock = std::clamp(term.epigenetic_lock - 0.002f, 0.0f, 0.95f);
            }

            // Plasticity of term coefficient governed by causal credit and epigenetic lock
            float plastic_rate = (1.0f - term.epigenetic_lock);
            term.coefficient += lr * 0.05f * plastic_rate * term.causal_credit;
            term.coefficient = std::clamp(term.coefficient, -3.0f, 3.0f);

            // Adapt term parameters under Oja dissipation
            if (term.type == DynamicTermType::SENSORY_SHOCK) {
                const size_t in_offset = static_cast<size_t>(input_byte) * current_dim_;
                for (size_t i = 0; i < current_dim_; ++i) {
                    size_t s_idx = in_offset + i;
                    float oja_decay = 0.02f * mean_z_sq * term.params[s_idx];
                    term.params[s_idx] += lr * 0.05f * plastic_rate * (causal_boost * z_[i] - oja_decay);
                }
            } else if (term.type == DynamicTermType::LINEAR_DRIFT) {
                for (size_t i = 0; i < current_dim_; ++i) {
                    for (size_t j = 0; j < current_dim_; ++j) {
                        size_t idx = i * current_dim_ + j;
                        float oja_decay = mean_z_sq * term.params[idx];
                        term.params[idx] += lr * 0.05f * plastic_rate * (causal_boost * dz_[i] * z_[j] - 0.02f * oja_decay);
                    }
                }
            } else if (term.type == DynamicTermType::EXPONENTIAL_ATTRACTOR) {
                for (size_t i = 0; i < current_dim_; ++i) {
                    for (size_t j = 0; j < current_dim_; ++j) {
                        size_t idx = i * current_dim_ + j;
                        float oja_decay = mean_z_sq * term.params[idx];
                        term.params[idx] += lr * 0.05f * plastic_rate * (causal_boost * z_[i] * std::tanh(z_[j]) - 0.02f * oja_decay);
                    }
                }
            }
        }

        // 3. AUTOPOIETIC GENESIS: Sprouting new mathematical laws or dimensions if trapped in high error!
        if (total_steps_ % 25000 == 0) {
            evaluate_and_evolve_laws();
        }
    }

    void evaluate_and_evolve_laws() {
        std::cout << "\n--- [AUTOPOIETIC AUDIT @ Step " << total_steps_ << "] ---\n";
        std::cout << "  Current Free Energy: " << free_energy_ << " | Active Laws: " << terms_.size() << " | Dim: " << current_dim_ << "\n";

        // Print active synthesized laws
        for (size_t k = 0; k < terms_.size(); ++k) {
            const auto& t = terms_[k];
            std::cout << "  Law [" << k << "] " << term_type_to_string(t.type)
                      << " | Coeff: " << t.coefficient
                      << " | Credit: " << t.causal_credit
                      << " | Lock: " << (t.epigenetic_lock * 100.0f) << "%\n"
                      << "     Formula: " << t.formula_str << "\n";
        }

        // Prune harmful/parasitic laws (Principle 5: Understand why deleting was good or bad)
        for (auto it = terms_.begin(); it != terms_.end(); ) {
            if (it->causal_credit < -0.5f && it->epigenetic_lock < 0.10f && terms_.size() > 2) {
                std::cout << "  [-] AUTO-PRUNING HARMFUL LAW: " << term_type_to_string(it->type) 
                          << " (Causal credit: " << it->causal_credit << ")\n";
                it = terms_.erase(it);
            } else {
                ++it;
            }
        }

        // If Free Energy is plateaued high (> 3.5 nats), synthesize NEW dynamic forms!
        if (free_energy_ > 3.5f) {
            // Check which law types are missing
            bool has_gauge = false;
            bool has_bilinear = false;
            bool has_metric = false;

            for (const auto& t : terms_) {
                if (t.type == DynamicTermType::ROTATIONAL_GAUGE) has_gauge = true;
                if (t.type == DynamicTermType::BILINEAR_COUPLING) has_bilinear = true;
                if (t.type == DynamicTermType::POWER_METRIC) has_metric = true;
            }

            if (!has_gauge) {
                std::cout << "  [+] AUTO-SYNTHESIZING: Rotational Gauge Field Omega_ij * dz_j\n";
                synthesize_term(DynamicTermType::ROTATIONAL_GAUGE, 0.25f);
            } else if (!has_bilinear) {
                std::cout << "  [+] AUTO-SYNTHESIZING: Bilinear Cross-Coupling Field (U*z)*(V*z)\n";
                synthesize_term(DynamicTermType::BILINEAR_COUPLING, 0.20f);
            } else if (!has_metric) {
                std::cout << "  [+] AUTO-SYNTHESIZING: Non-Euclidean Power Metric |z|^gamma * sgn(z)\n";
                synthesize_term(DynamicTermType::POWER_METRIC, 0.15f);
            } else if (current_dim_ < max_dim_) {
                // If all laws exist and FE is still high, the state space is cramped! Sprout dimension!
                bud_dimension();
            }
        }
    }

    float get_free_energy() const noexcept { return free_energy_; }
    float get_beta() const noexcept { return endogenous_beta_; }
    size_t get_active_terms() const noexcept { return terms_.size(); }
    size_t get_dim() const noexcept { return current_dim_; }

private:
    size_t current_dim_;
    size_t max_dim_;
    size_t total_steps_;
    float free_energy_;
    float prev_free_energy_;
    float dt_eff_;
    float endogenous_beta_;

    std::vector<float> z_;
    std::vector<float> dz_;
    std::vector<float> last_term_contributions_;

    std::vector<SynthesizedTerm> terms_;
    std::vector<float> motor_readout_;

    std::array<float, CHANNEL_IO_DIM> anchor_momentum_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-586: TOTIPOTENT SOVEREIGN LAW SYNTHESIZER (TSLS)             ===\n";
    std::cout << "=== Autopoietic Mathematical Morphogenesis & Open-Ended Genesis      ===\n";
    std::cout << "========================================================================\n";

    std::string text_path = "data/alpaca_data.json";
    std::ifstream file(text_path, std::ios::binary);
    if (!file.is_open()) {
        text_path = "data/web_downloads/alpaca_data.json";
        file.open(text_path, std::ios::binary);
    }

    std::vector<uint8_t> corpus;
    if (file.is_open()) {
        corpus = std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        std::cout << "[+] Corpus loaded: " << corpus.size() << " bytes.\n\n";
    } else {
        std::string fallback = "The capital of France is Paris. Language crystallizes as mathematical laws synthesize themselves bottom-up.";
        corpus = std::vector<uint8_t>(fallback.begin(), fallback.end());
    }

    const size_t train_steps = 300000;

    std::cout << ">>> Initializing Totipotent Law Synthesizer Organism (Initial Dim: 128)...\n";
    karyon::TotipotentLawSynthesizerOrganism tsls_org(128);

    std::cout << ">>> Launching Continuous Autopoietic Synthesis (" << train_steps << " steps)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        tsls_org.step(in_b);
        tsls_org.adapt(in_b, tgt_b, 0.04f);

        if ((step + 1) % 75000 == 0) {
            std::cout << "  >>> Step " << (step + 1)
                      << " | Free Energy: " << tsls_org.get_free_energy()
                      << " | Beta: " << tsls_org.get_beta()
                      << " | Dim: " << tsls_org.get_dim()
                      << " | Active Laws: " << tsls_org.get_active_terms() << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(t1 - t0).count();
    double throughput = static_cast<double>(train_steps) / duration;

    // Directed Speech Generation Test
    std::string prompt = "The capital of France is ";
    std::string speech_output = prompt;
    uint8_t curr = static_cast<uint8_t>(prompt.back());
    for (int i = 0; i < 200; ++i) {
        size_t next_act = tsls_org.step(curr);
        speech_output += static_cast<char>(next_act);
        curr = static_cast<uint8_t>(next_act);
    }
    auto metrics = karyon::evaluate_speech(speech_output);

    std::cout << "\n========================================================================\n";
    std::cout << "=== COMPARATIVE TELEMETRY (EXP-586 vs EXP-584 vs EXP-583)             ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Substrate Architecture          | Throughput (tok/s) | Final FE (nats) | Active Laws | State Dim | Alphabet %\n";
    std::cout << "--------------------------------|--------------------|-----------------|-------------|-----------|-----------\n";
    std::cout << "TSLS Law Synthesizer (EXP-586)  | " << throughput << "          | " << tsls_org.get_free_energy() << "          | " << tsls_org.get_active_terms() << "           | " << tsls_org.get_dim() << "       | " << (metrics.alphabet_ratio * 100.0) << "%\n";
    std::cout << "BR-HSC 3-Gateway Oja (EXP-584)  | 3251.74            | 4.0635          | Fixed (3)   | 256       | 64.44%\n";
    std::cout << "RH-SAC Unbounded (EXP-583)      | 3241.72            | 12.1355         | Fixed (3)   | 256       | 27.11%\n";
    std::cout << "========================================================================\n\n";

    std::cout << ">>> AUTONOMOUSLY SYNTHESIZED SPEECH (200 bytes):\n";
    std::cout << "\"" << speech_output << "\"\n\n";

    return 0;
}
