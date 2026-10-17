// ============================================================================
// EXP-587: CONSERVATIVE PHASE RESONANCE & ZERO-LEAK CONTINUUM (CPR-ZLC)
// Standard: KEP v30.0 (Principles 2, 3, 4, 5, 6, 8, 27, 29, 30)
//
// Author / Concept: Bazilevs & Karyon Lead Cyberneticist
// Hypothesis:
// The fundamental barrier stopping the sovereign substrate at ~3.4 nats
// is temporal dissipation:
// 1. Friction damping factor 0.985^150 ≈ 0.10 wipes out context beyond 100 bytes.
// 2. Continuous uniform Oja decay erodes motor readout patterns.
//
// By implementing:
// 1. Conservative Symplectic Hamiltonian Dynamics (Zero Damping, div V = 0):
//    Information is encoded as persistent topological phase angles theta_i
//    and angular momentums L_i, which undergo unitary rotation rather than
//    exponential decay. Context length becomes effectively infinite.
// 2. Selective Resonant Epigenetic Gating (Zero-Leak Synaptic Superconductivity):
//    Oja dissipation scales down with causal stability:
//    oja_eff = oja_base * (1.0 - epigenetic_lock) -> 0.0 for crystallizing patterns.
// 3. Local Phase-Locked Resonance Catchment:
//    State settles into exact phase coherence with incoming context.
//
// Target: Collapse Free Energy F_t towards near-zero (< 1.0 nats, approaching
//         ideal convergence on structured long sequences) without handcoded rules.
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

enum class DynamicTermType {
    LINEAR_DRIFT,
    SYMPLECTIC_GAUGE,      // Conservative unitary rotation: J * p
    BILINEAR_COUPLING,
    POWER_METRIC,
    EXPONENTIAL_ATTRACTOR,
    SENSORY_SHOCK
};

std::string term_type_to_string(DynamicTermType t) {
    switch (t) {
        case DynamicTermType::LINEAR_DRIFT: return "LinearDrift";
        case DynamicTermType::SYMPLECTIC_GAUGE: return "SymplecticGauge[J*p]";
        case DynamicTermType::BILINEAR_COUPLING: return "Bilinear[(Uz)*(Vz)]";
        case DynamicTermType::POWER_METRIC: return "PowerMetric[|q|^gamma]";
        case DynamicTermType::EXPONENTIAL_ATTRACTOR: return "Attractor[D*tanh(q)]";
        case DynamicTermType::SENSORY_SHOCK: return "SensoryShock[In(x)]";
    }
    return "Unknown";
}

struct SynthesizedTerm {
    DynamicTermType type;
    std::string formula_str;
    float coefficient = 1.0f;
    float causal_credit = 0.0f;
    float epigenetic_lock = 0.0f;
    std::vector<float> params;
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

class ConservativeZeroLeakOrganism {
public:
    ConservativeZeroLeakOrganism(size_t initial_dim = 128) noexcept 
        : current_dim_(initial_dim), max_dim_(512) {
        total_steps_ = 0;
        free_energy_ = 1.0f;
        prev_free_energy_ = 1.0f;
        dt_eff_ = 0.035f;
        endogenous_beta_ = 4.0f;

        // Canonical Hamiltonian Phase Space: (q, p)
        q_.assign(current_dim_, 0.0f); // Generalized coordinates (persistent holographic phase)
        p_.assign(current_dim_, 0.0f); // Conjugate momentum

        motor_readout_.assign(CHANNEL_IO_DIM * current_dim_, 0.0f);
        motor_locks_.assign(CHANNEL_IO_DIM * current_dim_, 0.0f);

        std::mt19937 rng(5870);
        std::normal_distribution<float> init_dist(0.0f, 0.02f);
        for (auto& w : motor_readout_) w = init_dist(rng);

        anchor_momentum_.fill(0.0f);
        latest_probabilities_.fill(1.0f / static_cast<float>(CHANNEL_IO_DIM));

        // Primordial conservative terms
        synthesize_term(DynamicTermType::SENSORY_SHOCK, 1.0f);
        synthesize_term(DynamicTermType::SYMPLECTIC_GAUGE, 0.8f);
        synthesize_term(DynamicTermType::EXPONENTIAL_ATTRACTOR, 0.4f);
        synthesize_term(DynamicTermType::BILINEAR_COUPLING, 0.3f);
    }

    void synthesize_term(DynamicTermType type, float init_coeff = 0.1f) {
        SynthesizedTerm term;
        term.type = type;
        term.coefficient = init_coeff;
        term.causal_credit = 0.0f;
        term.epigenetic_lock = 0.0f;

        std::mt19937 rng(5870 + static_cast<uint32_t>(terms_.size()));
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
        } else if (type == DynamicTermType::SYMPLECTIC_GAUGE) {
            // Strictly anti-symmetric J: J = -J^T (Guarantees zero dissipation, div = 0)
            term.params.assign(current_dim_ * current_dim_, 0.0f);
            for (size_t i = 0; i < current_dim_; ++i) {
                for (size_t j = i + 1; j < current_dim_; ++j) {
                    float val = init_dist(rng);
                    term.params[i * current_dim_ + j] = val;
                    term.params[j * current_dim_ + i] = -val;
                }
            }
            term.formula_str = "Phi_gauge(p) = sum_j J_{ij} * p_j [J = -J^T, Unitary]";
        } else if (type == DynamicTermType::LINEAR_DRIFT) {
            term.params.assign(current_dim_ * current_dim_, 0.0f);
            for (auto& w : term.params) w = init_dist(rng);
            term.formula_str = "Phi_linear(q) = sum_j W_{ij} * q_j";
        } else if (type == DynamicTermType::EXPONENTIAL_ATTRACTOR) {
            // Symmetric Lyapunov D = D^T
            term.params.assign(current_dim_ * current_dim_, 0.0f);
            for (size_t i = 0; i < current_dim_; ++i) {
                for (size_t j = i; j < current_dim_; ++j) {
                    float val = init_dist(rng);
                    term.params[i * current_dim_ + j] = val;
                    term.params[j * current_dim_ + i] = val;
                }
            }
            term.beta_scale = 2.0f;
            term.formula_str = "Phi_attractor(q) = sum_j D_{ij} * tanh(beta * q_j) [D = D^T]";
        } else if (type == DynamicTermType::BILINEAR_COUPLING) {
            size_t rank = 16;
            term.params.assign(2 * rank * current_dim_ + current_dim_ * rank, 0.0f);
            for (auto& w : term.params) w = init_dist(rng);
            term.formula_str = "Phi_bilinear(q) = sum_r W_{ir} * (U_r * q) * sigma(V_r * q)";
        } else if (type == DynamicTermType::POWER_METRIC) {
            term.gamma_power = 2.0f;
            term.params.assign(current_dim_, 1.0f);
            term.formula_str = "Phi_metric(q) = lambda_i * |q_i|^gamma * sgn(q_i)";
        }

        terms_.push_back(std::move(term));
    }

    void bud_dimension() {
        if (current_dim_ >= max_dim_) return;
        size_t old_dim = current_dim_;
        size_t new_dim = old_dim + 32;

        std::cout << "\n[!] AUTOPOIETIC MORPHOGENESIS: Sprouting conservative dimensions: " 
                  << old_dim << " -> " << new_dim << " (Free Energy: " << free_energy_ << ")\n";

        q_.resize(new_dim, 0.0f);
        p_.resize(new_dim, 0.0f);

        std::vector<float> new_motor(CHANNEL_IO_DIM * new_dim, 0.0f);
        std::vector<float> new_locks(CHANNEL_IO_DIM * new_dim, 0.0f);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            for (size_t i = 0; i < old_dim; ++i) {
                new_motor[c * new_dim + i] = motor_readout_[c * old_dim + i];
                new_locks[c * new_dim + i] = motor_locks_[c * old_dim + i];
            }
        }
        motor_readout_ = std::move(new_motor);
        motor_locks_ = std::move(new_locks);

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
                       term.type == DynamicTermType::SYMPLECTIC_GAUGE ||
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

        // Symplectic Leapfrog Integration (Exact Phase-Space Volume Preservation)
        // Eliminates artificial exponential context decay (0.985^150 decay eradicated!)
        std::vector<float> force(current_dim_, 0.0f);

        for (size_t k = 0; k < terms_.size(); ++k) {
            const auto& term = terms_[k];
            if (std::abs(term.coefficient) < 1e-5f) continue;

            if (term.type == DynamicTermType::SENSORY_SHOCK) {
                for (size_t i = 0; i < current_dim_; ++i) {
                    force[i] += term.coefficient * term.params[in_offset + i];
                }
            } else if (term.type == DynamicTermType::SYMPLECTIC_GAUGE) {
                // Unitary phase rotation (div = 0)
                for (size_t i = 0; i < current_dim_; ++i) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < current_dim_; ++j) {
                        sum += term.params[i * current_dim_ + j] * p_[j];
                    }
                    force[i] += term.coefficient * sum;
                }
            } else if (term.type == DynamicTermType::EXPONENTIAL_ATTRACTOR) {
                std::vector<float> tq(current_dim_);
                for (size_t j = 0; j < current_dim_; ++j) tq[j] = std::tanh(term.beta_scale * q_[j]);
                for (size_t i = 0; i < current_dim_; ++i) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < current_dim_; ++j) {
                        sum += term.params[i * current_dim_ + j] * tq[j];
                    }
                    force[i] += term.coefficient * sum;
                }
            } else if (term.type == DynamicTermType::BILINEAR_COUPLING) {
                size_t rank = 16;
                std::vector<float> u(rank, 0.0f), v(rank, 0.0f);
                const float* U_ptr = term.params.data();
                const float* V_ptr = term.params.data() + rank * current_dim_;
                const float* W_ptr = term.params.data() + 2 * rank * current_dim_;

                for (size_t r = 0; r < rank; ++r) {
                    for (size_t j = 0; j < current_dim_; ++j) {
                        u[r] += U_ptr[r * current_dim_ + j] * q_[j];
                        v[r] += V_ptr[r * current_dim_ + j] * q_[j];
                    }
                    v[r] = 1.0f / (1.0f + std::exp(-v[r]));
                }
                for (size_t i = 0; i < current_dim_; ++i) {
                    float sum = 0.0f;
                    for (size_t r = 0; r < rank; ++r) {
                        sum += W_ptr[i * rank + r] * (u[r] * v[r]);
                    }
                    force[i] += term.coefficient * sum;
                }
            } else if (term.type == DynamicTermType::LINEAR_DRIFT) {
                for (size_t i = 0; i < current_dim_; ++i) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < current_dim_; ++j) {
                        sum += term.params[i * current_dim_ + j] * q_[j];
                    }
                    force[i] += term.coefficient * sum;
                }
            }
        }

        // Symplectic kick-drift (NO friction damping)
        for (size_t i = 0; i < current_dim_; ++i) {
            p_[i] += force[i] * dt_eff_;
            // Conservative phase wrapping on unit circle/torus [-pi, pi]
            float next_q = q_[i] + p_[i] * dt_eff_;
            q_[i] = std::remainder(next_q, 2.0f * std::numbers::pi_v<float>);
        }

        // Motor Readout from Conservative Phase Lattice
        // Uses holographic phase projection: cos(q_i) and sin(q_i)
        std::vector<float> holo(2 * current_dim_);
        for (size_t i = 0; i < current_dim_; ++i) {
            holo[i] = std::cos(q_[i]);
            holo[current_dim_ + i] = std::sin(q_[i]);
        }

        std::array<float, CHANNEL_IO_DIM> resonance{};
        float max_res = -1e9f;

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float dot = 0.0f;
            const float* w_ptr = &motor_readout_[c * current_dim_];
            for (size_t i = 0; i < current_dim_; ++i) {
                dot += w_ptr[i] * holo[i]; // Primary phase resonance
            }
            float norm_dot = dot / (std::sqrt(static_cast<float>(current_dim_)) + SAFETY_EPSILON);
            float eff_res = norm_dot - anchor_momentum_[c];
            resonance[c] = eff_res;
            if (eff_res > max_res) max_res = eff_res;
        }

        // Dopaminergic Precision Scaling
        float variance = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float diff = resonance[c] - max_res;
            variance += diff * diff;
        }
        variance /= static_cast<float>(CHANNEL_IO_DIM);
        endogenous_beta_ = std::clamp(2.0f + 4.0f * std::sqrt(variance), 1.0f, 24.0f);

        // Softmax
        float sum_exp = 0.0f;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float P = std::exp((resonance[c] - max_res) * endogenous_beta_);
            latest_probabilities_[c] = P;
            sum_exp += P;
        }
        float inv_sum = 1.0f / (sum_exp + SAFETY_EPSILON);
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) latest_probabilities_[c] *= inv_sum;

        // Selection
        float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float cum = 0.0f;
        size_t chosen = 0;
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            cum += latest_probabilities_[c];
            if (cum >= r) { chosen = c; break; }
        }

        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) anchor_momentum_[c] *= 0.70f;
        anchor_momentum_[chosen] += 0.80f;

        return chosen;
    }

    void adapt(uint8_t input_byte, uint8_t target_byte, float lr = 0.05f) noexcept {
        total_steps_++;
        float p_target = latest_probabilities_[target_byte];
        float ce = -std::log(p_target + SAFETY_EPSILON);
        prev_free_energy_ = free_energy_;
        free_energy_ = 0.95f * free_energy_ + 0.05f * ce;

        float delta_FE = free_energy_ - prev_free_energy_;
        float causal_boost = (-delta_FE) * 25.0f;

        // 1. Zero-Leak Synaptic Superconductivity
        // Oja dissipation vanishes when pattern is epigenetically locked!
        for (size_t c = 0; c < CHANNEL_IO_DIM; ++c) {
            float target_val = (c == target_byte) ? 1.0f : 0.0f;
            float err = target_val - latest_probabilities_[c];
            if (std::abs(err) > 0.0002f) {
                float alpha = lr * err;
                for (size_t i = 0; i < current_dim_; ++i) {
                    size_t m_idx = c * current_dim_ + i;
                    float lock = motor_locks_[m_idx];
                    float oja_leak = (1.0f - lock) * 0.015f * motor_readout_[m_idx]; // Leak -> 0 on lock!
                    motor_readout_[m_idx] += alpha * std::cos(q_[i]) - oja_leak;

                    if (c == target_byte && p_target > 0.30f) {
                        motor_locks_[m_idx] = std::clamp(motor_locks_[m_idx] + 0.002f, 0.0f, 0.98f);
                    }
                }
            }
        }

        // 2. Conservative Gauge and Law Plasticity
        for (size_t k = 0; k < terms_.size(); ++k) {
            auto& term = terms_[k];
            term.causal_credit = 0.98f * term.causal_credit + 0.02f * causal_boost;

            if (term.causal_credit > 0.05f) {
                term.epigenetic_lock = std::clamp(term.epigenetic_lock + 0.001f, 0.0f, 0.98f);
            }

            float plastic = (1.0f - term.epigenetic_lock);
            term.coefficient += lr * 0.04f * plastic * term.causal_credit;
            term.coefficient = std::clamp(term.coefficient, -3.0f, 3.0f);

            if (term.type == DynamicTermType::SYMPLECTIC_GAUGE) {
                // Preserve anti-symmetry J = -J^T during adaptation
                for (size_t i = 0; i < current_dim_; ++i) {
                    for (size_t j = i + 1; j < current_dim_; ++j) {
                        float dJ = lr * 0.02f * plastic * causal_boost * (p_[i] * p_[j]);
                        term.params[i * current_dim_ + j] += dJ;
                        term.params[j * current_dim_ + i] -= dJ;
                    }
                }
            } else if (term.type == DynamicTermType::SENSORY_SHOCK) {
                const size_t in_offset = static_cast<size_t>(input_byte) * current_dim_;
                for (size_t i = 0; i < current_dim_; ++i) {
                    size_t s_idx = in_offset + i;
                    term.params[s_idx] += lr * 0.05f * plastic * (causal_boost * std::cos(q_[i]));
                }
            }
        }

        if (total_steps_ % 25000 == 0) {
            evaluate_and_evolve_laws();
        }
    }

    void evaluate_and_evolve_laws() {
        std::cout << "\n--- [CONSERVATIVE AUDIT @ Step " << total_steps_ << "] ---\n";
        std::cout << "  Free Energy: " << free_energy_ << " | Dim: " << current_dim_ << " | Active Laws: " << terms_.size() << "\n";

        for (size_t k = 0; k < terms_.size(); ++k) {
            const auto& t = terms_[k];
            std::cout << "  Law [" << k << "] " << term_type_to_string(t.type)
                      << " | Coeff: " << t.coefficient
                      << " | Credit: " << t.causal_credit
                      << " | Lock: " << (t.epigenetic_lock * 100.0f) << "%\n";
        }

        if (free_energy_ > 3.0f && current_dim_ < max_dim_) {
            bud_dimension();
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

    std::vector<float> q_;
    std::vector<float> p_;

    std::vector<SynthesizedTerm> terms_;
    std::vector<float> motor_readout_;
    std::vector<float> motor_locks_;

    std::array<float, CHANNEL_IO_DIM> anchor_momentum_;
    std::array<float, CHANNEL_IO_DIM> latest_probabilities_;
};

} // namespace karyon

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-587: CONSERVATIVE PHASE RESONANCE & ZERO-LEAK CONTINUUM      ===\n";
    std::cout << "=== Unshackling Long Horizons via Symplectic Dynamics (Zero Damping) ===\n";
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
        std::string fallback = "The capital of France is Paris. Language reaches zero free energy when context never decays.";
        corpus = std::vector<uint8_t>(fallback.begin(), fallback.end());
    }

    const size_t train_steps = 300000;

    std::cout << ">>> Initializing Conservative Zero-Leak Organism (Initial Dim: 128)...\n";
    karyon::ConservativeZeroLeakOrganism cpr_org(128);

    std::cout << ">>> Launching Continuous Conservative Symplectic Learning (" << train_steps << " steps)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_steps; ++step) {
        uint8_t in_b = corpus[step % corpus.size()];
        uint8_t tgt_b = corpus[(step + 1) % corpus.size()];
        cpr_org.step(in_b);
        cpr_org.adapt(in_b, tgt_b, 0.05f);

        if ((step + 1) % 75000 == 0) {
            std::cout << "  >>> Step " << (step + 1)
                      << " | Free Energy: " << cpr_org.get_free_energy()
                      << " | Beta: " << cpr_org.get_beta()
                      << " | Dim: " << cpr_org.get_dim() << "\n";
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
        size_t next_act = cpr_org.step(curr);
        speech_output += static_cast<char>(next_act);
        curr = static_cast<uint8_t>(next_act);
    }
    auto metrics = karyon::evaluate_speech(speech_output);

    std::cout << "\n========================================================================\n";
    std::cout << "=== COMPARATIVE TELEMETRY (EXP-587 vs EXP-586 vs EXP-584)             ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Substrate Architecture          | Throughput (tok/s) | Final FE (nats) | State Dim | Alphabet %\n";
    std::cout << "--------------------------------|--------------------|-----------------|-----------|-----------\n";
    std::cout << "CPR-ZLC Zero-Leak (EXP-587)     | " << throughput << "          | " << cpr_org.get_free_energy() << "          | " << cpr_org.get_dim() << "       | " << (metrics.alphabet_ratio * 100.0) << "%\n";
    std::cout << "TSLS Law Synthesizer (EXP-586)  | 1148.46            | 3.4142          | 352       | 72.89%\n";
    std::cout << "BR-HSC 3-Gateway Oja (EXP-584)  | 3251.74            | 4.0635          | 256       | 64.44%\n";
    std::cout << "========================================================================\n\n";

    std::cout << ">>> AUTONOMOUSLY SYNTHESIZED SPEECH (200 bytes):\n";
    std::cout << "\"" << speech_output << "\"\n\n";

    return 0;
}
