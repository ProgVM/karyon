// ============================================================================
// EXP-523: Continuous Sovereign String Field Language Engine (Pure Single-Pass N=1)
// Standards: KEP Principles 2, 3, 4, 27, 28, 29 (Law of Sovereign Self-Control), 30 (Anticipation)
// - NO Epochs (Single unbroken stream of text bytes)
// - NO Handcrafted Attractors (Endogenous non-linear standing waves of the continuum)
// - Online Plasticity on Metric Tensor g_{\mu\nu} (modal stiffness, damping, coupling)
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>

constexpr size_t VOCAB_SIZE = 258; // 0..255 = UTF-8 bytes, 256 = PAD, 257 = EOS
constexpr size_t DIM = 128;
constexpr size_t MODES = 16;

class SovereignStringLanguageContinuum {
public:
    SovereignStringLanguageContinuum() : gen_(1337) {
        emb_.resize(VOCAB_SIZE * DIM);
        head_.resize(DIM * VOCAB_SIZE);

        std::normal_distribution<float> d(0.0f, 0.05f);
        for (auto& w : emb_) w = d(gen_);
        for (auto& w : head_) w = d(gen_);

        // Modal Coordinates and Momenta
        q_.resize(DIM * MODES, 0.0f);
        p_.resize(DIM * MODES, 0.0f);

        // Plastic Metric Tensor: Modal Stiffness & Damping (Endogenous physical dynamics)
        stiffness_.resize(DIM * MODES, 2.0f);
        damping_.resize(DIM * MODES, 1.0f);

        for (size_t k = 0; k < MODES; ++k) {
            k_floats_[k] = static_cast<float>(k + 1);
        }
    }

    // Step continuous wave mechanics with Wheeler-Feynman anticipation (Principle 30)
    std::vector<float> step_perceive(uint8_t byte_in, float dt = 0.02f, float tau_adv = 0.01f) {
        std::vector<float> x(DIM);
        size_t emb_off = byte_in * DIM;
        for (size_t d = 0; d < DIM; ++d) x[d] = emb_[emb_off + d];

        std::vector<float> h_out(DIM, 0.0f);

        for (size_t d = 0; d < DIM; ++d) {
            float sum_q = 0.0f;
            for (size_t k = 0; k < MODES; ++k) {
                size_t idx = d * MODES + k;

                // Afferent plucking force
                float force = x[d] / k_floats_[k];
                float w_sq = stiffness_[idx] * k_floats_[k] * k_floats_[k];
                float gamma = 0.5f * damping_[idx];

                float q_star = force / (w_sq + 1e-5f);
                float q_rel = q_[idx] - q_star;
                float p_curr = p_[idx];

                // Anticipatory Wheeler-Feynman extrapolation along future light-cone (Principle 30)
                float q_adv = q_rel + tau_adv * p_curr;

                float disc = w_sq - gamma * gamma;
                float q_next = 0.0f;
                float p_next = 0.0f;

                if (disc > 1e-5f) {
                    float w_d = std::sqrt(disc);
                    float decay = std::exp(-gamma * dt);
                    q_next = decay * (q_adv * std::cos(w_d * dt) + (p_curr + gamma * q_adv) / w_d * std::sin(w_d * dt)) + q_star;
                    p_next = decay * (p_curr * std::cos(w_d * dt) - (w_sq * q_adv + gamma * p_curr) / w_d * std::sin(w_d * dt));
                } else {
                    float decay = std::exp(-gamma * dt);
                    q_next = decay * (q_adv + (p_curr + gamma * q_adv) * dt) + q_star;
                    p_next = decay * (p_curr - gamma * (p_curr + gamma * q_adv) * dt);
                }

                q_[idx] = std::clamp(q_next, -5.0f, 5.0f);
                p_[idx] = std::clamp(p_next, -10.0f, 10.0f);

                // Non-linear standing wave interferometry (Self-forming attractor basins)
                sum_q += k_floats_[k] * stiffness_[idx] * q_[idx];
            }
            h_out[d] = std::tanh(sum_q);
        }

        // Efferent Readout to Vocab Logits
        std::vector<float> logits(VOCAB_SIZE, 0.0f);
        const float synaptic_scale = 0.25f; // Principle 2 & EXP-57
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            float sum = 0.0f;
            size_t head_off = v * DIM;
            for (size_t d = 0; d < DIM; ++d) {
                sum += h_out[d] * head_[head_off + d];
            }
            logits[v] = sum * synaptic_scale;
        }

        return logits;
    }

    // Online single-pass step (Continuous plastic adaptation as bytes flow)
    float online_learn_byte(uint8_t input_b, uint8_t target_b, float lr = 0.015f) {
        std::vector<float> logits = step_perceive(input_b);

        // Softmax
        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        std::vector<float> probs(VOCAB_SIZE, 0.0f);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            probs[v] = std::exp(logits[v] - max_l);
            sum_exp += probs[v];
        }
        for (size_t v = 0; v < VOCAB_SIZE; ++v) probs[v] /= sum_exp;

        // Surprise / Free Energy on Next Byte
        float p_target = std::max(probs[target_b], 1e-7f);
        float fe_surprise = -std::log(p_target);

        // Online Plasticity w.r.t prediction error
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            float err = probs[v] - (v == target_b ? 1.0f : 0.0f);
            if (std::abs(err) < 1e-4f) continue;

            size_t head_off = v * DIM;
            size_t emb_off = input_b * DIM;

            for (size_t d = 0; d < DIM; ++d) {
                float h = q_[d * MODES]; // Modal readout proxy
                head_[head_off + d] -= lr * err * h;
                emb_[emb_off + d] -= lr * err * head_[head_off + d] * 0.1f;
            }
        }

        // Sovereign Metric Tensor Self-Tuning (Principle 29)
        // High surprise -> stiffen modes to increase temporal resolution
        float stiffness_delta = 0.0005f * (fe_surprise - 1.0f);
        for (size_t idx = 0; idx < DIM * MODES; ++idx) {
            stiffness_[idx] = std::clamp(stiffness_[idx] + stiffness_delta, 0.5f, 15.0f);
        }

        return fe_surprise;
    }

    // Generate speech from continuous wavefield
    std::string speak(const std::string& prompt, size_t max_bytes = 80, float temperature = 0.3f) {
        std::string text = prompt;
        uint8_t last_b = ' ';

        for (char c : prompt) {
            last_b = static_cast<uint8_t>(c);
            step_perceive(last_b);
        }

        for (size_t step = 0; step < max_bytes; ++step) {
            std::vector<float> logits = step_perceive(last_b);

            float max_l = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            std::vector<float> probs(VOCAB_SIZE, 0.0f);
            for (size_t v = 0; v < VOCAB_SIZE; ++v) {
                probs[v] = std::exp((logits[v] - max_l) / temperature);
                sum_exp += probs[v];
            }
            for (size_t v = 0; v < VOCAB_SIZE; ++v) probs[v] /= sum_exp;

            std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
            size_t sampled = dist(gen_);

            if (sampled == 257 || sampled == '\n') break;

            char c = static_cast<char>(sampled);
            text += c;
            last_b = static_cast<uint8_t>(sampled);
        }

        return text;
    }

private:
    std::vector<float> emb_, head_;
    std::vector<float> q_, p_;
    std::vector<float> stiffness_, damping_;
    std::mt19937 gen_;
    float k_floats_[MODES];
};

int main(int argc, char** argv) {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-523: SOVEREIGN CONTINUOUS STRING SINGLE-PASS STREAM RUNNER ====\n";
    std::cout << "=== Pure N=1 Stream | Zero Epochs | Sovereign Metric Tensor Plasticity ===\n";
    std::cout << "========================================================================\n\n";

    SovereignStringLanguageContinuum model;

    std::string stream_filename = "stream_text_buffer.txt";
    if (argc > 1) stream_filename = argv[1];

    std::ifstream file(stream_filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open stream buffer file: " << stream_filename << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string stream = buffer.str();
    file.close();

    std::cout << "Loaded continuous byte stream: " << stream.length() << " raw bytes.\n";
    std::cout << "Starting Unbroken Single-Pass Online Learning...\n\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    size_t total_bytes = stream.length() - 1;
    float rolling_fe = 5.55f;
    size_t window = 1000;
    float window_fe_sum = 0.0f;

    for (size_t i = 0; i < total_bytes; ++i) {
        uint8_t in_b = static_cast<uint8_t>(stream[i]);
        uint8_t target_b = static_cast<uint8_t>(stream[i + 1]);

        float fe = model.online_learn_byte(in_b, target_b, 0.02f);
        window_fe_sum += fe;

        if ((i + 1) % window == 0) {
            float mean_fe = window_fe_sum / window;
            float ppl = std::exp(mean_fe);
            window_fe_sum = 0.0f;

            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double bytes_per_sec = (i + 1) / (elapsed + 1e-6);

            std::cout << "  Byte " << std::setw(7) << (i + 1) << " / " << total_bytes
                      << " (" << std::fixed << std::setprecision(1) << (100.0 * (i + 1) / total_bytes) << "%)"
                      << " | Surprise (FE): " << std::setprecision(4) << mean_fe << " nats"
                      << " | PPL: " << std::setprecision(2) << ppl
                      << " | Speed: " << std::setprecision(0) << bytes_per_sec << " byte/s\n";
        }

        // Periodic speech generation check without interrupting the stream
        if ((i + 1) % 20000 == 0 || (i + 1) == total_bytes) {
            std::cout << "\n  --- [INTERMEDIATE SPEECH PROBE @ BYTE " << (i + 1) << "] ---\n";
            std::cout << "  Prompt: 'Once upon a time '\n";
            std::cout << "  Speech: \"" << model.speak("Once upon a time ", 60, 0.25f) << "\"\n";
            std::cout << "  Prompt: 'Lily '\n";
            std::cout << "  Speech: \"" << model.speak("Lily ", 60, 0.25f) << "\"\n";
            std::cout << "  ------------------------------------------------------------\n\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n========================================================================\n";
    std::cout << "Single-Pass Learning Finished in " << std::fixed << std::setprecision(2) << total_time << " s\n";
    std::cout << "Final Speech Synthesis Generation:\n";
    std::cout << "  -> Prompt: 'Once upon a time '\n";
    std::cout << "     Speech: \"" << model.speak("Once upon a time ", 80, 0.2f) << "\"\n\n";
    std::cout << "  -> Prompt: 'She was very '\n";
    std::cout << "     Speech: \"" << model.speak("She was very ", 80, 0.2f) << "\"\n\n";
    std::cout << "========================================================================\n";

    return 0;
}
