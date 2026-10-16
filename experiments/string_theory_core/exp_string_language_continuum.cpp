// ============================================================================
// EXPERIMENT: Language Learning & Speech Generation on Continuous String Field
// Model: Continuous String Field Language Core (V=258, D=256, 16 Modal Harmonics)
// Standard: KEP v28.0 (Principles 2, 3, 4, 27, 28, 29, 30)
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>

// Pure C++20 Lightweight Tensor & String Continuum Layer
constexpr size_t VOCAB_SIZE = 258; // 0..255 = UTF-8 bytes, 256 = PAD, 257 = EOS
constexpr size_t EMBED_DIM = 128;
constexpr size_t MODES = 16;

struct ModalField {
    std::vector<float> q; // Coordinates
    std::vector<float> p; // Momenta
    std::vector<float> stiffness;
    std::vector<float> damping;

    ModalField(size_t dim) : q(dim * MODES, 0.0f), p(dim * MODES, 0.0f), stiffness(dim * MODES, 2.0f), damping(dim * MODES, 1.0f) {}
};

class StringLanguageContinuum {
public:
    StringLanguageContinuum() : field_(EMBED_DIM), gen_(1337) {
        // Initialize projection matrices
        emb_weights_.resize(VOCAB_SIZE * EMBED_DIM);
        head_weights_.resize(EMBED_DIM * VOCAB_SIZE);

        std::normal_distribution<float> d(0.0f, 0.1f);
        for (auto& w : emb_weights_) w = d(gen_);
        for (auto& w : head_weights_) w = d(gen_);

        // Modal frequencies
        for (size_t k = 0; k < MODES; ++k) {
            k_floats_[k] = static_cast<float>(k + 1);
        }
    }

    void reset_state() {
        std::fill(field_.q.begin(), field_.q.end(), 0.0f);
        std::fill(field_.p.begin(), field_.p.end(), 0.0f);
    }

    // Step continuous string wave dynamics with incoming byte plucking
    std::vector<float> forward_byte(uint8_t byte_in, float dt = 0.02f) {
        std::vector<float> x_emb(EMBED_DIM, 0.0f);
        size_t emb_offset = byte_in * EMBED_DIM;
        for (size_t d = 0; d < EMBED_DIM; ++d) {
            x_emb[d] = emb_weights_[emb_offset + d];
        }

        // Pluck continuous string modes along dimension
        std::vector<float> h_out(EMBED_DIM, 0.0f);

        for (size_t d = 0; d < EMBED_DIM; ++d) {
            float total_mode_q = 0.0f;
            for (size_t k = 0; k < MODES; ++k) {
                size_t idx = d * MODES + k;
                float force = x_emb[d] / k_floats_[k];
                
                // Closed-form Lie-flow update on modal harmonic
                float w_sq = field_.stiffness[idx] * k_floats_[k] * k_floats_[k];
                float gamma = 0.5f * field_.damping[idx];

                float q_star = force / (w_sq + 1e-5f);
                float q_rel = field_.q[idx] - q_star;
                float p_curr = field_.p[idx];

                float disc = w_sq - gamma * gamma;
                float q_next = 0.0f;
                float p_next = 0.0f;

                if (disc > 1e-5f) {
                    float w_d = std::sqrt(disc);
                    float decay = std::exp(-gamma * dt);
                    q_next = decay * (q_rel * std::cos(w_d * dt) + (p_curr + gamma * q_rel) / w_d * std::sin(w_d * dt)) + q_star;
                    p_next = decay * (p_curr * std::cos(w_d * dt) - (w_sq * q_rel + gamma * p_curr) / w_d * std::sin(w_d * dt));
                } else {
                    float decay = std::exp(-gamma * dt);
                    q_next = decay * (q_rel + (p_curr + gamma * q_rel) * dt) + q_star;
                    p_next = decay * (p_curr - gamma * (p_curr + gamma * q_rel) * dt);
                }

                field_.q[idx] = std::clamp(q_next, -5.0f, 5.0f);
                field_.p[idx] = std::clamp(p_next, -10.0f, 10.0f);

                total_mode_q += k_floats_[k] * field_.stiffness[idx] * field_.q[idx];
            }
            h_out[d] = std::tanh(total_mode_q);
        }

        // Linear Readout Head to Logits (VOCAB_SIZE)
        std::vector<float> logits(VOCAB_SIZE, 0.0f);
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            float sum = 0.0f;
            size_t head_offset = v * EMBED_DIM;
            for (size_t d = 0; d < EMBED_DIM; ++d) {
                sum += h_out[d] * head_weights_[head_offset + d];
            }
            logits[v] = sum;
        }

        return logits;
    }

    // Train single step with Softmax Cross-Entropy Loss and Hebbian-Lie Gradient Plasticity
    float train_step(const std::string& text_sample, float lr = 0.01f) {
        reset_state();
        if (text_sample.length() < 2) return 0.0f;

        float total_loss = 0.0f;
        size_t steps = text_sample.length() - 1;

        for (size_t i = 0; i < steps; ++i) {
            uint8_t input_byte = static_cast<uint8_t>(text_sample[i]);
            uint8_t target_byte = static_cast<uint8_t>(text_sample[i + 1]);

            std::vector<float> logits = forward_byte(input_byte);

            // Compute Softmax probabilities
            float max_l = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            std::vector<float> probs(VOCAB_SIZE, 0.0f);
            for (size_t v = 0; v < VOCAB_SIZE; ++v) {
                probs[v] = std::exp(logits[v] - max_l);
                sum_exp += probs[v];
            }
            for (size_t v = 0; v < VOCAB_SIZE; ++v) probs[v] /= sum_exp;

            // Cross-Entropy Loss
            float target_prob = std::max(probs[target_byte], 1e-7f);
            float step_loss = -std::log(target_prob);
            total_loss += step_loss;

            // Gradient Plasticity w.r.t. Head weights & Emb weights (Hebbian-Active Inference)
            for (size_t v = 0; v < VOCAB_SIZE; ++v) {
                float grad_v = probs[v] - (v == target_byte ? 1.0f : 0.0f);
                size_t head_offset = v * EMBED_DIM;
                size_t emb_offset = input_byte * EMBED_DIM;

                for (size_t d = 0; d < EMBED_DIM; ++d) {
                    float h_d = field_.q[d * MODES]; // Modal readout proxy
                    head_weights_[head_offset + d] -= lr * grad_v * h_d;
                    emb_weights_[emb_offset + d] -= lr * grad_v * head_weights_[head_offset + d] * 0.1f;
                }
            }

            // Sovereign Plasticity on String Stiffness (Principle 29)
            if (step_loss > 0.5f) {
                for (size_t idx = 0; idx < EMBED_DIM * MODES; ++idx) {
                    field_.stiffness[idx] = std::min(field_.stiffness[idx] + 0.001f * step_loss, 10.0f);
                }
            }
        }

        return total_loss / steps;
    }

    // Autoregressive Waveform Speech Generation
    std::string speak(const std::string& prompt, size_t max_bytes = 100, float temperature = 0.5f) {
        reset_state();
        std::string generated = prompt;

        // Warmup string field with prompt
        uint8_t last_byte = static_cast<uint8_t>(' ');
        for (char c : prompt) {
            last_byte = static_cast<uint8_t>(c);
            forward_byte(last_byte);
        }

        // Generate next bytes
        for (size_t step = 0; step < max_bytes; ++step) {
            std::vector<float> logits = forward_byte(last_byte);

            // Apply Temperature
            float max_l = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            std::vector<float> probs(VOCAB_SIZE, 0.0f);
            for (size_t v = 0; v < VOCAB_SIZE; ++v) {
                probs[v] = std::exp((logits[v] - max_l) / temperature);
                sum_exp += probs[v];
            }
            for (size_t v = 0; v < VOCAB_SIZE; ++v) probs[v] /= sum_exp;

            // Sample next byte
            std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
            size_t sampled_byte = dist(gen_);

            if (sampled_byte == 257 || sampled_byte == '\n') break; // EOS or Newline

            char next_char = static_cast<char>(sampled_byte);
            generated += next_char;
            last_byte = static_cast<uint8_t>(sampled_byte);
        }

        return generated;
    }

private:
    ModalField field_;
    std::vector<float> emb_weights_;
    std::vector<float> head_weights_;
    std::mt19937 gen_;
    float k_floats_[MODES];
};

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== KARYON-CORE V28.0: CONTINUOUS STRING LANGUAGE CONTINUUM TEST ======\n";
    std::cout << "========================================================================\n\n";

    StringLanguageContinuum model;

    // Corpus of pure UTF-8 natural language sentences
    std::vector<std::string> corpus = {
        "Karyon is an autonomous continuous cybernetic organism.",
        "Mind and intelligence can be instantiated in continuous field dynamics.",
        "Language emerges from the resonance of modal string harmonics.",
        "The string continuum perceives, understands, and speaks naturally.",
        "Active inference minimizes variational free energy across spatial modes."
    };

    std::cout << "[1] UNTRAINED ZERO-SHOT GENERATION (Before Learning):\n";
    std::cout << "  Prompt: 'Karyon '\n";
    std::cout << "  Output: \"" << model.speak("Karyon ", 50, 0.7f) << "\"\n\n";

    std::cout << "[2] TRAINING STRING FIELD ON LANGUAGE CORPUS (100 Epochs):\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= 100; ++epoch) {
        float avg_loss = 0.0f;
        for (const auto& text : corpus) {
            avg_loss += model.train_step(text, 0.02f);
        }
        avg_loss /= corpus.size();

        if (epoch == 1 || epoch % 20 == 0) {
            float ppl = std::exp(avg_loss);
            std::cout << "  Epoch " << std::setw(3) << epoch << " | Loss: " << std::fixed << std::setprecision(4) 
                      << avg_loss << " | PPL: " << std::setprecision(2) << ppl << std::endl;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "  Training finished in " << std::fixed << std::setprecision(3) << duration << " seconds!\n\n";

    std::cout << "[3] EVALUATING LANGUAGE COMPREHENSION & GENERATION (Post-Learning):\n";
    std::vector<std::string> prompts = {
        "Karyon ",
        "Mind and ",
        "Language ",
        "Active "
    };

    for (const auto& p : prompts) {
        std::string speech = model.speak(p, 60, 0.2f);
        std::cout << "  -> Prompt: \"" << p << "\"\n";
        std::cout << "     Speech: \"" << speech << "\"\n\n";
    }

    std::cout << "========================================================================\n";
    std::cout << ">>> LANGUAGE CONTINUUM EXPERIMENT COMPLETED SUCCESSFULLY! <<<\n";
    std::cout << "========================================================================\n";

    return 0;
}
