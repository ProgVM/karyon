// ============================================================================
// EXP-555: Environment-Directed Training Script (Text Stream Task)
//
// Role of this script:
// 1. Represents the external linguistic environment.
// 2. Defines what is "Correct" (Target next byte in stream).
// 3. Encodes incoming bytes into orthogonal afferent pulses (one-hot vectors).
// 4. Measures discrepancy between core's continuous efferent prediction and target.
// 5. Converts probability error into a generalized Error Vector e(t) and feeds it to Core.
// 6. Probes autonomous text generation.
// ============================================================================

#include "karyon_core.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

int main() {
    std::cout << "============================================================\n";
    std::cout << "=== EXP-555: ENVIRONMENT-DIRECTED LANGUAGE TASK BENCHMARK ===\n";
    std::cout << "============================================================\n";

    std::string path = "data/alpaca_data.json";
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        return 1;
    }

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
    file.close();

    std::cout << "Corpus: " << bytes.size() << " bytes loaded.\n";

    // Instantiate Universal Agnostic Core
    karyon::UniversalEpistemicContinuum core;

    const float dt = 0.01f;
    const float plasticity_rate = 0.05f;
    const size_t train_steps = 300000;

    std::array<float, karyon::UNIVERSAL_IO_DIM> sensory_in{};
    std::array<float, karyon::UNIVERSAL_IO_DIM> pred_out{};
    std::array<float, karyon::UNIVERSAL_IO_DIM> error_vec{};

    double cum_surprise = 0.0;
    double last_20k_surprise = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t step = 0; step < train_steps; ++step) {
        size_t idx = step % (bytes.size() - 1);
        uint8_t in_b = bytes[idx];
        uint8_t target_b = bytes[idx + 1];

        // 1. Environment prepares one-hot orthogonal sensory stimulus
        sensory_in.fill(0.0f);
        sensory_in[in_b] = 1.0f;

        // 2. Universal Core steps its internal physical wavefield
        core.step_dynamics(sensory_in.data(), sensory_in.size(),
                           pred_out.data(), pred_out.size(), dt);

        // 3. Environment computes task-specific error (Categorical Softmax Cross-Entropy)
        float max_l = -1e9f;
        for (size_t c = 0; c < 256; ++c) {
            if (pred_out[c] > max_l) max_l = pred_out[c];
        }

        float sum_exp = 0.0f;
        std::array<float, 256> probs{};
        for (size_t c = 0; c < 256; ++c) {
            float p = std::exp(pred_out[c] - max_l);
            probs[c] = p;
            sum_exp += p;
        }

        const float inv_sum_exp = 1.0f / (sum_exp + 1e-7f);
        for (size_t c = 0; c < 256; ++c) {
            probs[c] *= inv_sum_exp;
        }

        // Epistemic surprise = -ln(P(target))
        float p_target = std::max(probs[target_b], 1e-7f);
        float surprise = -std::log(p_target);
        cum_surprise += surprise;
        if (step >= train_steps - 20000) last_20k_surprise += surprise;

        // 4. Environment formulates generalized error vector: e = P(c) - y(c)
        for (size_t c = 0; c < 256; ++c) {
            error_vec[c] = probs[c] - (c == target_b ? 1.0f : 0.0f);
        }

        // 5. Universal Core assimilates error vector into internal physical tension & plasticity
        core.assimilate_error(sensory_in.data(), sensory_in.size(),
                             error_vec.data(), error_vec.size(),
                             plasticity_rate, dt);

        if (step % 50000 == 0 && step > 0) {
            double cur_avg = cum_surprise / step;
            std::cout << "  Step " << step << " | Surprise: " << cur_avg
                      << " nats/byte | PPL: " << std::exp(cur_avg) << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    double final_surprise = last_20k_surprise / 20000.0;
    std::cout << "\n============================================================\n";
    std::cout << "=== TRAINING COMPLETE ===\n";
    std::cout << "Elapsed Time:         " << elapsed_s * 1000.0 << " ms\n";
    std::cout << "Throughput:           " << (train_steps / elapsed_s) << " bytes/sec\n";
    std::cout << "Final Mean Surprise:  " << final_surprise << " nats/byte\n";
    std::cout << "Final Perplexity:     " << std::exp(final_surprise) << "\n";

    // GENERATION EXPERIMENT
    std::cout << "\n============================================================\n";
    std::cout << "=== AUTONOMOUS TEXT GENERATION BENCHMARK ===\n";
    std::cout << "============================================================\n";

    std::vector<std::string> prompts = {
        "The ",
        "Instruction: Give a definition of machine learning.\nOutput: Machine learning is ",
        "Python is a "
    };

    for (const auto& prompt : prompts) {
        std::cout << "\n--- PROMPT: \"" << prompt << "\" ---\n";
        std::cout << prompt;

        // Feed prompt bytes
        uint8_t last_b = prompt.front();
        for (char ch : prompt) {
            uint8_t b = static_cast<uint8_t>(ch);
            sensory_in.fill(0.0f);
            sensory_in[b] = 1.0f;
            core.step_dynamics(sensory_in.data(), sensory_in.size(),
                               pred_out.data(), pred_out.size(), dt);
            last_b = b;
        }

        // Generate 150 bytes with Temperature sampling T=0.4
        for (size_t g = 0; g < 150; ++g) {
            sensory_in.fill(0.0f);
            sensory_in[last_b] = 1.0f;
            core.step_dynamics(sensory_in.data(), sensory_in.size(),
                               pred_out.data(), pred_out.size(), dt);

            // Softmax with T=0.4
            float max_l = -1e9f;
            for (size_t c = 0; c < 256; ++c) {
                if (pred_out[c] > max_l) max_l = pred_out[c];
            }

            float sum_exp = 0.0f;
            std::array<float, 256> probs{};
            for (size_t c = 0; c < 256; ++c) {
                float p = std::exp((pred_out[c] - max_l) / 0.4f);
                probs[c] = p;
                sum_exp += p;
            }

            float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * sum_exp;
            float cum = 0.0f;
            uint8_t next_b = 32;
            for (size_t c = 0; c < 256; ++c) {
                cum += probs[c];
                if (cum >= r) {
                    next_b = static_cast<uint8_t>(c);
                    break;
                }
            }

            char c_char = static_cast<char>(next_b);
            std::cout << c_char << std::flush;
            last_b = next_b;
        }
        std::cout << "\n";
    }

    std::cout << "\n============================================================\n";
    return 0;
}
