// ============================================================================
// KARYON-CORE EXP-856: Massive Dataset Stream & Bit-Level Speech Generation
// KEP v37.0 Master | Principles 8, 30 & 31 | V = 2 Pure Bits
//
// Substrate: Dynamic Horizon Suffix Automaton (DAWG) + Soliton Wavefield Generator
// Dataset: TinyShakespeare (1,115,394 bytes = 8,923,152 raw bits)
//
// Capabilities:
//  1. Continuous online assimilation of millions of raw bits.
//  2. Real-time telemetry: bit-level accuracy, Free Energy (F), dynamic depth.
//  3. Autoregressive Bit Generation: From a prompt, the substrate generates
//     continuous bit-waves and reconstructs them into UTF-8 text speech output!
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <random>

struct SAMState {
    int len = 0;
    int link = -1;
    int next[2] = {-1, -1};
    uint32_t count[2] = {0, 0};
};

class LargeScaleLanguageSubstrate {
private:
    std::vector<SAMState> st;
    int last = 0;
    int current_matching_state = 0;
    int current_matched_len = 0;

    std::mt19937_64 rng;

public:
    LargeScaleLanguageSubstrate(size_t expected_bits = 1000000) {
        st.reserve(expected_bits * 2 + 1000);
        st.push_back({0, -1, {-1, -1}, {0, 0}}); // Root
        last = 0;
        current_matching_state = 0;
        current_matched_len = 0;
        rng.seed(1337);
    }

    size_t get_state_count() const { return st.size(); }
    int get_matched_len() const { return current_matched_len; }

    // Predict probability of bit 1 given current online context
    float predict_bit(int& out_len) {
        out_len = current_matched_len;
        int s = current_matching_state;

        while (s != -1) {
            uint32_t c0 = st[s].count[0];
            uint32_t c1 = st[s].count[1];
            uint32_t tot = c0 + c1;

            if (tot > 0) {
                out_len = st[s].len;
                if (c0 > 0 && c1 == 0) return 0.000001f;
                if (c1 > 0 && c0 == 0) return 0.999999f;
                return static_cast<float>(c1) / static_cast<float>(tot);
            }
            s = st[s].link;
        }

        return 0.5f;
    }

    // Extend online topology with a single bit
    void extend_bit(float pred_p1, uint8_t bit, bool& is_correct, float& free_energy) {
        pred_p1 = std::clamp(pred_p1, 1e-6f, 1.0f - 1e-6f);
        float p_actual = (bit == 1) ? pred_p1 : (1.0f - pred_p1);
        free_energy = -std::log(p_actual);
        is_correct = ((pred_p1 >= 0.5f && bit == 1) || (pred_p1 < 0.5f && bit == 0));

        // 1. Update transition frequencies
        int s = current_matching_state;
        while (s != -1) {
            st[s].count[bit]++;
            // If already completely deterministic, propagate up suffix links smoothly
            if (st[s].count[0] > 0 && st[s].count[1] > 0) {
                s = st[s].link;
            } else {
                break;
            }
        }

        // 2. Extend SAM topology
        int cur = static_cast<int>(st.size());
        st.push_back({st[last].len + 1, -1, {-1, -1}, {0, 0}});

        int p = last;
        while (p != -1 && st[p].next[bit] == -1) {
            st[p].next[bit] = cur;
            p = st[p].link;
        }

        if (p == -1) {
            st[cur].link = 0;
        } else {
            int q = st[p].next[bit];
            if (st[p].len + 1 == st[q].len) {
                st[cur].link = q;
            } else {
                int clone = static_cast<int>(st.size());
                st.push_back({st[p].len + 1, st[q].link, {st[q].next[0], st[q].next[1]}, {st[q].count[0], st[q].count[1]}});
                while (p != -1 && st[p].next[bit] == q) {
                    st[p].next[bit] = clone;
                    p = st[p].link;
                }
                st[q].link = st[cur].link = clone;
            }
        }
        last = cur;

        // 3. Advance online matching pointer
        while (current_matching_state != -1 && st[current_matching_state].next[bit] == -1) {
            current_matching_state = st[current_matching_state].link;
            if (current_matching_state != -1) {
                current_matched_len = st[current_matching_state].len;
            }
        }
        if (current_matching_state == -1) {
            current_matching_state = 0;
            current_matched_len = 0;
        } else {
            current_matching_state = st[current_matching_state].next[bit];
            current_matched_len++;
        }
    }

    void feed_context_only(uint8_t bit) {
        while (current_matching_state != -1 && st[current_matching_state].next[bit] == -1) {
            current_matching_state = st[current_matching_state].link;
            if (current_matching_state != -1) {
                current_matched_len = st[current_matching_state].len;
            }
        }
        if (current_matching_state == -1) {
            current_matching_state = 0;
            current_matched_len = 0;
        } else {
            current_matching_state = st[current_matching_state].next[bit];
            current_matched_len++;
        }
    }

    // Continuous Autoregressive Bit-Wave Generation
    std::string generate_speech(const std::string& prompt, size_t num_bytes_to_generate, float temperature = 0.7f) {
        current_matching_state = 0;
        current_matched_len = 0;

        // Feed prompt bits into substrate matching state
        for (char c : prompt) {
            for (int b = 7; b >= 0; --b) {
                uint8_t bit = (c >> b) & 1;
                feed_context_only(bit);
            }
        }

        std::string generated_text = prompt;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (size_t byte_idx = 0; byte_idx < num_bytes_to_generate; ++byte_idx) {
            uint8_t gen_byte = 0;
            for (int b = 7; b >= 0; --b) {
                int dummy_len;
                float p1 = predict_bit(dummy_len);

                // Apply physical thermal wave sampling
                float sampled_p1 = p1;
                if (temperature > 0.01f && p1 > 1e-5f && p1 < 1.0f - 1e-5f) {
                    float logit = std::log(p1 / (1.0f - p1));
                    logit /= temperature;
                    sampled_p1 = 1.0f / (1.0f + std::exp(-logit));
                }

                uint8_t chosen_bit = (dist(rng) < sampled_p1) ? 1 : 0;
                gen_byte |= (chosen_bit << b);

                // Advance internal state along chosen bit
                feed_context_only(chosen_bit);
            }
            generated_text += static_cast<char>(gen_byte);
        }

        return generated_text;
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-856: MASSIVE DATASET STREAM & SPEECH GENERATION\n";
    std::cout << "  KEP v37.0 Master | Principles 8, 30 & 31 | V = 2 Pure Bits\n";
    std::cout << "=================================================================\n\n";

    std::string dataset_path = "data/tinyshakespeare.txt";
    std::ifstream file(dataset_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[-] Error opening " << dataset_path << std::endl;
        return 1;
    }

    std::string text_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    // We train on a substantial corpus: 100,000 bytes = 800,000 raw bits
    size_t train_bytes = 100000;
    if (train_bytes > text_data.size()) train_bytes = text_data.size();
    std::string train_slice = text_data.substr(0, train_bytes);

    std::vector<uint8_t> raw_bits;
    raw_bits.reserve(train_bytes * 8);
    for (char c : train_slice) {
        for (int b = 7; b >= 0; --b) {
            raw_bits.push_back((c >> b) & 1);
        }
    }

    std::cout << "[+] Loaded " << raw_bits.size() << " raw bits (" << train_bytes << " bytes) of literary Shakespeare text.\n\n";

    LargeScaleLanguageSubstrate substrate(raw_bits.size());

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int pass = 1; pass <= 2; ++pass) {
        std::cout << "--- [PASS " << pass << ": " 
                  << (pass == 1 ? "ONLINE ASSIMILATION & UNSHACKLED TOPOLOGY GROWTH" : "ATTRACTOR CONSOLIDATION")
                  << "] ---\n";

        size_t pass_correct = 0;
        size_t pass_errors = 0;
        double pass_fe_sum = 0.0;
        int max_matched_len = 0;

        for (size_t i = 0; i < raw_bits.size(); ++i) {
            int matched_len;
            float p1 = substrate.predict_bit(matched_len);
            bool correct;
            float fe;
            substrate.extend_bit(p1, raw_bits[i], correct, fe);

            max_matched_len = std::max(max_matched_len, matched_len);

            if (correct) pass_correct++;
            else pass_errors++;

            pass_fe_sum += fe;

            if ((i + 1) % 200000 == 0 || i + 1 == raw_bits.size()) {
                double cur_acc = (static_cast<double>(pass_correct) / (i + 1)) * 100.0;
                double cur_fe = pass_fe_sum / (i + 1);
                std::cout << "  [Progress: " << std::setw(6) << (i + 1) << " / " << raw_bits.size() << " bits] "
                          << "Acc: " << std::fixed << std::setprecision(2) << cur_acc << "% | "
                          << "F: " << std::setprecision(4) << cur_fe << " nats/bit | "
                          << "Max Horizon: " << max_matched_len << " bits\n";
            }
        }

        double acc = (static_cast<double>(pass_correct) / raw_bits.size()) * 100.0;
        double avg_fe = pass_fe_sum / raw_bits.size();

        std::cout << "\n>>> PASS " << pass << " SUMMARY:\n"
                  << "  - Final Accuracy               : " << std::fixed << std::setprecision(4) << acc << " %\n"
                  << "  - Remaining Errors             : " << pass_errors << " / " << raw_bits.size() << "\n"
                  << "  - Free Energy (F)              : " << avg_fe << " nats/bit\n"
                  << "  - Deepest Active Dynamic Horizon: " << max_matched_len << " bits (" << (max_matched_len/8) << " Bytes)\n"
                  << "  - Total Substrate SAM States   : " << substrate.get_state_count() << "\n\n";
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    double train_sec = std::chrono::duration<double>(train_end - start_time).count();
    std::cout << "[+] Training assimilation complete in " << train_sec << " seconds!\n\n";

    // =========================================================================
    // CONTINUOUS AUTOREGRESSIVE BIT-WAVE SPEECH GENERATION
    // =========================================================================
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE SPEECH GENERATION DIRECTLY FROM PURE BITS\n";
    std::cout << "=================================================================\n\n";

    std::vector<std::string> test_prompts = {
        "First Citizen:\nBefore we proceed any further, hear me speak.",
        "KING RICHARD II:\nWhat must the king do now?",
        "ROMEO:\nDid my heart love till now? forswear it, sight!"
    };

    for (size_t p_idx = 0; p_idx < test_prompts.size(); ++p_idx) {
        const auto& prompt = test_prompts[p_idx];
        std::cout << ">>> TEST PROMPT #" << (p_idx + 1) << ":\n\"" << prompt << "\"\n";
        std::cout << "--- KARYON GENERATED SPEECH (Temperature = 0.5): ---\n";
        
        std::string speech = substrate.generate_speech(prompt, 300, 0.5f);
        std::cout << speech << "\n";
        std::cout << "-----------------------------------------------------------------\n\n";
    }

    return 0;
}
