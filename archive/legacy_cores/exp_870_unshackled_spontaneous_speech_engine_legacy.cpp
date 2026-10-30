// ============================================================================
// KARYON-CORE EXPERIMENT EXP-870
// Title: Unshackled Autopoietic Spontaneous Speech Engine (UASSE)
// Standard: KEP v37.0 Master Protocol | Principles 2, 4, 8, 27, 30 & 31
// Substrate: Universal V = 2 Pure Bitstream (s_1 = 1 bit carrier wave)
// Author: Karyon Autonomous Lead AI Cybernetics Engine & Bazilevs (ProgVM)
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
#include <cstdint>
#include <sstream>

// Universal Bit State (SAM Node)
struct SpeechSAMNode {
    int32_t len = 0;
    int32_t link = -1;
    int32_t next[2] = {-1, -1};
    uint32_t count[2] = {0, 0};
    double causal_credit = 1.0;
    double free_energy_potential = 0.0;
};

// Substrate Spontaneous Speech Engine Class
class UnshackledSpeechSubstrate {
private:
    std::vector<SpeechSAMNode> st;
    int32_t last = 0;
    int32_t current_state = 0;
    int32_t current_len = 0;
    std::mt19937_64 rng;

    std::vector<int32_t> byte_boundary_states; // States corresponding to character/byte boundaries

public:
    UnshackledSpeechSubstrate(size_t reserve_nodes = 2000000) {
        st.reserve(reserve_nodes);
        st.push_back({0, -1, {-1, -1}, {0, 0}, 1.0, 0.0});
        last = 0;
        current_state = 0;
        current_len = 0;
        rng.seed(1337);
    }

    size_t get_node_count() const { return st.size(); }

    // Predict probability of bit 1 given current matching state
    float predict_bit(int32_t state, int32_t& out_horizon) {
        int32_t s = state;
        out_horizon = 0;

        while (s != -1) {
            uint32_t c0 = st[s].count[0];
            uint32_t c1 = st[s].count[1];
            uint32_t total = c0 + c1;

            if (total > 0) {
                out_horizon = st[s].len;
                if (c0 > 0 && c1 == 0) return 0.00001f;
                if (c1 > 0 && c0 == 0) return 0.99999f;
                return static_cast<float>(c1) / static_cast<float>(total);
            }
            s = st[s].link;
        }

        return 0.5f;
    }

    // Extend substrate topology with online bit stream
    void assimilate_bit(uint8_t bit, bool is_byte_boundary) {
        int32_t dummy_h;
        float pred_p1 = predict_bit(current_state, dummy_h);
        pred_p1 = std::clamp(pred_p1, 1e-6f, 1.0f - 1e-6f);
        float p_actual = (bit == 1) ? pred_p1 : (1.0f - pred_p1);
        double local_fe = -std::log(p_actual);

        // 1. Update frequency statistics along suffix path
        int32_t s = current_state;
        while (s != -1) {
            st[s].count[bit]++;
            st[s].causal_credit = 0.999 * st[s].causal_credit + 0.001 * (1.0 - local_fe);
            if (st[s].count[0] > 0 && st[s].count[1] > 0) {
                s = st[s].link;
            } else {
                break;
            }
        }

        // 2. Extend Suffix Automaton topology
        int32_t cur = static_cast<int32_t>(st.size());
        st.push_back({st[last].len + 1, -1, {-1, -1}, {0, 0}, 1.0, local_fe});

        int32_t p = last;
        while (p != -1 && st[p].next[bit] == -1) {
            st[p].next[bit] = cur;
            p = st[p].link;
        }

        if (p == -1) {
            st[cur].link = 0;
        } else {
            int32_t q = st[p].next[bit];
            if (st[p].len + 1 == st[q].len) {
                st[cur].link = q;
            } else {
                int32_t clone = static_cast<int32_t>(st.size());
                st.push_back({st[p].len + 1, st[q].link, {st[q].next[0], st[q].next[1]}, {st[q].count[0], st[q].count[1]}, st[q].causal_credit, st[q].free_energy_potential});
                while (p != -1 && st[p].next[bit] == q) {
                    st[p].next[bit] = clone;
                    p = st[p].link;
                }
                st[q].link = st[cur].link = clone;
            }
        }
        last = cur;

        if (is_byte_boundary) {
            byte_boundary_states.push_back(cur);
        }

        // 3. Advance online pointer
        advance_pointer(bit);
    }

    void advance_pointer(uint8_t bit) {
        while (current_state != -1 && st[current_state].next[bit] == -1) {
            current_state = st[current_state].link;
            if (current_state != -1) {
                current_len = st[current_state].len;
            }
        }
        if (current_state == -1) {
            current_state = 0;
            current_len = 0;
        } else {
            current_state = st[current_state].next[bit];
            current_len++;
        }
    }

    // SPONTANEOUS EMERGENCE & AUTONOMOUS SPEECH GENERATION
    // Driven purely by internal Free Energy dynamics starting from byte-aligned attractors
    std::string generate_spontaneous_speech(size_t max_bytes = 250, float temperature = 0.55f) {
        std::string output = "";
        
        // Pick an aligned byte-boundary state as spontaneous thought seed
        int32_t active_state = 0;
        if (!byte_boundary_states.empty()) {
            size_t idx = rng() % byte_boundary_states.size();
            active_state = byte_boundary_states[idx];
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (size_t byte_i = 0; byte_i < max_bytes; ++byte_i) {
            uint8_t gen_byte = 0;
            for (int b = 7; b >= 0; --b) {
                int32_t h;
                float p1 = predict_bit(active_state, h);

                // Thermal wave sampling
                float sampled_p1 = p1;
                if (temperature > 0.01f && p1 > 1e-5f && p1 < 1.0f - 1e-5f) {
                    float logit = std::log(p1 / (1.0f - p1));
                    logit /= temperature;
                    sampled_p1 = 1.0f / (1.0f + std::exp(-logit));
                }

                uint8_t bit = (dist(rng) < sampled_p1) ? 1 : 0;
                gen_byte |= (bit << b);

                // Advance internal state
                while (active_state != -1 && st[active_state].next[bit] == -1) {
                    active_state = st[active_state].link;
                }
                if (active_state == -1) {
                    active_state = 0;
                } else {
                    active_state = st[active_state].next[bit];
                }
            }

            char c = static_cast<char>(gen_byte);
            output += c;
        }

        return output;
    }

    // PROMPT-RESPONSIVE AUTONOMOUS SPEECH
    std::string respond_to_input(const std::string& input_text, size_t response_bytes = 250, float temperature = 0.5f) {
        current_state = 0;
        current_len = 0;

        for (char c : input_text) {
            for (int b = 7; b >= 0; --b) {
                uint8_t bit = (c >> b) & 1;
                advance_pointer(bit);
            }
        }

        std::string response = "";
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (size_t byte_i = 0; byte_i < response_bytes; ++byte_i) {
            uint8_t gen_byte = 0;
            for (int b = 7; b >= 0; --b) {
                int32_t h;
                float p1 = predict_bit(current_state, h);

                float sampled_p1 = p1;
                if (temperature > 0.01f && p1 > 1e-5f && p1 < 1.0f - 1e-5f) {
                    float logit = std::log(p1 / (1.0f - p1));
                    logit /= temperature;
                    sampled_p1 = 1.0f / (1.0f + std::exp(-logit));
                }

                uint8_t bit = (dist(rng) < sampled_p1) ? 1 : 0;
                gen_byte |= (bit << b);

                advance_pointer(bit);
            }

            char c = static_cast<char>(gen_byte);
            response += c;
        }

        return response;
    }
};

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-870: UNSHACKLED AUTOPOIETIC SPONTANEOUS SPEECH ENGINE\n";
    std::cout << " KEP v37.0 Master Protocol | Universal V = 2 Pure Bits\n";
    std::cout << "===================================================================\n\n";

    std::string dataset_path = "data/tinyshakespeare.txt";
    std::ifstream file(dataset_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[-] Error: Could not open dataset file " << dataset_path << std::endl;
        return 1;
    }

    std::string text_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    size_t train_bytes = 250000; // 250 KB = 2,000,000 raw bits
    if (train_bytes > text_data.size()) train_bytes = text_data.size();

    std::cout << "[1/3] Loading " << train_bytes << " bytes (" << (train_bytes * 8) << " raw bits) of training corpus...\n";

    UnshackledSpeechSubstrate engine(train_bytes * 2 * 8);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Online Assimilation Pass
    std::cout << "[2/3] Assimilating raw bitstream into Suffix Automaton Manifold...\n";
    for (size_t i = 0; i < train_bytes; ++i) {
        char c = text_data[i];
        for (int b = 7; b >= 0; --b) {
            uint8_t bit = (c >> b) & 1;
            engine.assimilate_bit(bit, (b == 0));
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "[+] Assimilation finished in " << elapsed_sec << " seconds! Substrate nodes: " << engine.get_node_count() << "\n\n";

    std::cout << "[3/3] DEMONSTRATING UNSHACKLED SPONTANEOUS SPEECH GENERATION\n";
    std::cout << "===================================================================\n\n";

    // 1. Spontaneous Emergence Speech (Zero Prompt - Substrate speaks on its own!)
    std::cout << "--- 🗣️ SPONTANEOUS EMERGENCE SPEECH #1 (Pure Internal Free Energy Impulse): ---\n";
    std::cout << engine.generate_spontaneous_speech(300, 0.5f) << "\n";
    std::cout << "-------------------------------------------------------------------\n\n";

    std::cout << "--- 🗣️ SPONTANEOUS EMERGENCE SPEECH #2 (Pure Internal Free Energy Impulse): ---\n";
    std::cout << engine.generate_spontaneous_speech(300, 0.55f) << "\n";
    std::cout << "-------------------------------------------------------------------\n\n";

    // 2. Dialogic Response Speech
    std::string prompt1 = "ROMEO:\nSpeak again, bright angel!\n";
    std::cout << "--- 💬 INTERACTIVE DIALOGUE PROMPT:\n\"" << prompt1 << "\"\n";
    std::cout << "--- 🎙️ KARYON RESPONSE SPEECH: ---\n";
    std::cout << engine.respond_to_input(prompt1, 300, 0.45f) << "\n";
    std::cout << "-------------------------------------------------------------------\n\n";

    return 0;
}
