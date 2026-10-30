// ============================================================================
// KARYON-CORE EXP-857: Unshackled Visual Wavefield Generation (Raw CIFAR-10)
// KEP v37.0 Master | Principles 8, 30 & 31 | V = 2 Pure Bits
//
// Modal Invariance in Action:
//  1. Load 500 RGB images (32x32x3 = 3,072 bytes per image).
//  2. Convert pixels to a continuous raw bit stream: 1,536,000 raw bits.
//  3. Assimilate images into Suffix Automaton Substrate (DAWG).
//  4. Perform Autoregressive Bit-Wave Visual Completion & Generation:
//     - Top-Half Prompt (first 16 rows = 1,536 bytes): Karyon generates the bottom half!
//     - Visual Outpainting: Save generated images as PNGs.
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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct SAMState {
    int len = 0;
    int link = -1;
    int next[2] = {-1, -1};
    uint32_t count[2] = {0, 0};
};

class VisualWavefieldSubstrate {
private:
    std::vector<SAMState> st;
    int last = 0;
    int current_matching_state = 0;
    int current_matched_len = 0;
    std::mt19937_64 rng;

public:
    VisualWavefieldSubstrate(size_t expected_bits = 2000000) {
        st.reserve(expected_bits * 2 + 1000);
        st.push_back({0, -1, {-1, -1}, {0, 0}});
        last = 0;
        current_matching_state = 0;
        current_matched_len = 0;
        rng.seed(42);
    }

    size_t get_state_count() const { return st.size(); }

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

    void extend_bit(float pred_p1, uint8_t bit, bool& is_correct, float& free_energy) {
        pred_p1 = std::clamp(pred_p1, 1e-6f, 1.0f - 1e-6f);
        float p_actual = (bit == 1) ? pred_p1 : (1.0f - pred_p1);
        free_energy = -std::log(p_actual);
        is_correct = ((pred_p1 >= 0.5f && bit == 1) || (pred_p1 < 0.5f && bit == 0));

        int s = current_matching_state;
        while (s != -1) {
            st[s].count[bit]++;
            if (st[s].count[0] > 0 && st[s].count[1] > 0) {
                s = st[s].link;
            } else {
                break;
            }
        }

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

    void reset_context() {
        current_matching_state = 0;
        current_matched_len = 0;
    }

    // Autoregressive Visual Generation (Bit-Wave Image Completion)
    std::vector<uint8_t> generate_image_completion(const std::vector<uint8_t>& top_half_bytes, size_t total_target_bytes, float temperature = 0.6f) {
        reset_context();

        std::vector<uint8_t> result_bytes = top_half_bytes;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        // Feed top-half visual prompt into matching state
        for (uint8_t b_val : top_half_bytes) {
            for (int b = 7; b >= 0; --b) {
                uint8_t bit = (b_val >> b) & 1;
                feed_context_only(bit);
            }
        }

        size_t bytes_to_generate = total_target_bytes - top_half_bytes.size();

        for (size_t byte_idx = 0; byte_idx < bytes_to_generate; ++byte_idx) {
            uint8_t gen_byte = 0;
            for (int b = 7; b >= 0; --b) {
                int dummy_len;
                float p1 = predict_bit(dummy_len);

                float sampled_p1 = p1;
                if (temperature > 0.01f && p1 > 1e-5f && p1 < 1.0f - 1e-5f) {
                    float logit = std::log(p1 / (1.0f - p1));
                    logit /= temperature;
                    sampled_p1 = 1.0f / (1.0f + std::exp(-logit));
                }

                uint8_t chosen_bit = (dist(rng) < sampled_p1) ? 1 : 0;
                gen_byte |= (chosen_bit << b);

                feed_context_only(chosen_bit);
            }
            result_bytes.push_back(gen_byte);
        }

        return result_bytes;
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  KARYON-CORE EXP-857: UNSHACKLED VISUAL WAVEFIELD GENERATION\n";
    std::cout << "  KEP v37.0 Master | Principles 8, 30 & 31 | V = 2 Pure Bits\n";
    std::cout << "=================================================================\n\n";

    // Load CIFAR-10 raw binary dataset batch
    std::string bin_path = "data/cifar10/cifar-10-batches-py/data_batch_1";
    // CIFAR-10 python format is pickle, let's use python helper to dump 500 images as raw bytes
    std::ifstream file("data/cifar10_500_raw.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[-] Please run python helper to prepare data/cifar10_500_raw.bin first!\n";
        return 1;
    }

    std::vector<uint8_t> raw_image_bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    size_t num_images = raw_image_bytes.size() / 3072;

    std::cout << "[+] Loaded " << num_images << " RGB images (" << raw_image_bytes.size() << " bytes = " 
              << raw_image_bytes.size() * 8 << " raw bits).\n\n";

    VisualWavefieldSubstrate substrate(raw_image_bytes.size() * 8);

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "--- [ONLINE ASSIMILATION & VISUAL TOPOLOGY GROWTH] ---\n";
    size_t total_bits = raw_image_bytes.size() * 8;
    size_t correct_bits = 0;
    double fe_sum = 0.0;

    for (size_t i = 0; i < raw_image_bytes.size(); ++i) {
        uint8_t b_val = raw_image_bytes[i];
        for (int b = 7; b >= 0; --b) {
            uint8_t bit = (b_val >> b) & 1;
            int len;
            float p1 = substrate.predict_bit(len);
            bool correct;
            float fe;
            substrate.extend_bit(p1, bit, correct, fe);
            if (correct) correct_bits++;
            fe_sum += fe;
        }

        if ((i + 1) % 307200 == 0 || i + 1 == raw_image_bytes.size()) {
            size_t processed_bits = (i + 1) * 8;
            double acc = (static_cast<double>(correct_bits) / processed_bits) * 100.0;
            double cur_fe = fe_sum / processed_bits;
            std::cout << "  [Processed " << std::setw(3) << (i + 1) / 3072 << " / " << num_images << " images] "
                      << "Acc: " << std::fixed << std::setprecision(2) << acc << "% | "
                      << "F: " << std::setprecision(4) << cur_fe << " nats/bit | "
                      << "SAM States: " << substrate.get_state_count() << "\n";
        }
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    double train_sec = std::chrono::duration<double>(train_end - start_time).count();
    std::cout << "\n[+] Visual wavefield assimilation completed in " << train_sec << " seconds!\n\n";

    // =========================================================================
    // GENERATE VISUAL COMPLETIONS & OUTPAINTING
    // =========================================================================
    std::cout << "=================================================================\n";
    std::cout << "  GENERATING VISUAL COMPLETIONS (IMAGE OUTPAINTING)\n";
    std::cout << "=================================================================\n\n";

    // Take the 1st test image (3072 bytes)
    std::vector<uint8_t> original_img(raw_image_bytes.begin(), raw_image_bytes.begin() + 3072);

    // Prompt: Top 16 rows (16 * 32 * 3 = 1,536 bytes)
    std::vector<uint8_t> top_half_prompt(original_img.begin(), original_img.begin() + 1536);

    std::vector<uint8_t> generated_completion = substrate.generate_image_completion(top_half_prompt, 3072, 0.4f);

    // Save images for comparison: Original, Prompt Half, Karyon Generated
    // Convert CIFAR CHW format (3x32x32) to standard RGB HWC (32x32x3)
    auto convert_chw_to_hwc = [](const std::vector<uint8_t>& chw) {
        std::vector<uint8_t> hwc(32 * 32 * 3);
        for (int r = 0; r < 32; ++r) {
            for (int c = 0; c < 32; ++c) {
                for (int ch = 0; ch < 3; ++ch) {
                    hwc[(r * 32 + c) * 3 + ch] = chw[ch * 1024 + r * 32 + c];
                }
            }
        }
        return hwc;
    };

    std::vector<uint8_t> orig_hwc = convert_chw_to_hwc(original_img);
    std::vector<uint8_t> gen_hwc = convert_chw_to_hwc(generated_completion);

    // Save PNG files
    stbi_write_png("experiments/karyon_visual_original.png", 32, 32, 3, orig_hwc.data(), 32 * 3);
    stbi_write_png("experiments/karyon_visual_generated.png", 32, 32, 3, gen_hwc.data(), 32 * 3);

    std::cout << "[+] Saved visual artifacts:\n";
    std::cout << "  - Original Ground Truth: experiments/karyon_visual_original.png\n";
    std::cout << "  - Karyon Outpainted Image: experiments/karyon_visual_generated.png\n\n";

    return 0;
}
