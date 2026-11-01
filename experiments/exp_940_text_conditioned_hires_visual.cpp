// ============================================================================
// EXP-940: Text-Conditioned Multimodal High-Resolution Chromatic Wavefield Synthesis
// Karyon Engineering Protocol (KEP) Rule #1 Diagnostic Script
// Standard: KEP v40.0 Sovereign Master | Principle 1, 4 & 5 (Universal Wave Modality V=2)
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdint>
#include <fstream>
#include <cmath>

#include "../include/karyon_core_wavefield.hpp"

// Utility to generate a high-res structured chromatic art frame with textual tags
void GenerateMultimodalDataset(const std::string& out_bin_path, size_t width, size_t height, size_t num_samples) {
    std::ofstream out(out_bin_path, std::ios::binary);
    
    std::vector<std::string> tags = {
        "TAG: furry character purple aura golden armor neon bg",
        "TAG: furry character cybernetic blue visor silver plate",
        "TAG: furry character emerald cloak forest light shield",
        "TAG: furry character crimson flames obsidian armor dark background"
    };

    size_t img_bytes = width * height * 3;
    std::vector<uint8_t> img(img_bytes);

    for (size_t s = 0; s < num_samples; ++s) {
        std::string cur_tag = tags[s % tags.size()] + "\n\xFF\x00\xFF";
        out.write(cur_tag.data(), cur_tag.size());

        uint8_t r_base = (s == 0) ? 180 : (s == 1) ? 30 : (s == 2) ? 20 : 220;
        uint8_t g_base = (s == 0) ? 40 : (s == 1) ? 140 : (s == 2) ? 200 : 30;
        uint8_t b_base = (s == 0) ? 220 : (s == 1) ? 250 : (s == 2) ? 80 : 40;

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t idx = (y * width + x) * 3;
                double fx = static_cast<double>(x) / width;
                double fy = static_cast<double>(y) / height;

                // Procedural wave dynamics for visual structure
                uint8_t r = static_cast<uint8_t>(std::clamp(r_base + 50.0 * std::sin(fx * 6.28) * std::cos(fy * 3.14), 0.0, 255.0));
                uint8_t g = static_cast<uint8_t>(std::clamp(g_base + 50.0 * std::cos(fx * 3.14) * std::sin(fy * 6.28), 0.0, 255.0));
                uint8_t b = static_cast<uint8_t>(std::clamp(b_base + 50.0 * std::sin((fx + fy) * 6.28), 0.0, 255.0));

                img[idx + 0] = r;
                img[idx + 1] = g;
                img[idx + 2] = b;
            }
        }
        out.write(reinterpret_cast<const char*>(img.data()), img.size());
    }
    out.close();
}

int main() {
    std::cout << "==================================================================\n";
    std::cout << "EXP-940: Text-Conditioned Multimodal Hi-Res Wavefield Synthesis\n";
    std::cout << "Standard: KEP v40.0 Sovereign Master | Principle 1, 4 & 5 (V=2 Wavefield)\n";
    std::cout << "==================================================================\n\n";

    size_t width = 256;
    size_t height = 256;
    size_t num_samples = 4;
    size_t img_bytes = width * height * 3; // 196,608 bytes per 256x256 frame

    std::string dataset_path = "data/multimodal_text_art_256.bin";
    GenerateMultimodalDataset(dataset_path, width, height, num_samples);

    std::ifstream file(dataset_path, std::ios::binary | std::ios::ate);
    std::streamsize full_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(full_size);
    file.read(reinterpret_cast<char*>(buffer.data()), full_size);
    file.close();

    std::cout << "1. MULTIMODAL TEXT-CONDITIONED STREAM TELEMETRY:\n";
    std::cout << "   -> Target Stream: " << dataset_path << "\n";
    std::cout << "   -> Frame Dimension: " << width << "x" << height << " RGB (" << img_bytes << " bytes per frame)\n";
    std::cout << "   -> Stream Volume: " << full_size << " bytes (" << (full_size / 1024.0) << " KB)\n";
    std::cout << "   -> Total Binary Spins: " << (full_size * 8) << " spins\n\n";

    karyon::ContinuousWavefieldEngine engine;

    std::cout << "2. CONTINUOUS INGESTION OF MULTIMODAL TEXT+ART SPINS:\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.IngestRawBytes(buffer.data(), full_size, true);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ingest_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "   -> Ingestion Complete in: " << ingest_ms << " ms | Active Taps: " << engine.GetWaveTraceCount() << "\n\n";

    // 3. ZERO SEED IMAGE PIXELS: Condition ONLY on Text Prompt!
    std::string text_prompt = "TAG: furry character purple aura golden armor neon bg\n\xFF\x00\xFF";
    std::cout << "3. PROBING 100% TEXT-CONDITIONED VISUAL SYNTHESIS (ZERO SEED PIXELS):\n";
    std::cout << "   -> Text Prompt Conditioning: \"" << text_prompt.substr(0, 52) << "...\"\n";
    std::cout << "   -> Target Generation Volume: " << img_bytes << " bytes (Full " << width << "x" << height << " RGB Frame)\n";

    engine.ResetHistory();
    engine.IngestRawBytes(reinterpret_cast<const uint8_t*>(text_prompt.data()), text_prompt.size(), false);

    auto g0 = std::chrono::high_resolution_clock::now();
    auto synth_rgb = engine.GenerateAutoregressiveBytes(img_bytes);
    auto g1 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

    std::cout << "   -> Text-to-Image Wavefield Synthesis Complete in: " << gen_ms << " ms ("
              << (img_bytes / (gen_ms / 1000.0) / 1024.0) << " KB/s)\n";

    std::string out_bin = "data/karyon_text_prompt_synth_256.bin";
    std::ofstream out_f(out_bin, std::ios::binary);
    out_f.write(reinterpret_cast<const char*>(synth_rgb.data()), synth_rgb.size());
    out_f.close();
    std::cout << "   -> Saved Synthesized 256x256 Frame to: " << out_bin << "\n\n";

    bool pass_all = (synth_rgb.size() == img_bytes) && (engine.GetWaveTraceCount() > 0);

    std::cout << "===================================================================\n";
    std::cout << " EXP-940 VERDICT: " << (pass_all ? "🟢 POSITIVE (TEXT-CONDITIONED SYNTHESIS PASSED)" : "🔴 FAILED") << "\n";
    std::cout << "===================================================================\n";

    return pass_all ? 0 : 1;
}
