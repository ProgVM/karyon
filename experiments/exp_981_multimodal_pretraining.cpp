#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <iomanip>
#include "karyon_core_wavefield.hpp"

// ============================================================================
// KARYON-CORE EXP-981: MULTI-MODAL SOVEREIGN PRE-TRAINING BENCHMARK
// KEP v40.0 Sovereign Master | Principle 1 ($V=258 / V=2$), Principle 7 & 11
// Goal: 
// 1. Ingest continuous multi-modal raw byte stream (Text, Images, Audio, Bytecode).
// 2. Perform continuous online unsupervised wavefield pre-training on uint64_t SIMD manifold.
// 3. Minimize Variational Free Energy F -> 0 and measure multi-modal stability & throughput.
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-981: MULTI-MODAL SOVEREIGN PRE-TRAINING BENCHMARK\n";
    std::cout << " KEP v40.0 Sovereign Master | Raw Multi-Modal Byte Stream Ingestion\n";
    std::cout << " Substrate: uint64_t Bit-Packed Wavefield (V=2 / ZERO WEIGHT MATRICES)\n";
    std::cout << "===================================================================\n\n";

    std::string stream_path = "data/multimodal_stream/multimodal_continuous_stream.bin";
    if (argc > 1) stream_path = argv[1];

    std::ifstream file(stream_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[!] Failed to open multi-modal stream file: " << stream_path << std::endl;
        return 1;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> raw_bytes(file_size);
    if (!file.read(reinterpret_cast<char*>(raw_bytes.data()), file_size)) {
        std::cerr << "[!] Failed to read bytes from file." << std::endl;
        return 1;
    }

    std::cout << "[+] Successfully loaded Multi-Modal Stream: " << (file_size / (1024.0 * 1024.0)) << " MB (" << file_size << " raw bytes)\n";

    ContinuousWavefieldEngine engine;
    
    // Ingest multi-modal stream in 4096-byte frames
    constexpr size_t FRAME_SIZE_BYTES = 4096;
    engine.IngestDataset(raw_bytes.data(), raw_bytes.size(), FRAME_SIZE_BYTES);

    size_t total_frames = engine.GetFrameCount();
    size_t words_per_frame = engine.GetWordsPerFrame();

    std::cout << "[+] Dataset Ingested: " << total_frames << " frames (" << words_per_frame << " uint64_t words per frame)\n";

    // Setup working pre-training buffers
    std::vector<uint64_t> current_state(words_per_frame);
    std::vector<uint64_t> next_state(words_per_frame);

    // Initialize with first frame
    const uint64_t* manifold_data = engine.GetPackedManifoldData();
    std::memcpy(current_state.data(), manifold_data, words_per_frame * sizeof(uint64_t));

    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t total_bit_matches = 0;
    int total_steps = 0;

    // Multi-Modal Continuous Pre-Training Pass (Wave Resonance & Phase Attractor Minimization)
    constexpr int EPOCHS = 10;
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        for (size_t f = 1; f < total_frames; ++f) {
            const uint64_t* target_frame = manifold_data + f * words_per_frame;

            uint64_t matches = engine.StepBitpackedWavefieldSIMD(
                current_state.data(), target_frame, next_state.data(), words_per_frame
            );

            total_bit_matches += matches;
            total_steps++;
            current_state.swap(next_state);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();

    double total_processed_bits = (double)total_steps * words_per_frame * 64;
    double giga_spins_sec = (total_processed_bits / 1e9) / total_sec;
    double mb_sec = ((total_processed_bits / 8.0) / (1024.0 * 1024.0)) / total_sec;

    double max_bits = (double)total_steps * words_per_frame * 64;
    double avg_bit_accuracy = 100.0 * (double)total_bit_matches / max_bits;
    double free_energy_f = 1.0 - (avg_bit_accuracy / 100.0);

    std::cout << "\n===================================================================\n";
    std::cout << " === EXP-981 MULTI-MODAL PRE-TRAINING TELEMETRY REPORT ===\n";
    std::cout << "===================================================================\n";
    std::cout << "[+] Total Frames Processed     : " << (total_frames * EPOCHS) << " Frames\n";
    std::cout << "[+] Total Ingested Bit-Spins   : " << std::fixed << std::setprecision(0) << total_processed_bits << " Spins\n";
    std::cout << "[+] Multi-Modal Pre-Train Speed: " << std::setprecision(2) << (total_steps / total_sec) << " Frames/sec\n";
    std::cout << "[+] Processing Bandwidth       : " << std::setprecision(2) << giga_spins_sec << " G-Spins/sec 🚀 | " << mb_sec << " MB/sec ⚡\n";
    std::cout << "[+] Bit-Phase Phase Alignment  : " << std::setprecision(2) << avg_bit_accuracy << "%\n";
    std::cout << "[+] Final Variational Free E.  : " << std::setprecision(6) << free_energy_f << " nats (F -> 0)\n";
    std::cout << "[+] Total Execution Time       : " << std::setprecision(4) << (total_sec * 1000.0) << " ms\n";
    std::cout << "===================================================================\n";

    return 0;
}
