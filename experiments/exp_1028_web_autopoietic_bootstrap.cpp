// ============================================================================
// KARYON-CORE EXP-1028: SOVEREIGN AUTOPOIETIC SELF-BOOTSTRAPPING VIA WEB STREAM
// Standard: KEP v40.0 Master Specification
// Principles:
//   Principle 0 (Bazilevs' Axiom): Zero Imposed Mechanisms & Zero Error
//   Principle 1: Bit-Spin V=2 Universal Substrate (Raw Bit/Byte Streams)
//   Principle 5: Weightless Topological Container (.kcore Persistence)
//   Principle 6: Causal Autopoiesis & Self-Directed Structural Evolution
//   Principle 8: Axiom of Unshackled Flow (Boundless Dimensional Sprouting)
//   Principle 9: Law of Sovereign Self-Control (F -> 0.0 nats)
//   KEP Rule #7: Continuous High-Entropy Megabyte Bit-Spin Streams
//
// HYPOTHESIS:
// Karyon-CoRE V=2 Monad can autonomously bootstrap its internal representation field
// directly by ingesting raw web content scraped live from the Internet, evolving its
// state registers (D -> D + Delta D) and causal bit-spin laws bottom-up under Free Energy
// minimization, and persisting its weightless topological state into a binary .kcore container.
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include "../include/karyon_monad_v2.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::cout << "==================================================================================\n";
    std::cout << "  KARYON-CORE EXP-1028: SOVEREIGN AUTOPOIETIC WEB BOOTSTRAPPING ENGINE\n";
    std::cout << "  Standard: KEP v40.0 Master Specification | Pure Bit-Spin Substrate V=2\n";
    std::cout << "  Substrate: Zero Imposed Attractors | Direct Web Stream Ingestion\n";
    std::cout << "==================================================================================\n\n";

    std::string data_dir = "data/web_bootstrap_corpus";
    if (!fs::exists(data_dir)) {
        std::cerr << "[!] Web bootstrap corpus directory not found: " << data_dir << "\n";
        return 1;
    }

    // Collect all raw web scrapings
    std::vector<fs::path> corpus_files;
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.is_regular_file()) {
            corpus_files.push_back(entry.path());
        }
    }

    if (corpus_files.empty()) {
        std::cerr << "[!] No raw web text files found in " << data_dir << "\n";
        return 1;
    }

    std::cout << "[+] Found " << corpus_files.size() << " raw web stream sources for sovereign bootstrapping:\n";
    size_t total_corpus_bytes = 0;
    for (const auto& file_path : corpus_files) {
        size_t sz = fs::file_size(file_path);
        total_corpus_bytes += sz;
        std::cout << "    - " << file_path.filename().string() << " (" << sz << " bytes / " << (sz * 8) << " bits)\n";
    }
    std::cout << "  -> Total Bootstrap Stream: " << total_corpus_bytes << " bytes ("
              << (total_corpus_bytes * 8) << " bits / " << (total_corpus_bytes / 1024.0) << " KB)\n\n";

    // Initialize Sovereign Monad V=2
    karyon::SovereignMonadV2 monad(32, 16, 512);

    std::cout << "  [1/3] Initializing Autopoietic Web Bootstrap Online Learning Stream...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    uint64_t total_bits_ingested = 0;
    uint64_t step_idx = 0;
    double total_fe = 0.0;
    uint64_t total_steps = 0;

    for (const auto& file_path : corpus_files) {
        std::ifstream in(file_path, std::ios::binary);
        if (!in.is_open()) continue;

        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();

        // Convert byte stream to uint64_t bit-spin blocks (64 bits per step)
        size_t num_blocks = content.size() / sizeof(uint64_t);
        const uint64_t* raw_spins = reinterpret_cast<const uint64_t*>(content.data());

        for (size_t b = 0; b < num_blocks; ++b) {
            uint64_t input_spin = raw_spins[b];
            uint64_t target_spin = (b + 1 < num_blocks) ? raw_spins[b + 1] : raw_spins[0];

            double fe = monad.Step(input_spin, target_spin, step_idx++);
            total_fe += fe;
            total_steps++;
            total_bits_ingested += 64;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double learning_duration = std::chrono::duration<double>(end_time - start_time).count();

    monad.CompileTransform();

    std::cout << "\n  [2/3] Autopoietic Web Bootstrapping Telemetry:\n";
    std::cout << "    - Bits Ingested & Modelled    : " << total_bits_ingested << " bits (" << (total_bits_ingested / 8) << " bytes)\n";
    std::cout << "    - Online Learning Time        : " << std::fixed << std::setprecision(4) << learning_duration << " seconds\n";
    std::cout << "    - Mean Variational Free Energy: " << std::setprecision(7) << monad.GetMeanFreeEnergy() << " nats\n";
    std::cout << "    - Final State Dimension (D)   : " << monad.GetDimension() << " registers (Sprouted: +" << monad.GetTotalSprouts() << ")\n";
    std::cout << "    - Synthesized Causal Laws     : " << monad.GetLawCount() << " (Epigenetically Locked: " << monad.GetTotalLocks() << ")\n";
    std::cout << "    - Ingestion Throughput        : " << std::setprecision(2) << ((total_bits_ingested / 8.0) / (1024.0 * 1024.0)) / learning_duration << " MB/s\n\n";

    // Serialization to .kcore
    std::string kcore_file = "sovereign_web_bootstrapped.kcore";
    std::cout << "  [3/3] Serializing Bootstrapped Sovereign Container to: " << kcore_file << "...\n";
    bool save_ok = monad.SaveKCore(kcore_file);

    size_t kcore_bytes = 0;
    if (fs::exists(kcore_file)) {
        kcore_bytes = fs::file_size(kcore_file);
    }

    std::cout << "    - Serialization Status        : " << (save_ok ? "SUCCESS 🟢" : "FAILED 🔴") << "\n";
    std::cout << "    - .kcore Container Size       : " << kcore_bytes << " bytes (" << (kcore_bytes / 1024.0) << " KB)\n\n";

    // Verification of Deserialization (Parity Check)
    std::cout << "  [Verification] Testing Cold-Boot Restoration into Fresh Engine Monad B...\n";
    karyon::SovereignMonadV2 monad_b;
    bool load_ok = monad_b.LoadKCore(kcore_file);

    bool parity_pass = load_ok && (monad_b.GetDimension() == monad.GetDimension()) && (monad_b.GetLawCount() == monad.GetLawCount());
    std::cout << "    - Restoration Status          : " << (load_ok ? "SUCCESS 🟢" : "FAILED 🔴") << "\n";
    std::cout << "    - Dimension Parity Match      : " << monad_b.GetDimension() << " == " << monad.GetDimension() << "\n";
    std::cout << "    - Causal Laws Parity Match    : " << monad_b.GetLawCount() << " == " << monad.GetLawCount() << "\n\n";

    bool passed = save_ok && parity_pass && (monad.GetMeanFreeEnergy() == 0.0);

    std::cout << "==================================================================================\n";
    if (passed) {
        std::cout << "  VERDICT: 🟢 POSITIVE — SOVEREIGN AUTOPOIETIC WEB BOOTSTRAPPING MASTERED!\n";
    } else {
        std::cout << "  VERDICT: 🔴 REJECTED — Ingestion/Restoration failed\n";
    }
    std::cout << "==================================================================================\n";

    // Record Telemetry CSV
    std::ofstream csv("experiments/exp_1028_telemetry.csv");
    csv << "exp_id,total_bits,learning_sec,mean_fe,sprouted_dim,synthesized_laws,kcore_bytes,verdict\n";
    csv << "EXP-1028," << total_bits_ingested << "," << learning_duration << ","
        << monad.GetMeanFreeEnergy() << "," << monad.GetDimension() << ","
        << monad.GetLawCount() << "," << kcore_bytes << ","
        << (passed ? "POSITIVE" : "REJECTED") << "\n";
    csv.close();

    return 0;
}
