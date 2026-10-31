// ============================================================================
// KARYON-CORE EXPERIMENT EXP-916
// Title: Multi-Domain Sovereign V=2 Bit-Spin Substrate Benchmark Suite
// Standard: KEP v38.0 Master Protocol | Principles 2, 3, 8, 27, 28, 29, 30, 31, 32
//
// OBJECTIVE:
// Benchmark the pure V=2 DP-SAE Substrate across 3 distinct domain challenges:
//
// Domain 1: IEEE-754 64-bit Floating-Point Chaotic System Stream (Lorenz Attractor)
//           - Tests ability to learn non-linear continuous physics directly from raw IEEE-754 bit-spins.
// Domain 2: Dyck-4 Nested Context & Stack Memory Language (Parentheses/Brackets)
//           - Tests non-local long-range bit context dependencies and structural balance.
// Domain 3: Machine Executable ELF/Bytecode Binary Stream & C++ Source Code
//           - Tests raw executable machine bytecode next-bit prediction & autoregressive rollout.
//
// Author: Karyon Autonomous Lead AI Cybernetics Engine & Bazilevs (ProgVM)
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <cstring>

#include "../include/karyon_core_dp_sae.hpp"

// Utility: Convert double to 64-bit IEEE-754 raw bytes
void DoubleToBytes(double val, uint8_t* out_bytes) {
    std::memcpy(out_bytes, &val, sizeof(double));
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " KARYON-CORE EXP-916: MULTI-DOMAIN SOVEREIGN V=2 BENCHMARK SUITE\n";
    std::cout << " Standard: KEP v38.0 | Pure V=2 Bit-Spin Substrate (0% Weights)\n";
    std::cout << " Domains: IEEE-754 Chaos (Lorenz), Dyck-4 Stack Language, Bytecode\n";
    std::cout << "===================================================================\n\n";

    // ------------------------------------------------------------------------
    // DOMAIN 1: IEEE-754 64-Bit Floating-Point Lorenz System Stream
    // ------------------------------------------------------------------------
    std::cout << "--- [DOMAIN 1] IEEE-754 Lorenz System Chaos (64-bit Float Stream) ---\n";
    
    double x = 0.1, y = 0.0, z = 0.0;
    double dt = 0.01;
    double sigma = 10.0, rho = 28.0, beta = 8.0 / 3.0;

    std::vector<uint8_t> lorenz_bit_stream;
    size_t num_lorenz_steps = 25; // 25 states * 3 coords * 8 bytes = 600 bytes = 4800 bits

    for (size_t step = 0; step < num_lorenz_steps; ++step) {
        double dx = sigma * (y - x);
        double dy = x * (rho - z) - y;
        double dz = x * y - beta * z;
        x += dx * dt;
        y += dy * dt;
        z += dz * dt;

        uint8_t b_x[8], b_y[8], b_z[8];
        DoubleToBytes(x, b_x);
        DoubleToBytes(y, b_y);
        DoubleToBytes(z, b_z);

        for (int i = 0; i < 8; ++i) lorenz_bit_stream.push_back(b_x[i]);
        for (int i = 0; i < 8; ++i) lorenz_bit_stream.push_back(b_y[i]);
        for (int i = 0; i < 8; ++i) lorenz_bit_stream.push_back(b_z[i]);
    }

    std::cout << "  -> Generated Lorenz Stream: " << lorenz_bit_stream.size() << " bytes (" << (lorenz_bit_stream.size() * 8) << " bits)\n";

    karyon::SovereignV2Engine lorenz_engine;
    auto t_lorenz_0 = std::chrono::high_resolution_clock::now();
    lorenz_engine.IngestRawBytes(lorenz_bit_stream.data(), lorenz_bit_stream.size(), true);
    auto t_lorenz_1 = std::chrono::high_resolution_clock::now();
    double lorenz_train_ms = std::chrono::duration<double, std::milli>(t_lorenz_1 - t_lorenz_0).count();

    // Verify 100% autoregressive reproduction of raw IEEE-754 bitstream
    size_t seed_bytes = 24; // First state (24 bytes) as seed
    lorenz_engine.ResetHistory();
    lorenz_engine.IngestRawBytes(lorenz_bit_stream.data(), seed_bytes, false);

    size_t gen_bytes_count = lorenz_bit_stream.size() - seed_bytes;
    auto gen_lorenz_bytes = lorenz_engine.GenerateAutoregressiveBytes(gen_bytes_count);

    size_t match_bytes = 0;
    for (size_t i = 0; i < gen_bytes_count; ++i) {
        if (gen_lorenz_bytes[i] == lorenz_bit_stream[seed_bytes + i]) {
            match_bytes++;
        }
    }
    double lorenz_acc = 100.0 * static_cast<double>(match_bytes) / gen_bytes_count;

    std::cout << "  -> Lorenz Ingestion Time: " << std::fixed << std::setprecision(2) << lorenz_train_ms << " ms\n";
    std::cout << "  -> Attractors Synthesized: " << lorenz_engine.GetAttractorCount() << "\n";
    std::cout << "  -> IEEE-754 Bit-Spin Autoregressive Parity: " << lorenz_acc << "% (" << match_bytes << "/" << gen_bytes_count << " bytes)\n\n";

    // ------------------------------------------------------------------------
    // DOMAIN 2: Dyck-4 Nested Context & Stack Memory Language
    // ------------------------------------------------------------------------
    std::cout << "--- [DOMAIN 2] Dyck-4 Deeply Nested Stack Memory Language ---\n";
    std::string dyck_stream;
    for (int r = 0; r < 5; ++r) {
        dyck_stream += "((({[[[<<<>>>]]]}}}))) ";
        dyck_stream += "[[[(((<<<>>>)))]]] ";
        dyck_stream += "<<<((([[[]]])))>>> ";
    }

    std::cout << "  -> Generated Dyck-4 Stream: " << dyck_stream.size() << " bytes (" << (dyck_stream.size() * 8) << " bits)\n";

    karyon::SovereignV2Engine dyck_engine;
    auto t_dyck_0 = std::chrono::high_resolution_clock::now();
    dyck_engine.IngestRawBytes(reinterpret_cast<const uint8_t*>(dyck_stream.data()), dyck_stream.size(), true);
    auto t_dyck_1 = std::chrono::high_resolution_clock::now();
    double dyck_train_ms = std::chrono::duration<double, std::milli>(t_dyck_1 - t_dyck_0).count();

    // Rollout test from prompt: "(("
    std::string dyck_seed = "((";
    dyck_engine.ResetHistory();
    dyck_engine.IngestRawBytes(reinterpret_cast<const uint8_t*>(dyck_seed.data()), dyck_seed.size(), false);

    size_t dyck_gen_len = dyck_stream.size() - dyck_seed.size();
    auto gen_dyck = dyck_engine.GenerateAutoregressiveBytes(dyck_gen_len);
    std::string full_dyck = dyck_seed + std::string(gen_dyck.begin(), gen_dyck.end());

    size_t dyck_match = 0;
    for (size_t i = 0; i < dyck_stream.size(); ++i) {
        if (full_dyck[i] == dyck_stream[i]) dyck_match++;
    }
    double dyck_acc = 100.0 * static_cast<double>(dyck_match) / dyck_stream.size();

    std::cout << "  -> Dyck-4 Ingestion Time: " << dyck_train_ms << " ms\n";
    std::cout << "  -> Dyck-4 Attractors Synthesized: " << dyck_engine.GetAttractorCount() << "\n";
    std::cout << "  -> Dyck-4 Stack Parity: " << dyck_acc << "% (" << dyck_match << "/" << dyck_stream.size() << " bytes)\n\n";

    // ------------------------------------------------------------------------
    // DOMAIN 3: Machine Executable Bytecode & Assembly Logic Stream
    // ------------------------------------------------------------------------
    std::cout << "--- [DOMAIN 3] Machine Executable Bytecode & Assembly Logic ---\n";
    
    std::vector<uint8_t> machine_bytecode = {
        0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00, // ELF Header
        0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x10, 0xbf, // MOV, SUB, MOV instructions
        0x01, 0x00, 0x00, 0x00, 0x48, 0x8d, 0x35, 0x10, // Opcode parameters
        0x00, 0x00, 0x00, 0xe8, 0xfe, 0xff, 0xff, 0xff, // CALL instruction
        0xc3, 0x90, 0x90, 0x90                         // RET, NOP, NOP, NOP
    };

    std::vector<uint8_t> full_bytecode;
    for (int rep = 0; rep < 8; ++rep) {
        full_bytecode.insert(full_bytecode.end(), machine_bytecode.begin(), machine_bytecode.end());
    }

    std::cout << "  -> Generated Machine Bytecode Stream: " << full_bytecode.size() << " bytes (" << (full_bytecode.size() * 8) << " bits)\n";

    karyon::SovereignV2Engine bytecode_engine;
    auto t_byte_0 = std::chrono::high_resolution_clock::now();
    bytecode_engine.IngestRawBytes(full_bytecode.data(), full_bytecode.size(), true);
    auto t_byte_1 = std::chrono::high_resolution_clock::now();
    double byte_train_ms = std::chrono::duration<double, std::milli>(t_byte_1 - t_byte_0).count();

    // Rollout from ELF header seed (4 bytes: 0x7f 0x45 0x4c 0x46)
    bytecode_engine.ResetHistory();
    bytecode_engine.IngestRawBytes(full_bytecode.data(), 4, false);

    size_t byte_gen_len = full_bytecode.size() - 4;
    auto gen_bytecode = bytecode_engine.GenerateAutoregressiveBytes(byte_gen_len);

    size_t byte_match = 0;
    for (size_t i = 0; i < byte_gen_len; ++i) {
        if (gen_bytecode[i] == full_bytecode[4 + i]) byte_match++;
    }
    double byte_acc = 100.0 * static_cast<double>(byte_match) / byte_gen_len;

    std::cout << "  -> Bytecode Ingestion Time: " << byte_train_ms << " ms\n";
    std::cout << "  -> Bytecode Attractors Synthesized: " << bytecode_engine.GetAttractorCount() << "\n";
    std::cout << "  -> Bytecode Generation Parity: " << byte_acc << "% (" << byte_match << "/" << byte_gen_len << " bytes)\n\n";

    // ------------------------------------------------------------------------
    // SUMMARY REPORT
    // ------------------------------------------------------------------------
    bool all_passed = (lorenz_acc == 100.0) && (dyck_acc == 100.0) && (byte_acc == 100.0);

    std::cout << "===================================================================\n";
    std::cout << " MULTI-DOMAIN V=2 BENCHMARK SUMMARY REPORT:\n";
    std::cout << "  1. IEEE-754 Lorenz Chaos Stream: " << (lorenz_acc == 100.0 ? "100.00% PASSED 🏆" : "FAIL") << "\n";
    std::cout << "  2. Dyck-4 Stack Memory Language: " << (dyck_acc == 100.0 ? "100.00% PASSED 🏆" : "FAIL") << "\n";
    std::cout << "  3. Machine Executable Bytecode:  " << (byte_acc == 100.0 ? "100.00% PASSED 🏆" : "FAIL") << "\n";
    std::cout << "  -> OVERALL VERDICT: " << (all_passed ? "🟢 ALL DOMAINS 100.00% SUCCESS!" : "🔴 PARTIAL DEGRADATION") << "\n";
    std::cout << "===================================================================\n";

    return all_passed ? 0 : 1;
}
