// ============================================================================
// KARYON-CORE EXP-1022: DISTRIBUTED FEYNMAN WAVEFIELD MESH (16-NODE MONAD TOPOLOGY)
// Standard: KEP v40.0 Master Specification
// Principles:
//   Principle 0: Zero Imposed Mechanisms
//   Principle 1: Bit-Spin V=2 Universal Substrate
//   Principle 3: Feynman Path Integral Wave Interference (Least Action in Imaginary Time)
//   Principle 9: Law of Sovereign Self-Control (F -> 0.0 nats across all nodes)
//   KEP Rule #7: Continuous High-Entropy Megabyte Bit-Spin Streams
//
// HYPOTHESIS:
// A distributed 2D torus mesh of N = 16 parallel Sovereign Monads V=2 exchanging
// wave interference packets based on Feynman least action S[x] will achieve collective
// autopoietic convergence with F = 0.0 nats across all 16 nodes simultaneously
// while scaling throughput to multi-gigabyte processing levels (> 1.0 GB/s).
// ============================================================================

#include "../include/karyon_monad_v2.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <future>

// 2D Torus Topology Mesh node wrapper
struct MeshNode {
    size_t node_id;
    karyon::SovereignMonadV2 monad;
    uint64_t wave_packet = 0x123456789ABCDEF0ULL;
    double node_fe = 0.0;

    MeshNode(size_t id) : node_id(id), monad(64, 16, 256) {}
};

int main() {
    std::cout << "==================================================================================\n";
    std::cout << "  KARYON-CORE EXP-1022: DISTRIBUTED FEYNMAN WAVEFIELD MESH (16 NODES)\n";
    std::cout << "  Standard: KEP v40.0 Master Specification | Principle 3 Feynman Wave Interference\n";
    std::cout << "  Topology: 4x4 Torus Grid (16 Parallel Sovereign Monads V=2)\n";
    std::cout << "  Target Stream: 10,000,000 Words per Node (160,000,000 TOTAL WORDS / 10.24 BILLION BITS)\n";
    std::cout << "==================================================================================\n\n";

    const size_t NUM_NODES = 16;
    const size_t WORDS_PER_NODE = 10000000;

    std::vector<MeshNode> mesh;
    mesh.reserve(NUM_NODES);
    for (size_t i = 0; i < NUM_NODES; ++i) {
        mesh.emplace_back(i);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Multithreaded execution across 16 parallel Monad nodes
    std::vector<std::future<double>> futures;
    futures.reserve(NUM_NODES);

    for (size_t i = 0; i < NUM_NODES; ++i) {
        futures.push_back(std::async(std::launch::async, [&mesh, i, WORDS_PER_NODE]() {
            std::mt19937_64 node_rng(1022 + i);
            double total_node_fe = 0.0;

            // 4x4 Torus Neighbor Indices
            size_t row = i / 4;
            size_t col = i % 4;
            size_t north = ((row + 3) % 4) * 4 + col;
            size_t south = ((row + 1) % 4) * 4 + col;
            size_t east  = row * 4 + ((col + 1) % 4);
            size_t west  = row * 4 + ((col + 3) % 4);

            for (size_t step = 0; step < WORDS_PER_NODE; ++step) {
                uint64_t input_spin = node_rng();

                // Feynman Constructive Interference Packet from 4 Torus Neighbors
                uint64_t neighbor_packet = mesh[north].wave_packet ^
                                          mesh[south].wave_packet ^
                                          mesh[east].wave_packet  ^
                                          mesh[west].wave_packet;

                uint64_t target_spin = input_spin ^ (neighbor_packet >> 3) ^ 0x0F0F0F0F0F0F0F0FULL;

                // Step Monad node
                double fe = mesh[i].monad.Step(input_spin, target_spin, step);
                total_node_fe += fe;

                // Update node's output wave packet
                mesh[i].wave_packet = input_spin ^ target_spin;
            }

            return total_node_fe / WORDS_PER_NODE;
        }));
    }

    // Gather metrics from all 16 mesh nodes
    double collective_fe = 0.0;
    for (size_t i = 0; i < NUM_NODES; ++i) {
        double node_mean_fe = futures[i].get();
        collective_fe += node_mean_fe;
        mesh[i].node_fe = node_mean_fe;
    }
    collective_fe /= NUM_NODES;

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration<double>(end_time - start_time).count();

    uint64_t grand_total_words = NUM_NODES * WORDS_PER_NODE;
    double grand_total_bytes = static_cast<double>(grand_total_words * sizeof(uint64_t) * 2);
    double throughput_mb_s = (grand_total_bytes / (1024.0 * 1024.0)) / total_duration;
    double throughput_gb_s = throughput_mb_s / 1024.0;

    size_t total_synthesized_laws = 0;
    for (size_t i = 0; i < NUM_NODES; ++i) {
        total_synthesized_laws += mesh[i].monad.GetLawCount();
    }

    bool passed = (collective_fe == 0.0);

    std::cout << "\n==================================================================================\n";
    std::cout << "  EXP-1022 DISTRIBUTED FEYNMAN WAVEFIELD MESH TELEMETRY REPORT\n";
    std::cout << "==================================================================================\n";
    std::cout << "  Active Mesh Topology            : 4x4 Torus Grid (" << NUM_NODES << " Parallel Nodes)\n";
    std::cout << "  Grand Total Stream Processed     : " << grand_total_words << " words (10.24 BILLION BITS)\n";
    std::cout << "  Collective Mean Free Energy (F)  : " << std::fixed << std::setprecision(7) << collective_fe << " nats\n";
    std::cout << "  Execution Duration              : " << std::setprecision(4) << total_duration << " seconds\n";
    std::cout << "  Collective Parallel Throughput  : " << std::setprecision(2) << throughput_mb_s << " MB/s (" << std::setprecision(3) << throughput_gb_s << " GB/s)\n";
    std::cout << "  Total Active Laws Across Mesh   : " << total_synthesized_laws << " laws\n";
    std::cout << "==================================================================================\n";

    if (passed) {
        std::cout << "  VERDICT: 🟢 POSITIVE — DISTRIBUTED FEYNMAN MESH MASTERED WITH ZERO ERROR!\n";
    } else {
        std::cout << "  VERDICT: 🔴 REJECTED — Collective FE = " << collective_fe << "\n";
    }
    std::cout << "==================================================================================\n";

    std::ofstream csv("experiments/exp_1022_telemetry.csv");
    csv << "exp_id,nodes,total_words,collective_fe,duration_sec,throughput_mb_s,throughput_gb_s,verdict\n";
    csv << "EXP-1022," << NUM_NODES << "," << grand_total_words << "," << collective_fe << ","
        << total_duration << "," << throughput_mb_s << "," << throughput_gb_s << ","
        << (passed ? "POSITIVE" : "REJECTED") << "\n";
    csv.close();

    return 0;
}
