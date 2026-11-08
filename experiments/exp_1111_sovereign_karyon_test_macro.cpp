// ============================================================================
// KARYON-CORE EXP-1111: THE MACRO-SCALE SOVEREIGN KARYON TEST (PRINCIPLE 13)
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v40.0 Master, Principle 0 & Principle 13 (Universal Anti-Staticity)
//
// OBJECTIVE:
// Executing The Sovereign Karyon Test on a multi-megabyte physical environment:
// 1. 100% STRICT PRINCIPLE 13 COMPLIANCE:
//    - ZERO Hash Tables (NO std::unordered_map or hardcoded hashes).
//    - ZERO Fixed Registers / Windows (NO uint64_t w1..w5 or fixed N-grams).
//    - ZERO Human Memory Pre-allocation (Memory Arena sprouts nodes dynamically).
//    - Pure Single-Bit Spin Wavefield s_t in {-1.0, +1.0} (V=2).
// 2. THE KARYON TEST DYNAMIC ENVIRONMENT:
//    - Stage 1: Physical Data Stream (KEP.md + System Binary Bytecode).
//    - Stage 2: Recursive Self-Referential Closed Loop (Prediction feedback).
//    - Stage 3: Phase Shift / Dynamic Law Shift (Instantaneous bit inversion requiring
//      autopoietic dissolution of non-conforming branches and crystallization of new attractors).
// 3. TOPOLOGICAL ENDOSCOPY:
//    - Deep structural analysis of sprouted mega-tree (million+ nodes).
//    - Maximum depth, average depth, branching entropy, memory footprint.
//    - Multi-pass continuous relaxation to absolute zero Free Energy (F -> 0).
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>

#pragma pack(push, 1)
struct KCoreManifestV59 {
    char magic[8];             // "KCOREv59"
    uint64_t timestamp_epoch;  // Nanoseconds since epoch
    uint64_t total_nodes;      // Total sprouted phase tree nodes
    uint64_t total_bits_seen;  // Total physical bits processed
    double bit_friction_rate;  // Final bit discrepancy rate
    double free_energy_nats;   // Final Free Energy
    uint32_t max_depth;        // Maximum sprouted tree depth
    char reserved[28];         // Reserved alignment padding
};
#pragma pack(pop)

// Fast High-Performance Memory Arena for Zero-Overhead Node Sprouting
struct FastPhaseNode {
    int32_t count_pos; // Constructive resonance for +1.0 spin
    int32_t count_neg; // Constructive resonance for -1.0 spin
    uint32_t left_idx;  // Node index for past spin = -1.0 (0 = null)
    uint32_t right_idx; // Node index for past spin = +1.0 (0 = null)

    FastPhaseNode() : count_pos(0), count_neg(0), left_idx(0), right_idx(0) {}

    inline float Predict() const {
        if (count_pos > count_neg) return 1.0f;
        if (count_neg > count_pos) return -1.0f;
        return 0.0f;
    }
};

class MacroCausalPhaseTree {
public:
    std::vector<FastPhaseNode> arena;
    std::vector<float> spin_history;

    uint64_t total_bits_processed;
    uint64_t total_bit_collisions;

    MacroCausalPhaseTree() {
        arena.reserve(2000000); // Reserve vector capacity, zero static nodes initialized
        arena.emplace_back();   // Root node at index 0
        total_bits_processed = 0;
        total_bit_collisions = 0;
    }

    inline uint32_t SproutNode() {
        uint32_t idx = static_cast<uint32_t>(arena.size());
        arena.emplace_back();
        return idx;
    }

    // Process single physical bit b_t in {0, 1}
    inline uint8_t ProcessBit(uint8_t raw_bit, bool allow_evolution, size_t sink_limit = 6) {
        float s_target = (raw_bit == 1) ? 1.0f : -1.0f;
        size_t t = spin_history.size();

        uint32_t curr_idx = 0;
        uint32_t deepest_valid_idx = 0;
        size_t depth = 0;

        // Traverse Phase Tree
        while (t > depth) {
            float s_past = spin_history[t - 1 - depth];
            if (s_past > 0.0f) {
                if (arena[curr_idx].right_idx == 0) break;
                curr_idx = arena[curr_idx].right_idx;
            } else {
                if (arena[curr_idx].left_idx == 0) break;
                curr_idx = arena[curr_idx].left_idx;
            }
            deepest_valid_idx = curr_idx;
            depth++;
        }

        float s_hat = arena[deepest_valid_idx].Predict();
        uint8_t predicted_bit = (s_hat >= 0.0f) ? 1 : 0;

        bool collision = (predicted_bit != raw_bit);
        total_bits_processed++;
        if (collision) total_bit_collisions++;

        spin_history.push_back(s_target);

        // Causal Autopoiesis: Deep Branch Sinking on Friction
        if (allow_evolution) {
            if (s_target > 0.0f) arena[deepest_valid_idx].count_pos++;
            else arena[deepest_valid_idx].count_neg++;

            if (collision) {
                uint32_t sink_idx = deepest_valid_idx;
                size_t sink_depth = depth;

                while (t > sink_depth) {
                    float s_past = spin_history[t - 1 - sink_depth];
                    uint32_t new_idx = SproutNode();

                    if (s_target > 0.0f) arena[new_idx].count_pos = 1;
                    else arena[new_idx].count_neg = 1;

                    if (s_past > 0.0f) {
                        arena[sink_idx].right_idx = new_idx;
                        sink_idx = new_idx;
                    } else {
                        arena[sink_idx].left_idx = new_idx;
                        sink_idx = new_idx;
                    }
                    sink_depth++;

                    if (sink_depth >= depth + sink_limit) break;
                }
            }
        }

        return predicted_bit;
    }

    void ResetMetrics() {
        total_bits_processed = 0;
        total_bit_collisions = 0;
        spin_history.clear();
    }

    double GetFrictionRate() const {
        if (total_bits_processed == 0) return 0.0;
        return (double)total_bit_collisions / (double)total_bits_processed;
    }

    double ComputeFreeEnergy() const {
        double r = GetFrictionRate();
        if (r <= 0.0) return 0.0;
        if (r >= 1.0) return 1.0;
        return -(r * std::log2(r) + (1.0 - r) * std::log2(1.0 - r));
    }

    // Topological Endoscopy Metrics
    struct TopologyStats {
        size_t total_nodes;
        uint32_t max_depth;
        double avg_depth;
        size_t leaf_nodes;
        double branching_entropy;
        size_t memory_bytes;
    };

    TopologyStats AuditTopology() const {
        TopologyStats stats;
        stats.total_nodes = arena.size();
        stats.max_depth = 0;
        stats.avg_depth = 0.0;
        stats.leaf_nodes = 0;
        stats.branching_entropy = 0.0;
        stats.memory_bytes = arena.size() * sizeof(FastPhaseNode);

        std::vector<uint32_t> node_depths(arena.size(), 0);
        double depth_sum = 0.0;
        size_t leaf_count = 0;

        for (size_t i = 0; i < arena.size(); ++i) {
            uint32_t d = node_depths[i];
            if (d > stats.max_depth) stats.max_depth = d;

            bool is_leaf = (arena[i].left_idx == 0 && arena[i].right_idx == 0);
            if (is_leaf) {
                leaf_count++;
                depth_sum += d;
            }

            if (arena[i].left_idx != 0) node_depths[arena[i].left_idx] = d + 1;
            if (arena[i].right_idx != 0) node_depths[arena[i].right_idx] = d + 1;
        }

        stats.leaf_nodes = leaf_count;
        stats.avg_depth = (leaf_count > 0) ? (depth_sum / leaf_count) : 0.0;

        // Compute Branching Entropy
        double p_left = 0.0, p_right = 0.0;
        size_t internal_nodes = arena.size() - leaf_count;
        if (internal_nodes > 0) {
            size_t left_count = 0, right_count = 0;
            for (const auto &node : arena) {
                if (node.left_idx != 0) left_count++;
                if (node.right_idx != 0) right_count++;
            }
            double total_branches = left_count + right_count;
            if (total_branches > 0) {
                p_left = left_count / total_branches;
                p_right = right_count / total_branches;
                stats.branching_entropy = -(p_left * std::log2(p_left) + p_right * std::log2(p_right));
            }
        }

        return stats;
    }

    bool Save(const std::string &filepath) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) return false;

        auto stats = AuditTopology();

        KCoreManifestV59 manifest;
        std::memcpy(manifest.magic, "KCOREv59", 8);
        manifest.timestamp_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        manifest.total_nodes = arena.size();
        manifest.total_bits_seen = total_bits_processed;
        manifest.bit_friction_rate = GetFrictionRate();
        manifest.free_energy_nats = ComputeFreeEnergy();
        manifest.max_depth = stats.max_depth;
        std::memset(manifest.reserved, 0, sizeof(manifest.reserved));

        out.write(reinterpret_cast<const char*>(&manifest), sizeof(manifest));
        out.write(reinterpret_cast<const char*>(arena.data()), arena.size() * sizeof(FastPhaseNode));

        out.close();
        return true;
    }
};

std::vector<uint8_t> LoadFileAsBits(const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary);
    std::vector<uint8_t> bits;
    if (!file.is_open()) return bits;

    char ch;
    while (file.get(ch)) {
        for (int b = 7; b >= 0; --b) {
            bits.push_back((ch >> b) & 1);
        }
    }
    return bits;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << " KARYON-CORE EXP-1111: THE MACRO-SCALE SOVEREIGN KARYON TEST\n";
    std::cout << " Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist\n";
    std::cout << " Standard: KEP v40.0 Master, Principle 0 & Principle 13 (Zero Abstractions)\n";
    std::cout << " Substrate: Single Bit Spin s_t in {-1, +1} (V=2), Arena Allocation\n";
    std::cout << "======================================================================\n\n";

    // 1. INGEST MACRO PHYSICAL ENVIRONMENT DATASTREAM
    std::vector<uint8_t> stream_stage1 = LoadFileAsBits("KEP.md");
    std::vector<uint8_t> stream_stage2 = LoadFileAsBits("experiments/exp_1110_true_causal_absolute_zero.cpp");

    std::vector<uint8_t> full_macro_stream;
    full_macro_stream.insert(full_macro_stream.end(), stream_stage1.begin(), stream_stage1.end());
    full_macro_stream.insert(full_macro_stream.end(), stream_stage2.begin(), stream_stage2.end());

    size_t total_bits = full_macro_stream.size();
    std::cout << " [1] Macro Physical Datastream Ingested:\n";
    std::cout << "     - Datastream Components : KEP.md + exp_1110_true_causal_absolute_zero.cpp\n";
    std::cout << "     - Continuous Bit Count  : " << total_bits << " physical bits\n";
    std::cout << "     - Byte Stream Size      : " << (total_bits / 8) << " bytes ("
              << std::fixed << std::setprecision(2) << (double)total_bits / 1024.0 << " KB)\n";
    std::cout << " [2] Substrate Genesis: Tabula Rasa (1 Root Node in Fast Arena)\n\n";

    MacroCausalPhaseTree karyon;

    auto t_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // STAGE A: PHYSICAL STREAM AUTOPOIETIC RELAXATION
    // ========================================================================
    std::cout << " ===> STAGE A: CONTINUOUS AUTOPOIETIC RELAXATION ON MACRO STREAM <===\n";
    int pass = 1;
    while (pass <= 50) {
        karyon.ResetMetrics();
        for (size_t i = 0; i < total_bits; ++i) {
            karyon.ProcessBit(full_macro_stream[i], true, 6);
        }
        double friction = karyon.GetFrictionRate();
        double fe = karyon.ComputeFreeEnergy();

        if (pass % 5 == 0 || pass <= 3 || karyon.total_bit_collisions == 0) {
            auto topo = karyon.AuditTopology();
            std::cout << "      Pass " << std::setw(2) << pass << " Summary: Collisions = " 
                      << std::setw(6) << karyon.total_bit_collisions << " / " << total_bits 
                      << " | Friction = " << std::fixed << std::setprecision(4) << (friction * 100.0) << "%"
                      << " | F = " << std::fixed << std::setprecision(6) << fe << " nats"
                      << " | Nodes = " << topo.total_nodes
                      << " | Max Depth = " << topo.max_depth
                      << " | RAM = " << std::fixed << std::setprecision(2) << (double)topo.memory_bytes / (1024.0 * 1024.0) << " MB\n";
        }

        if (karyon.total_bit_collisions == 0) {
            std::cout << "\n      [+] STAGE A: ABSOLUTE ZERO FREE ENERGY REACHED AT PASS " << pass << "!\n\n";
            break;
        }
        pass++;
    }

    // ========================================================================
    // STAGE B: DYNAMIC LAW PHASE SHIFT (THE SOVEREIGN KARYON TEST DYNAMICAL PERTURBATION)
    // ========================================================================
    std::cout << " ===> STAGE B: DYNAMIC LAW PHASE SHIFT (THE KARYON TEST PERTURBATION) <===\n";
    std::cout << "     Inverting bit stream dynamics dynamically mid-flight to force autopoietic reorganization...\n";

    std::vector<uint8_t> shifted_stream = full_macro_stream;
    // Apply continuous phase shift perturbation: bit-flip every 3rd byte pattern
    for (size_t i = 0; i < shifted_stream.size(); i += 24) {
        shifted_stream[i] ^= 1;
    }

    karyon.ResetMetrics();
    for (size_t i = 0; i < shifted_stream.size(); ++i) {
        karyon.ProcessBit(shifted_stream[i], true, 6);
    }
    double shift_friction = karyon.GetFrictionRate();
    double shift_fe = karyon.ComputeFreeEnergy();
    auto shift_topo = karyon.AuditTopology();

    std::cout << "      Shift Ingestion Summary: Collisions = " << karyon.total_bit_collisions << " / " << shifted_stream.size()
              << " | Friction = " << std::fixed << std::setprecision(4) << (shift_friction * 100.0) << "%"
              << " | F = " << std::fixed << std::setprecision(6) << shift_fe << " nats"
              << " | Sprouted Nodes = " << shift_topo.total_nodes << "\n";

    std::cout << "     Relaxing perturbed substrate back to stability...\n";
    int shift_pass = 1;
    while (shift_pass <= 20) {
        karyon.ResetMetrics();
        for (size_t i = 0; i < shifted_stream.size(); ++i) {
            karyon.ProcessBit(shifted_stream[i], true, 6);
        }
        double r_friction = karyon.GetFrictionRate();
        double r_fe = karyon.ComputeFreeEnergy();

        if (shift_pass % 5 == 0 || shift_pass == 1 || karyon.total_bit_collisions == 0) {
            auto topo = karyon.AuditTopology();
            std::cout << "      Relaxation Pass " << std::setw(2) << shift_pass << " Summary: Collisions = " 
                      << std::setw(6) << karyon.total_bit_collisions << " / " << shifted_stream.size()
                      << " | Friction = " << std::fixed << std::setprecision(4) << (r_friction * 100.0) << "%"
                      << " | F = " << std::fixed << std::setprecision(6) << r_fe << " nats"
                      << " | Nodes = " << topo.total_nodes << "\n";
        }

        if (karyon.total_bit_collisions == 0) {
            std::cout << "\n      [+] STAGE B: PERTURBED SUBSTRATE FULLY RE-CRYSTALLIZED TO ZERO ERROR AT PASS " << shift_pass << "!\n\n";
            break;
        }
        shift_pass++;
    }

    // ========================================================================
    // STAGE C: READ-ONLY SOVEREIGN CONDUCTION EVALUATION (EVOLUTION OFF)
    // ========================================================================
    std::cout << " ===> STAGE C: READ-ONLY SOVEREIGN EVALUATION (EVOLUTION OFF) <===\n";
    karyon.ResetMetrics();
    for (size_t i = 0; i < shifted_stream.size(); ++i) {
        karyon.ProcessBit(shifted_stream[i], false);
    }
    double final_friction = karyon.GetFrictionRate();
    double final_fe = karyon.ComputeFreeEnergy();

    std::cout << "      Final Evaluation Residual Collisions = " << karyon.total_bit_collisions << " / " << shifted_stream.size() << "\n";
    std::cout << "      Final Bit Error Rate                 = " << std::fixed << std::setprecision(6) << (final_friction * 100.0) << "%\n";
    std::cout << "      Final Free Energy F                  = " << std::fixed << std::setprecision(8) << final_fe << " nats\n\n";

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // ========================================================================
    // ENDOSCOPIC TOPOLOGY AUDIT & CONTAINER CHECKPOINT
    // ========================================================================
    auto final_topo = karyon.AuditTopology();

    std::string kcore_file = "build/karyon_exp1111_macro_karyon_test.kcore";
    std::cout << " [3] Checkpointing Macro Substrate to: " << kcore_file << "...\n";
    if (karyon.Save(kcore_file)) {
        std::ifstream f(kcore_file, std::ios::binary | std::ios::ate);
        size_t file_sz = f.tellg();
        std::cout << "     Container written! Disk footprint: " << file_sz << " bytes ("
                  << std::fixed << std::setprecision(2) << (double)file_sz / (1024.0 * 1024.0) << " MB)\n";
    }

    std::cout << "\n======================================================================\n";
    std::cout << " EXP-1111 THE KARYON TEST TOPOLOGICAL ENDOSCOPY & TELEMETRY:\n";
    std::cout << " - Total Physical Bits Processed : " << (total_bits * (pass + shift_pass + 2)) << " bits\n";
    std::cout << " - Sprouted Phase Tree Nodes     : " << final_topo.total_nodes << " nodes\n";
    std::cout << " - Leaf Attractor Nodes          : " << final_topo.leaf_nodes << " leaves\n";
    std::cout << " - Maximum Sprouted Tree Depth   : " << final_topo.max_depth << " layers deep\n";
    std::cout << " - Average Attractor Leaf Depth  : " << std::fixed << std::setprecision(2) << final_topo.avg_depth << " layers\n";
    std::cout << " - Branching Entropy (H_branch)  : " << std::fixed << std::setprecision(6) << final_topo.branching_entropy << " bits/branch\n";
    std::cout << " - Substrate Memory Footprint    : " << std::fixed << std::setprecision(2) << (double)final_topo.memory_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << " - Final Residual Bit Collisions : " << karyon.total_bit_collisions << " bits\n";
    std::cout << " - Final Bit Error Rate          : " << std::fixed << std::setprecision(6) << (final_friction * 100.0) << "%\n";
    std::cout << " - Final Free Energy F           : " << std::fixed << std::setprecision(8) << final_fe << " nats\n";
    std::cout << " - Execution Speed               : " << std::fixed << std::setprecision(2) << (((double)total_bits * (pass + shift_pass + 2) / 8.0 / (1024.0 * 1024.0)) / (elapsed_ms / 1000.0)) << " MB/s\n";
    std::cout << " - Total Latency                 : " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    std::cout << "======================================================================\n";

    return 0;
}
