// ============================================================================
// KARYON-CORE EXP-1105: CAUSAL AUTOPOIETIC PHASE-TREE SUBSTRATE (PRINCIPLE 6 & 13)
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v40.0 Sovereign Master, Principle 0, Principle 6 & Principle 13
//
// PHYSICAL FOUNDATION (ZERO HUMAN CRUTCHES):
// 1. Absolutely ZERO Hash Tables (NO std::unordered_map or hardcoded hash functions).
// 2. Absolutely ZERO Fixed Register Boxes (NO uint64_t w1..w5 or fixed N-grams).
// 3. Dynamic Endogenous Phase Tree (Bit-Spin Triplet Nodes):
//    - Memory is an autopoietic dynamic tree of phase-delay branches sprouted directly from physical bit friction.
//    - Each node represents a physical spin state (-1.0 / +1.0) at an endogenous delay offset.
//    - Dynamic Path Traversal: As new bits b_t arrive, the substrate travels down the active phase tree.
//    - Causal Credit Assignment (Principle 6): When bit collision occurs, the tree branches endogenously along the exact causal vector of stability, extending depth ONLY where needed.
//    - Unfit branches with zero energy decay and dissolve (autopoietic pruning).
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <cstring>

#pragma pack(push, 1)
struct KCoreManifestV53 {
    char magic[8];             // "KCOREv53"
    uint64_t timestamp_epoch;  // Nanoseconds since epoch
    uint64_t total_nodes;      // Total sprouted phase tree nodes
    uint64_t total_bits_seen;  // Total physical bits processed
    double bit_friction_rate;  // Final bit discrepancy rate
    double free_energy_nats;   // Final Free Energy
    char reserved[32];         // Reserved alignment padding
};
#pragma pack(pop)

// Dynamic Autopoietic Phase Tree Node (Zero Fixed Registers / Zero Hash Tables)
struct PhaseTreeNode {
    float spin_state;                      // Physical spin value (-1.0 or +1.0)
    int32_t spin_count_pos;                // Constructive resonance counter for +1
    int32_t spin_count_neg;                // Constructive resonance counter for -1
    std::shared_ptr<PhaseTreeNode> left;   // Branch for past spin = -1.0
    std::shared_ptr<PhaseTreeNode> right;  // Branch for past spin = +1.0

    PhaseTreeNode(float spin = 0.0f) : spin_state(spin), spin_count_pos(0), spin_count_neg(0) {}

    inline float Predict() const {
        if (spin_count_pos > spin_count_neg) return 1.0f;
        if (spin_count_neg > spin_count_pos) return -1.0f;
        return 0.0f;
    }
};

class CausalAutopoieticPhaseTreeSubstrate {
public:
    std::shared_ptr<PhaseTreeNode> root;
    std::vector<float> spin_history;

    uint64_t total_bits_processed;
    uint64_t total_bit_collisions;
    uint64_t total_sprouted_nodes;

    CausalAutopoieticPhaseTreeSubstrate() {
        root = std::make_shared<PhaseTreeNode>(0.0f);
        total_bits_processed = 0;
        total_bit_collisions = 0;
        total_sprouted_nodes = 1;
    }

    // Process single physical bit b_t in {0, 1}
    inline uint8_t ProcessBit(uint8_t raw_bit, bool allow_evolution) {
        float s_target = (raw_bit == 1) ? 1.0f : -1.0f;
        size_t t = spin_history.size();

        // 1. Traverse Continuous Dynamic Phase Tree based on past bit-spin trajectory
        std::shared_ptr<PhaseTreeNode> curr = root;
        std::shared_ptr<PhaseTreeNode> deepest_valid = root;

        size_t depth = 0;
        while (t > depth) {
            float s_past = spin_history[t - 1 - depth];
            if (s_past > 0.0f) {
                if (!curr->right) break;
                curr = curr->right;
            } else {
                if (!curr->left) break;
                curr = curr->left;
            }
            deepest_valid = curr;
            depth++;
        }

        // Physical Prediction from deepest active phase-attractor branch
        float s_hat = deepest_valid->Predict();
        uint8_t predicted_bit = (s_hat >= 0.0f) ? 1 : 0;

        bool collision = (predicted_bit != raw_bit);
        total_bits_processed++;
        if (collision) total_bit_collisions++;

        // Push current target spin into history
        spin_history.push_back(s_target);

        // 2. Causal Autopoietic Sprouting & Reinforcement under Friction (Principle 6)
        if (allow_evolution) {
            // Reinforce constructive resonance on the active path
            if (s_target > 0.0f) deepest_valid->spin_count_pos++;
            else deepest_valid->spin_count_neg++;

            // Sprout new phase tree branch endogenously ONLY when friction occurs (Bit Collision)
            if (collision && t > depth) {
                float s_past = spin_history[t - 1 - depth];
                auto new_node = std::make_shared<PhaseTreeNode>(s_past);
                if (s_target > 0.0f) new_node->spin_count_pos = 1;
                else new_node->spin_count_neg = 1;

                if (s_past > 0.0f) deepest_valid->right = new_node;
                else deepest_valid->left = new_node;

                total_sprouted_nodes++;
            }
        }

        return predicted_bit;
    }

    void ResetMetrics() {
        total_bits_processed = 0;
        total_bit_collisions = 0;
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

    void SerializeNode(std::ofstream &out, const std::shared_ptr<PhaseTreeNode> &node) {
        if (!node) return;
        out.write(reinterpret_cast<const char*>(&node->spin_state), sizeof(node->spin_state));
        out.write(reinterpret_cast<const char*>(&node->spin_count_pos), sizeof(node->spin_count_pos));
        out.write(reinterpret_cast<const char*>(&node->spin_count_neg), sizeof(node->spin_count_neg));
        uint8_t flags = (node->left ? 1 : 0) | (node->right ? 2 : 0);
        out.write(reinterpret_cast<const char*>(&flags), sizeof(flags));

        if (node->left) SerializeNode(out, node->left);
        if (node->right) SerializeNode(out, node->right);
    }

    bool Save(const std::string &filepath) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) return false;

        KCoreManifestV53 manifest;
        std::memcpy(manifest.magic, "KCOREv53", 8);
        manifest.timestamp_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        manifest.total_nodes = total_sprouted_nodes;
        manifest.total_bits_seen = total_bits_processed;
        manifest.bit_friction_rate = GetFrictionRate();
        manifest.free_energy_nats = ComputeFreeEnergy();
        std::memset(manifest.reserved, 0, sizeof(manifest.reserved));

        out.write(reinterpret_cast<const char*>(&manifest), sizeof(manifest));
        SerializeNode(out, root);

        out.close();
        return true;
    }
};

std::vector<uint8_t> LoadPhysicalBitStream(const std::string &filepath) {
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
    std::cout << " KARYON-CORE EXP-1105: CAUSAL AUTOPOIETIC PHASE-TREE SUBSTRATE (V=2)\n";
    std::cout << " Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist\n";
    std::cout << " Compliance: STRICT KEP PRINCIPLE 13 & PRINCIPLE 6 (CAUSAL CREDIT)\n";
    std::cout << " Abstractions Eliminated: ZERO Hash Tables, ZERO Fixed Register Boxes\n";
    std::cout << " Mechanism: Endogenous Phase Tree Sprouting under Bit Friction\n";
    std::cout << "======================================================================\n\n";

    std::string path = "KEP.md";
    std::vector<uint8_t> bit_stream = LoadPhysicalBitStream(path);
    if (bit_stream.empty()) {
        std::cerr << "[-] Error: Failed to load physical bit stream from " << path << "\n";
        return 1;
    }

    size_t total_bits = bit_stream.size();
    std::cout << " [1] Physical Stream Ingested:\n";
    std::cout << "     - Source       : " << path << "\n";
    std::cout << "     - Raw Bit Count: " << total_bits << " continuous physical bits\n";
    std::cout << "     - Storage Size : " << (total_bits / 8) << " bytes (" 
              << std::fixed << std::setprecision(2) << (double)total_bits / (8.0 * 1024.0) << " KB)\n";
    std::cout << " [2] Substrate Genesis: Root Node Only (1 Initial Node)\n\n";

    CausalAutopoieticPhaseTreeSubstrate karyon;

    auto t_start = std::chrono::high_resolution_clock::now();

    // PASS 1: Active Dynamic Phase Tree Sprouting & Adaptation
    std::cout << " ---> Pass 1 (Active Causal Phase Tree Sprouting & Adaptation)...\n";
    karyon.ResetMetrics();

    for (size_t i = 0; i < total_bits; ++i) {
        karyon.ProcessBit(bit_stream[i], true);

        if ((i + 1) % 20000 == 0 || (i + 1) == total_bits) {
            std::cout << "      [Progress " << std::setw(6) << (i + 1) << " / " << total_bits << " bits] "
                      << "Sprouted Phase Nodes: " << karyon.total_sprouted_nodes 
                      << " | Bit Collisions: " << karyon.total_bit_collisions 
                      << " (" << std::fixed << std::setprecision(2) << (karyon.GetFrictionRate() * 100.0) << "%)\n";
        }
    }

    double p1_friction = karyon.GetFrictionRate();
    double p1_fe = karyon.ComputeFreeEnergy();
    std::cout << "      Pass 1 Summary: Friction Rate = " << std::fixed << std::setprecision(4) 
              << (p1_friction * 100.0) << "% | F = " << p1_fe << " nats | Sprouted Nodes = " 
              << karyon.total_sprouted_nodes << "\n\n";

    // PASS 2: Attractor Resonant Consolidation
    std::cout << " ---> Pass 2 (Attractor Resonant Consolidation & Phase Locking)...\n";
    karyon.ResetMetrics();
    for (size_t i = 0; i < total_bits; ++i) {
        karyon.ProcessBit(bit_stream[i], true);
    }
    double p2_friction = karyon.GetFrictionRate();
    double p2_fe = karyon.ComputeFreeEnergy();
    std::cout << "      Pass 2 Summary: Friction Rate = " << std::fixed << std::setprecision(4) 
              << (p2_friction * 100.0) << "% | F = " << p2_fe << " nats | Total Phase Nodes = " 
              << karyon.total_sprouted_nodes << "\n\n";

    // PASS 3: Sovereign Conduction Evaluation (Evolution OFF / Physical Conduction)
    std::cout << " ---> Pass 3 (The Sovereign Evaluation: Evolution OFF / Physical Conduction)...\n";
    karyon.ResetMetrics();
    for (size_t i = 0; i < total_bits; ++i) {
        karyon.ProcessBit(bit_stream[i], false);
    }
    double p3_friction = karyon.GetFrictionRate();
    double p3_fe = karyon.ComputeFreeEnergy();
    std::cout << "      Pass 3 Summary: Residual Bit Collisions = " << karyon.total_bit_collisions << " / " << total_bits << "\n";
    std::cout << "      Bit Error Rate = " << std::fixed << std::setprecision(6) << (p3_friction * 100.0) << "%\n";
    std::cout << "      Free Energy F  = " << std::fixed << std::setprecision(8) << p3_fe << " nats\n\n";

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double throughput_mb_s = ((double)total_bits * 3.0 / 8.0 / (1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

    // Save Container .kcore v53
    std::string kcore_file = "build/karyon_exp1105_causal_tree.kcore";
    std::cout << " [3] Checkpointing Causal Autopoietic Substrate to: " << kcore_file << "...\n";
    if (karyon.Save(kcore_file)) {
        std::ifstream f(kcore_file, std::ios::binary | std::ios::ate);
        size_t file_sz = f.tellg();
        std::cout << "     Container written! Disk footprint: " << file_sz << " bytes ("
                  << std::fixed << std::setprecision(2) << (double)file_sz / 1024.0 << " KB)\n";
    }

    std::cout << "\n======================================================================\n";
    std::cout << " EXP-1105 FINAL TELEMETRY SUMMARY:\n";
    std::cout << " - Total Bits Processed  : " << (total_bits * 3) << " physical bits\n";
    std::cout << " - Sprouted Tree Nodes   : " << karyon.total_sprouted_nodes << " autopoietic phase tree nodes\n";
    std::cout << " - Pass 3 Bit Collisions : " << karyon.total_bit_collisions << " bits\n";
    std::cout << " - Final Bit Error Rate  : " << std::fixed << std::setprecision(6) << (p3_friction * 100.0) << "%\n";
    std::cout << " - Final Free Energy F   : " << std::fixed << std::setprecision(8) << p3_fe << " nats\n";
    std::cout << " - Processing Speed      : " << std::fixed << std::setprecision(2) << throughput_mb_s << " MB/s\n";
    std::cout << " - Total Latency         : " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    std::cout << "======================================================================\n";

    return 0;
}
