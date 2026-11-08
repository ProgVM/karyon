// ============================================================================
// KARYON-CORE EXP-1114: SOVEREIGN ASHBY ULTRASTABLE TWO-LOOP SUBSTRATE (V=2)
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v40.0 Sovereign Master, KARYON_PHILOSOPHICAL_FOUNDATIONS.md
//
// CYBERNETIC PRINCIPLES (W. ROSS ASHBY - DESIGN FOR A BRAIN):
// 1. ERROR AS SOMATIC STRESS (Free Energy F_t): Error is not a scalar gradient,
//    but an integral homeostatic stress S_t = F_t signaling structural inadequacy.
// 2. TWO-LOOP ULTRASTABILITY:
//    - FIRST LOOP (Fast Conductance): While stress S_t <= S_target, the substrate
//      uses existing phase trajectories without memory allocation (zero energy cost).
//    - SECOND LOOP (Topological Sprouting & Reorganization): When stress S_t > S_target,
//      homeostatic barrier is breached. Scalar stress acts as a somatic motive to
//      sprout new phase nodes and reorganize local topology.
// 3. ASHBY'S LAW OF REQUISITE VARIETY: The internal topological variety V_internal
//    (sprouted nodes & depth) dynamically matches the environmental entropy H_env.
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
#include <unordered_map>
#include <random>

#pragma pack(push, 1)
struct KCoreManifestV62 {
    char magic[8];             // "KCOREv62"
    uint64_t timestamp_epoch;  // Epoch nanoseconds
    uint64_t sprouted_nodes;   // Internal topological variety (V_internal)
    double final_stress_S;     // Final Somatic Stress S_t
    double free_energy_nats;   // Final Free Energy
    double requisite_variety;  // Ratio V_internal / H_env
    char reserved[32];         // Alignment padding
};
#pragma pack(pop)

class AshbyUltrastableSubstrate {
public:
    // Memory nodes sprout ONLY when Second Loop is activated
    std::unordered_map<uint64_t, uint64_t> topology_nodes;

    // Homeostatic Variables (Ashby Somatic System)
    double somatic_stress_S;   // Current integral scalar stress S_t
    double stress_target;      // Homeostatic baseline S_target (e.g. 0.05)
    double plasticity_melt;    // Dynamic melting rate (Second Loop activity)

    // Performance Counters
    uint64_t total_words;
    uint64_t total_bit_errors;
    uint64_t loop1_conductance_count;
    uint64_t loop2_sprout_count;

    AshbyUltrastableSubstrate(double target_stress = 0.05) 
        : somatic_stress_S(1.0), stress_target(target_stress), plasticity_melt(0.0),
          total_words(0), total_bit_errors(0), loop1_conductance_count(0), loop2_sprout_count(0) {
        topology_nodes.reserve(500000);
    }

    __attribute__((always_inline))
    inline uint64_t Conduct64(uint64_t target_word, uint64_t &reg) {
        // Compute 64-bit hardware bit-tap context signature
        uint64_t tap_signature = reg ^ (reg >> 7) ^ (reg >> 13) ^ (reg >> 19) ^ (reg >> 31);

        auto it = topology_nodes.find(tap_signature);
        uint64_t predicted_word = (it != topology_nodes.end()) ? it->second : 0ULL;

        // Compute bit discrepancy (Hardware POPCOUNT)
        uint64_t diff = predicted_word ^ target_word;
        int bit_errors = __builtin_popcountll(diff);

        total_words++;
        total_bit_errors += bit_errors;

        // Instantly update Somatic Stress S_t (Exponential Moving Average)
        double instantaneous_error_rate = (double)bit_errors / 64.0;
        somatic_stress_S = 0.95 * somatic_stress_S + 0.05 * instantaneous_error_rate;

        // ====================================================================
        // ASHBY TWO-LOOP ULTRASTABILITY MECHANISM
        // ====================================================================
        if (somatic_stress_S <= stress_target) {
            // --- FIRST LOOP: Fast Conductance (Homeostatic Balance) ---
            // System is in homeostatic equilibrium. Zero memory sprouting.
            loop1_conductance_count++;
            plasticity_melt = std::max(0.0, plasticity_melt - 0.01);
        } else {
            // --- SECOND LOOP: Evolutionary Reorganization (Homeostatic Stress Breach) ---
            // Scalar stress acts as somatic motive to melt topology and sprout nodes.
            loop2_sprout_count++;
            plasticity_melt = std::min(1.0, plasticity_melt + 0.05);

            // Sprout or update topological node in response to somatic tension
            topology_nodes[tap_signature] = target_word;
        }

        // Shift 64-bit hardware register
        reg = (reg << 1) | (target_word & 1ULL);

        return predicted_word;
    }

    void ResetMetrics() {
        total_words = 0;
        total_bit_errors = 0;
        loop1_conductance_count = 0;
        loop2_sprout_count = 0;
    }

    double GetFrictionRate() const {
        uint64_t total_bits = total_words * 64;
        if (total_bits == 0) return 0.0;
        return (double)total_bit_errors / (double)total_bits;
    }

    double ComputeFreeEnergy() const {
        double r = GetFrictionRate();
        if (r <= 0.0) return 0.0;
        if (r >= 1.0) return 1.0;
        return -(r * std::log2(r) + (1.0 - r) * std::log2(1.0 - r));
    }

    bool Save(const std::string &filepath) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) return false;

        KCoreManifestV62 manifest;
        std::memcpy(manifest.magic, "KCOREv62", 8);
        manifest.timestamp_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        manifest.sprouted_nodes = topology_nodes.size();
        manifest.final_stress_S = somatic_stress_S;
        manifest.free_energy_nats = ComputeFreeEnergy();
        manifest.requisite_variety = (double)topology_nodes.size();
        std::memset(manifest.reserved, 0, sizeof(manifest.reserved));

        out.write(reinterpret_cast<const char*>(&manifest), sizeof(manifest));
        for (const auto &pair : topology_nodes) {
            out.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
            out.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
        }

        out.close();
        return true;
    }
};

std::vector<uint64_t> LoadFileAs64BitWords(const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary);
    std::vector<uint64_t> words;
    if (!file.is_open()) return words;

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    size_t num_words = (bytes.size() + 7) / 8;
    words.resize(num_words, 0ULL);

    for (size_t i = 0; i < bytes.size(); ++i) {
        size_t word_idx = i / 8;
        size_t byte_shift = (i % 8) * 8;
        words[word_idx] |= ((uint64_t)bytes[i] << byte_shift);
    }
    return words;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << " KARYON-CORE EXP-1114: SOVEREIGN ASHBY ULTRASTABLE TWO-LOOP SUBSTRATE\n";
    std::cout << " Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist\n";
    std::cout << " Theoretical Compass: W. Ross Ashby (Design for a Brain & Requisite Variety)\n";
    std::cout << " Paradigm: Somatic Stress S_t -> Two-Loop Ultrastable Dynamic Adaptation\n";
    std::cout << "======================================================================\n\n";

    // 1. INGEST ENVIRONMENT DATASTREAM
    std::vector<uint64_t> macro_stream = LoadFileAs64BitWords("KARYON_PHILOSOPHICAL_FOUNDATIONS.md");
    size_t num_words = macro_stream.size();
    size_t total_bits = num_words * 64;

    std::cout << " [1] Physical Environment Stream Ingested:\n";
    std::cout << "     - Source Stream        : KARYON_PHILOSOPHICAL_FOUNDATIONS.md\n";
    std::cout << "     - 64-bit Word Count    : " << num_words << " words\n";
    std::cout << "     - Continuous Bit Count : " << total_bits << " physical bits ("
              << std::fixed << std::setprecision(2) << (double)total_bits / (8.0 * 1024.0) << " KB)\n\n";

    // Initialize Substrate with Homeostatic Target Stress S_target = 0.01
    AshbyUltrastableSubstrate karyon(0.01);
    uint64_t hw_reg = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // TEST 1: CONTINUOUS RELAXATION & TWO-LOOP REGULATION AUDIT
    // ========================================================================
    std::cout << " ===> TEST 1: CONTINUOUS RELAXATION & TWO-LOOP ULTRASTABILITY <===\n";

    int pass = 1;
    while (pass <= 20) {
        karyon.ResetMetrics();
        hw_reg = 0;
        for (size_t i = 0; i < num_words; ++i) {
            karyon.Conduct64(macro_stream[i], hw_reg);
        }
        double friction = karyon.GetFrictionRate();
        double fe = karyon.ComputeFreeEnergy();

        std::cout << " ---> Pass " << std::setw(2) << pass 
                  << " | Stress S_t = " << std::fixed << std::setprecision(6) << karyon.somatic_stress_S
                  << " | Friction = " << std::fixed << std::setprecision(4) << (friction * 100.0) << "%"
                  << " | F = " << std::fixed << std::setprecision(6) << fe << " nats"
                  << " | Loop1 (Fast) = " << karyon.loop1_conductance_count
                  << " | Loop2 (Sprout) = " << karyon.loop2_sprout_count
                  << " | Nodes = " << karyon.topology_nodes.size() << "\n";

        if (karyon.total_bit_collisions == 0) {
            std::cout << "\n      [+] PERFECT HOMEOSTATIC EQUILIBRIUM (S_t -> 0, F = 0) REACHED AT PASS " << pass << "!\n\n";
            break;
        }
        pass++;
    }

    // ========================================================================
    // TEST 2: ENVIRONMENTAL CHAOS & REQUISITE VARIETY SURGE (ASHBY LAW TEST)
    // ========================================================================
    std::cout << " ===> TEST 2: ASHBY LAW OF REQUISITE VARIETY UNDER NOISE INJECTION <===\n";
    std::cout << "     Injecting 15% physical bit noise to test homeostatic stress response and loop switching...\n";

    std::vector<uint64_t> noisy_stream = macro_stream;
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < noisy_stream.size(); ++i) {
        if (rng() % 100 < 15) { // 15% word noise
            noisy_stream[i] ^= (1ULL << (rng() % 64));
        }
    }

    karyon.ResetMetrics();
    hw_reg = 0;
    for (size_t i = 0; i < noisy_stream.size(); ++i) {
        karyon.Conduct64(noisy_stream[i], hw_reg);
    }
    double noisy_friction = karyon.GetFrictionRate();
    double noisy_fe = karyon.ComputeFreeEnergy();

    std::cout << "      Noisy Environment Execution Summary:\n";
    std::cout << "      - Somatic Stress S_t     : " << std::fixed << std::setprecision(6) << karyon.somatic_stress_S << "\n";
    std::cout << "      - Residual Bit Friction  : " << std::fixed << std::setprecision(4) << (noisy_friction * 100.0) << "%\n";
    std::cout << "      - Free Energy F          : " << std::fixed << std::setprecision(6) << noisy_fe << " nats\n";
    std::cout << "      - Loop 1 (Fast) Words    : " << karyon.loop1_conductance_count << "\n";
    std::cout << "      - Loop 2 (Sprouting) Words: " << karyon.loop2_sprout_count << "\n";
    std::cout << "      - Total Sprouted Variety : " << karyon.topology_nodes.size() << " nodes\n\n";

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    double total_processed_mb = ((double)total_bits * (double)(pass + 1) / 8.0) / (1024.0 * 1024.0);
    double throughput_mb_s = total_processed_mb / (elapsed_ms / 1000.0);

    std::string kcore_file = "build/karyon_exp1114_ashby_ultrastable.kcore";
    karyon.Save(kcore_file);

    std::cout << "======================================================================\n";
    std::cout << " EXP-1114 ASHBY ULTRASTABLE TELEMETRY SUMMARY:\n";
    std::cout << " - Total Bits Processed     : " << (total_bits * (pass + 1)) << " physical bits\n";
    std::cout << " - Final Somatic Stress S_t  : " << std::fixed << std::setprecision(6) << karyon.somatic_stress_S << "\n";
    std::cout << " - Final Free Energy F      : " << std::fixed << std::setprecision(8) << karyon.ComputeFreeEnergy() << " nats\n";
    std::cout << " - Internal Variety (Nodes) : " << karyon.topology_nodes.size() << " autopoietic nodes\n";
    std::cout << " - Throughput Conduction    : " << std::fixed << std::setprecision(2) << throughput_mb_s << " MB/s ("
              << std::fixed << std::setprecision(2) << (throughput_mb_s * 8.0) << " Mbit/s)\n";
    std::cout << " - Total Execution Latency  : " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    std::cout << "======================================================================\n";

    return 0;
}
