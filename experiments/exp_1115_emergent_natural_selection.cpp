// ============================================================================
// KARYON-CORE EXP-1115: UNCONSTRAINED NATURAL SELECTION OF EMERGENT ARCHITECTURES
// Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist
// Standard: KEP v40.0 Master, Principle 0 (Bazilevs' Dream) & Principle 13 (Zero Imposed Abstractions)
//
// PHILOSOPHICAL REVOLUTION:
// Stop imposing human-designed architectures (no predetermined taps, no fixed loops,
// no pre-baked tree depths, no hardcoded gas laws, no fixed registers).
//
// THE GENOMIC SUBSTRATE OF PRISTINE PHYSICAL DYNAMICS:
// An autopoietic population of primitive computational organisms.
// Each organism is represented by an open-ended dynamic bitwise genome:
//   - Primitive bitwise operators on 64-bit hardware registers (XOR, AND, OR, SHIFT, ROT, POPCOUNT, ADD, NOT)
//   - Self-governed dynamic memory sprouting rules (when to allocate, what to index, how to retain)
//   - Self-governed somatic threshold regulation
//
// NATURAL SELECTION MECHANISM:
// - Environmental Friction & Free Energy (F = Errors + Structural Complexity).
// - Zero human-guided fitness functions: organisms that fail to stabilize in the physical
//   environment accumulate friction and physically dissolve.
// - Survival of the fittest: Survivors reproduce, recombine emergent operators,
//   and sprout higher-order architectural principles entirely bottom-up.
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
#include <sstream>

#pragma pack(push, 1)
struct KCoreManifestV63 {
    char magic[8];             // "KCOREv63"
    uint64_t timestamp_epoch;
    uint32_t generation;
    uint32_t population_size;
    uint64_t best_sprouted_nodes;
    double best_free_energy;
    double throughput_mb_s;
    char reserved[28];
};
#pragma pack(pop)

// Fundamental Hardware Bit-Primitive Opcodes (The Universal Substrate)
enum BitOp : uint8_t {
    OP_XOR_SHIFT_RIGHT = 0,
    OP_XOR_SHIFT_LEFT  = 1,
    OP_AND_ROTATE      = 2,
    OP_OR_SHIFT_RIGHT  = 3,
    OP_ADD_WRAPPED     = 4,
    OP_NOT_XOR         = 5,
    OP_BIT_REVERSE     = 6,
    OP_POPCOUNT_MASK   = 7,
    NUM_OPS            = 8
};

struct GeneInstruction {
    uint8_t op;       // BitOp
    uint8_t shift1;   // 1..63
    uint8_t shift2;   // 1..63
    uint64_t mask;    // Bitmask for projection
};

struct OrganismGenome {
    std::vector<GeneInstruction> instructions; // Dynamic instruction chain (variable length!)
    uint8_t memory_sprout_condition;          // 0: always on error, 1: on error > threshold, 2: on bit pattern match
    uint8_t sprout_threshold;                 // Bit error threshold for memory sprouting
    uint8_t feedback_shift;                   // How environment feedback integrates back into state
};

class EmergentOrganism {
public:
    OrganismGenome genome;
    std::unordered_map<uint64_t, uint64_t> memory_arena;

    uint64_t total_words_tested;
    uint64_t total_bit_collisions;
    double free_energy;

    EmergentOrganism() : total_words_tested(0), total_bit_collisions(0), free_energy(1.0) {}

    // Execute the organism's emergent bitwise genome to transform hardware register
    __attribute__((always_inline))
    inline uint64_t ComputeAddress(uint64_t reg) const {
        uint64_t acc = reg;
        for (const auto &gene : genome.instructions) {
            switch (gene.op) {
                case OP_XOR_SHIFT_RIGHT:
                    acc ^= (acc >> (gene.shift1 & 63)) ^ (acc >> (gene.shift2 & 63));
                    break;
                case OP_XOR_SHIFT_LEFT:
                    acc ^= (acc << (gene.shift1 & 63));
                    break;
                case OP_AND_ROTATE: {
                    uint8_t s = gene.shift1 & 63;
                    acc = (acc & gene.mask) ^ ((acc >> s) | (acc << (64 - s)));
                    break;
                }
                case OP_OR_SHIFT_RIGHT:
                    acc ^= (acc | gene.mask) >> (gene.shift1 & 63);
                    break;
                case OP_ADD_WRAPPED:
                    acc = (acc + gene.mask) ^ (acc >> (gene.shift1 & 63));
                    break;
                case OP_NOT_XOR:
                    acc = (~acc) ^ (acc >> (gene.shift1 & 63));
                    break;
                case OP_BIT_REVERSE:
                    acc = __builtin_bswap64(acc) ^ gene.mask;
                    break;
                case OP_POPCOUNT_MASK: {
                    uint64_t count = __builtin_popcountll(acc & gene.mask);
                    acc ^= (acc << (count & 15));
                    break;
                }
            }
        }
        return acc;
    }

    __attribute__((always_inline))
    inline uint64_t ProcessWord(uint64_t target_word, uint64_t &reg, bool evolve) {
        uint64_t addr = ComputeAddress(reg);

        auto it = memory_arena.find(addr);
        uint64_t pred = (it != memory_arena.end()) ? it->second : 0ULL;

        uint64_t diff = pred ^ target_word;
        int err = __builtin_popcountll(diff);

        total_words_tested++;
        total_bit_collisions += err;

        if (evolve) {
            bool should_sprout = false;
            switch (genome.memory_sprout_condition) {
                case 0:
                    should_sprout = (err > 0);
                    break;
                case 1:
                    should_sprout = (err >= genome.sprout_threshold);
                    break;
                case 2:
                    should_sprout = (err > 0) && ((addr & 1ULL) == 1ULL);
                    break;
            }

            if (should_sprout) {
                memory_arena[addr] = target_word;
            }
        }

        // Feedback shift into hardware register
        reg = (reg << (genome.feedback_shift & 7)) ^ (target_word & 0xFFULL);

        return pred;
    }

    void FinalizeEvaluation() {
        uint64_t total_bits = total_words_tested * 64;
        if (total_bits == 0) {
            free_energy = 1.0;
            return;
        }
        double r = (double)total_bit_collisions / (double)total_bits;
        if (r <= 0.0) {
            // Free energy penalizes unnecessary genome complexity (Occam's razor / Minimum Description Length)
            free_energy = 1e-9 * genome.instructions.size();
        } else if (r >= 1.0) {
            free_energy = 1.0;
        } else {
            double entropy = -(r * std::log2(r) + (1.0 - r) * std::log2(1.0 - r));
            double complexity_penalty = 0.0001 * genome.instructions.size();
            free_energy = entropy + complexity_penalty;
        }
    }

    void ResetEvaluation() {
        total_words_tested = 0;
        total_bit_collisions = 0;
        free_energy = 1.0;
    }

    void ClearMemory() {
        memory_arena.clear();
    }
};

// Evolution Engine: Mutation, Recombination, and Natural Selection
class EmergentEvolutionEngine {
public:
    std::vector<EmergentOrganism> population;
    std::mt19937_64 rng;

    EmergentEvolutionEngine(size_t pop_size, uint64_t seed = 42) : rng(seed) {
        population.resize(pop_size);
        for (auto &org : population) {
            // Start with minimal random genome (2 to 5 instructions)
            size_t num_genes = 2 + (rng() % 4);
            for (size_t g = 0; g < num_genes; ++g) {
                GeneInstruction gene;
                gene.op = static_cast<uint8_t>(rng() % NUM_OPS);
                gene.shift1 = static_cast<uint8_t>(1 + (rng() % 62));
                gene.shift2 = static_cast<uint8_t>(1 + (rng() % 62));
                gene.mask = rng();
                org.genome.instructions.push_back(gene);
            }
            org.genome.memory_sprout_condition = static_cast<uint8_t>(rng() % 3);
            org.genome.sprout_threshold = static_cast<uint8_t>(1 + (rng() % 32));
            org.genome.feedback_shift = static_cast<uint8_t>(1 + (rng() % 4));
        }
    }

    void Mutate(EmergentOrganism &org) {
        int action = rng() % 5;
        if (action == 0 && org.genome.instructions.size() < 12) {
            // Sprout a new gene (Expansion mutation)
            GeneInstruction gene;
            gene.op = static_cast<uint8_t>(rng() % NUM_OPS);
            gene.shift1 = static_cast<uint8_t>(1 + (rng() % 62));
            gene.shift2 = static_cast<uint8_t>(1 + (rng() % 62));
            gene.mask = rng();
            org.genome.instructions.push_back(gene);
        } else if (action == 1 && org.genome.instructions.size() > 1) {
            // Prune an unfit gene (Compression mutation)
            size_t idx = rng() % org.genome.instructions.size();
            org.genome.instructions.erase(org.genome.instructions.begin() + idx);
        } else if (action == 2 && !org.genome.instructions.empty()) {
            // Mutate existing instruction
            size_t idx = rng() % org.genome.instructions.size();
            org.genome.instructions[idx].op = static_cast<uint8_t>(rng() % NUM_OPS);
            org.genome.instructions[idx].shift1 = static_cast<uint8_t>(1 + (rng() % 62));
            org.genome.instructions[idx].shift2 = static_cast<uint8_t>(1 + (rng() % 62));
            org.genome.instructions[idx].mask ^= (1ULL << (rng() % 64));
        } else if (action == 3) {
            // Mutate memory sprouting rule
            org.genome.memory_sprout_condition = static_cast<uint8_t>(rng() % 3);
            org.genome.sprout_threshold = static_cast<uint8_t>(1 + (rng() % 32));
        } else {
            // Mutate feedback dynamics
            org.genome.feedback_shift = static_cast<uint8_t>(1 + (rng() % 4));
        }
    }

    EmergentOrganism Crossover(const EmergentOrganism &p1, const EmergentOrganism &p2) {
        EmergentOrganism child;
        child.genome.memory_sprout_condition = (rng() % 2 == 0) ? p1.genome.memory_sprout_condition : p2.genome.memory_sprout_condition;
        child.genome.sprout_threshold = (rng() % 2 == 0) ? p1.genome.sprout_threshold : p2.genome.sprout_threshold;
        child.genome.feedback_shift = (rng() % 2 == 0) ? p1.genome.feedback_shift : p2.genome.feedback_shift;

        size_t min_len = std::min(p1.genome.instructions.size(), p2.genome.instructions.size());
        size_t max_len = std::max(p1.genome.instructions.size(), p2.genome.instructions.size());
        size_t child_len = min_len + (rng() % (max_len - min_len + 1));

        for (size_t i = 0; i < child_len; ++i) {
            if (i < p1.genome.instructions.size() && (rng() % 2 == 0 || i >= p2.genome.instructions.size())) {
                child.genome.instructions.push_back(p1.genome.instructions[i]);
            } else if (i < p2.genome.instructions.size()) {
                child.genome.instructions.push_back(p2.genome.instructions[i]);
            }
        }
        return child;
    }

    void NaturalSelection() {
        // Sort ascending by Free Energy (Lowest Free Energy = Most Fit)
        std::sort(population.begin(), population.end(), [](const EmergentOrganism &a, const EmergentOrganism &b) {
            return a.free_energy < b.free_energy;
        });

        // Top 20% survive (Elite breeding pool)
        size_t elite_count = population.size() / 5;
        std::vector<EmergentOrganism> next_gen;
        next_gen.reserve(population.size());

        for (size_t i = 0; i < elite_count; ++i) {
            EmergentOrganism survivor = population[i];
            survivor.ResetEvaluation();
            survivor.ClearMemory();
            next_gen.push_back(survivor);
        }

        // Fill remaining 80% via Crossover and Mutation
        while (next_gen.size() < population.size()) {
            size_t p1 = rng() % elite_count;
            size_t p2 = rng() % elite_count;
            EmergentOrganism child = Crossover(population[p1], population[p2]);
            if (rng() % 100 < 70) { // 70% mutation rate
                Mutate(child);
            }
            child.ResetEvaluation();
            child.ClearMemory();
            next_gen.push_back(child);
        }

        population = std::move(next_gen);
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

std::string DescribeGenome(const EmergentOrganism &org) {
    std::stringstream ss;
    ss << "Ops(" << org.genome.instructions.size() << "): [";
    for (size_t i = 0; i < org.genome.instructions.size(); ++i) {
        const auto &g = org.genome.instructions[i];
        switch (g.op) {
            case OP_XOR_SHIFT_RIGHT: ss << "XOR_SHR(" << (int)g.shift1 << "," << (int)g.shift2 << ")"; break;
            case OP_XOR_SHIFT_LEFT:  ss << "XOR_SHL(" << (int)g.shift1 << ")"; break;
            case OP_AND_ROTATE:      ss << "AND_ROT(" << (int)g.shift1 << ")"; break;
            case OP_OR_SHIFT_RIGHT:  ss << "OR_SHR(" << (int)g.shift1 << ")"; break;
            case OP_ADD_WRAPPED:     ss << "ADD_WRAP(" << (int)g.shift1 << ")"; break;
            case OP_NOT_XOR:         ss << "NOT_XOR(" << (int)g.shift1 << ")"; break;
            case OP_BIT_REVERSE:     ss << "BSWAP"; break;
            case OP_POPCOUNT_MASK:   ss << "POPCNT_SHL"; break;
        }
        if (i + 1 < org.genome.instructions.size()) ss << " -> ";
    }
    ss << "] | SproutRule: " << (int)org.genome.memory_sprout_condition
       << " | Thresh: " << (int)org.genome.sprout_threshold
       << " | FbShift: " << (int)org.genome.feedback_shift;
    return ss.str();
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << " KARYON-CORE EXP-1115: UNCONSTRAINED NATURAL SELECTION OF ARCHITECTURES\n";
    std::cout << " Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist\n";
    std::cout << " Standard: KEP Principle 0 (Zero Imposed Concepts) & Principle 13 (Silicon Speed)\n";
    std::cout << " Paradigm: Spontaneous Open-Ended Evolution of Bit-Level Computational Organisms\n";
    std::cout << "======================================================================\n\n";

    // 1. INGEST MULTI-STAGE MACRO ENVIRONMENT DATASTREAM
    std::vector<uint64_t> stream1 = LoadFileAs64BitWords("KARYON_PHILOSOPHICAL_FOUNDATIONS.md");
    std::vector<uint64_t> stream2 = LoadFileAs64BitWords("KEP.md");
    std::vector<uint64_t> stream3 = LoadFileAs64BitWords("experiments/exp_1114_ashby_ultrastable_two_loop_substrate.cpp");

    std::vector<uint64_t> macro_stream;
    macro_stream.insert(macro_stream.end(), stream1.begin(), stream1.end());
    macro_stream.insert(macro_stream.end(), stream2.begin(), stream2.end());
    macro_stream.insert(macro_stream.end(), stream3.begin(), stream3.end());

    size_t num_words = macro_stream.size();
    size_t total_bits = num_words * 64;

    std::cout << " [1] Physical Environment Stream Ingested:\n";
    std::cout << "     - Datastream Components : Foundations + KEP.md + exp_1114 C++ Code\n";
    std::cout << "     - 64-bit Word Count     : " << num_words << " words\n";
    std::cout << "     - Continuous Bit Count  : " << total_bits << " physical bits ("
              << std::fixed << std::setprecision(2) << (double)total_bits / (8.0 * 1024.0) << " KB)\n\n";

    // 2. INITIALIZE TABULA RASA POPULATION
    const size_t POPULATION_SIZE = 50;
    const size_t GENERATIONS = 30;

    std::cout << " [2] Inoculating Tabula Rasa Population:\n";
    std::cout << "     - Population Size : " << POPULATION_SIZE << " primitive organisms\n";
    std::cout << "     - Generations     : " << GENERATIONS << " evolutionary epochs\n";
    std::cout << "     - Starting State  : Random opcode genomes, zero human architecture\n\n";

    EmergentEvolutionEngine evolution(POPULATION_SIZE, 2026);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t gen = 1; gen <= GENERATIONS; ++gen) {
        // Evaluate all organisms on the physical environment
        for (auto &org : evolution.population) {
            org.ResetEvaluation();
            org.ClearMemory();

            // Autopoietic Adaptation Pass
            uint64_t reg = 0;
            for (size_t i = 0; i < num_words; ++i) {
                org.ProcessWord(macro_stream[i], reg, true);
            }

            // Pure Conduction Evaluation Pass (Free Energy Measurement)
            org.ResetEvaluation();
            reg = 0;
            for (size_t i = 0; i < num_words; ++i) {
                org.ProcessWord(macro_stream[i], reg, false);
            }
            org.FinalizeEvaluation();
        }

        // Audit best organism of current generation
        auto best_it = std::min_element(evolution.population.begin(), evolution.population.end(),
            [](const EmergentOrganism &a, const EmergentOrganism &b) {
                return a.free_energy < b.free_energy;
            });

        double best_error_pct = ((double)best_it->total_bit_collisions / (double)total_bits) * 100.0;

        std::cout << " Gen " << std::setw(2) << gen 
                  << " | Best F = " << std::fixed << std::setprecision(8) << best_it->free_energy << " nats"
                  << " | Collisions = " << std::setw(6) << best_it->total_bit_collisions << " / " << total_bits
                  << " (" << std::fixed << std::setprecision(3) << best_error_pct << "%)"
                  << " | Sprouted Memory = " << best_it->memory_arena.size() << " nodes\n"
                  << "        Emergent Architecture: " << DescribeGenome(*best_it) << "\n";

        if (gen < GENERATIONS) {
            evolution.NaturalSelection();
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Final Best Organism Inspection
    auto final_best = std::min_element(evolution.population.begin(), evolution.population.end(),
        [](const EmergentOrganism &a, const EmergentOrganism &b) {
            return a.free_energy < b.free_energy;
        });

    double total_processed_mb = ((double)total_bits * 2.0 * POPULATION_SIZE * GENERATIONS / 8.0) / (1024.0 * 1024.0);
    double throughput_mb_s = total_processed_mb / (elapsed_ms / 1000.0);

    // Serialize Winning Emergent Architecture into .kcore v63
    std::string kcore_file = "build/karyon_exp1115_emergent_evolution.kcore";
    std::ofstream out(kcore_file, std::ios::binary);
    if (out.is_open()) {
        KCoreManifestV63 manifest;
        std::memcpy(manifest.magic, "KCOREv63", 8);
        manifest.timestamp_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        manifest.generation = GENERATIONS;
        manifest.population_size = POPULATION_SIZE;
        manifest.best_sprouted_nodes = final_best->memory_arena.size();
        manifest.best_free_energy = final_best->free_energy;
        manifest.throughput_mb_s = throughput_mb_s;
        std::memset(manifest.reserved, 0, sizeof(manifest.reserved));
        out.write(reinterpret_cast<const char*>(&manifest), sizeof(manifest));
        out.close();
    }

    std::cout << "\n======================================================================\n";
    std::cout << " EXP-1115 NATURAL SELECTION TELEMETRY SUMMARY:\n";
    std::cout << " - Total Population Evaluated  : " << (POPULATION_SIZE * GENERATIONS) << " organism instances\n";
    std::cout << " - Total Bits Processed        : " << (total_bits * 2 * POPULATION_SIZE * GENERATIONS) << " physical bits\n";
    std::cout << " - Winning Architecture Nodes  : " << final_best->memory_arena.size() << " sprouted nodes\n";
    std::cout << " - Final Bit Discrepancy       : " << final_best->total_bit_collisions << " / " << total_bits << " bits ("
              << std::fixed << std::setprecision(6) << (((double)final_best->total_bit_collisions / (double)total_bits) * 100.0) << "%)\n";
    std::cout << " - Winning Free Energy F       : " << std::fixed << std::setprecision(8) << final_best->free_energy << " nats\n";
    std::cout << " - Evolution Throughput Speed  : " << std::fixed << std::setprecision(2) << throughput_mb_s << " MB/s ("
              << std::fixed << std::setprecision(2) << (throughput_mb_s * 8.0) << " Mbit/s)\n";
    std::cout << " - Total Execution Time        : " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    std::cout << " - Synthesized Architecture    : " << DescribeGenome(*final_best) << "\n";
    std::cout << "======================================================================\n";

    return 0;
}
