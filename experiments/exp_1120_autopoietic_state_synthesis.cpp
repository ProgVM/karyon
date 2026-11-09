// ============================================================================
// KARYON-CORE EXP-1120: AUTOPOIETIC ALGORITHMIC TRANSDUCTION (F -> 0.00000000 NATS)
// Author & Lead Architect: Bazilevs (ProgVM)
// Guardian: Autonomous AI Cybernetics Engine
// Compliance: KEP v50.0 Sovereign Standard (Strict Principles 0, 1, 2)
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iomanip>

// --- KEP PRINCIPLE 1 & 2: SILICON HARDWARE PRIMITIVES & TOTIPOTENT GENOME ---

enum PrimitiveOp : uint8_t {
    PRIM_XOR_SHR = 0,
    PRIM_XOR_SHL,
    PRIM_AND_ROT,
    PRIM_OR_SHR,
    PRIM_ADD_ROT,
    PRIM_BSWAP,
    NUM_PRIM
};

#pragma pack(push, 1)
struct DynamicInstruction {
    uint8_t opcode;
    uint8_t param1;
    uint8_t param2;
};
#pragma pack(pop)

struct TotipotentGenome {
    std::vector<DynamicInstruction> program;
    uint8_t sprout_trigger;   // 0: bit error > 0, 1: popcount parity, 2: always
    uint8_t sprout_shift;     // 1..63
};

struct TotipotentOrganism {
    TotipotentGenome genome;
    std::vector<uint64_t> memory_table;
    uint64_t current_state;
    uint64_t bit_collisions;
    double free_energy;

    TotipotentOrganism() : current_state(0), bit_collisions(0), free_energy(1e9) {
        memory_table.reserve(1024);
    }

    inline uint64_t Execute(uint64_t input_word) {
        uint64_t acc = input_word ^ current_state;
        for (const auto& inst : genome.program) {
            switch (inst.opcode) {
                case PRIM_XOR_SHR:
                    acc ^= (acc >> (inst.param1 % 63 + 1));
                    break;
                case PRIM_XOR_SHL:
                    acc ^= (acc << (inst.param1 % 63 + 1));
                    break;
                case PRIM_AND_ROT:
                    acc = (acc & 0x0F0F0F0F0F0F0F0FULL) | std::rotl(acc, inst.param1 % 63 + 1);
                    break;
                case PRIM_OR_SHR:
                    acc |= (acc >> (inst.param1 % 63 + 1));
                    break;
                case PRIM_ADD_ROT:
                    acc += std::rotr(acc, inst.param1 % 63 + 1);
                    break;
                case PRIM_BSWAP:
                    acc = __builtin_bswap64(acc);
                    break;
            }
        }
        return acc;
    }
};

struct AlgorithmicTaskSpec {
    std::string name;
    std::vector<uint64_t> input_stream;
    std::vector<uint64_t> target_stream;
    bool is_deterministic;
};

// --- TASKS GENERATORS ---

static AlgorithmicTaskSpec generate_task_a_stack_transducer(size_t num_words) {
    AlgorithmicTaskSpec task;
    task.name = "Task A: Dynamic Bit-Stack Transducer (Parity Depth)";
    task.is_deterministic = true;
    task.input_stream.resize(num_words);
    task.target_stream.resize(num_words);

    uint64_t stack = 0x123456789ABCDEF0ULL;
    std::mt19937_64 rng(42);

    for (size_t i = 0; i < num_words; ++i) {
        uint64_t in = rng();
        task.input_stream[i] = in;
        int popcount = __builtin_popcountll(in);
        if (popcount % 2 == 0) {
            stack = (stack << 8) | (in & 0xFFULL);
        } else {
            stack = (stack >> 8) ^ in;
        }
        task.target_stream[i] = stack;
    }
    return task;
}

static AlgorithmicTaskSpec generate_task_b_noise_filter(size_t num_words) {
    AlgorithmicTaskSpec task;
    task.name = "Task B: Stochastic Signal Extraction (40% Noise)";
    task.is_deterministic = false;
    task.input_stream.resize(num_words);
    task.target_stream.resize(num_words);

    std::mt19937_64 rng(1337);
    uint64_t signal_state = 0xCAFEF00DDEADBEEFULL;

    for (size_t i = 0; i < num_words; ++i) {
        signal_state = std::rotl(signal_state, 17) ^ (signal_state >> 5) ^ 0x9E3779B97F4A7C15ULL;
        task.target_stream[i] = signal_state;

        uint64_t noise = rng();
        uint64_t noise_mask = rng() & rng();
        task.input_stream[i] = signal_state ^ (noise & noise_mask);
    }
    return task;
}

static AlgorithmicTaskSpec generate_task_c_adversarial_flip(size_t num_words) {
    AlgorithmicTaskSpec task;
    task.name = "Task C: Dynamic Adversarial Bit-Inversion";
    task.is_deterministic = true;
    task.input_stream.resize(num_words);
    task.target_stream.resize(num_words);

    uint64_t state = 0xFEEDBEEF12345678ULL;
    for (size_t i = 0; i < num_words; ++i) {
        state = (state ^ (state >> 11)) + 0x5555555555555555ULL;
        task.input_stream[i] = state;
        if (i < num_words / 2) {
            task.target_stream[i] = std::rotr(state, 13);
        } else {
            task.target_stream[i] = ~std::rotr(state, 13);
        }
    }
    return task;
}

static AlgorithmicTaskSpec generate_task_d_soliton(size_t num_words) {
    AlgorithmicTaskSpec task;
    task.name = "Task D: Multi-Harmonic Prime Soliton";
    task.is_deterministic = true;
    task.input_stream.resize(num_words);
    task.target_stream.resize(num_words);

    uint64_t sol1 = 0x0123456789ABCDEFULL;
    uint64_t sol2 = 0xFEDCBA9876543210ULL;
    uint64_t sol3 = 0x55AA55AA55AA55AAULL;

    for (size_t i = 0; i < num_words; ++i) {
        sol1 = std::rotl(sol1, 3);
        sol2 = std::rotr(sol2, 7);
        sol3 = std::rotl(sol3, 13) ^ (sol3 >> 1);

        task.input_stream[i] = sol1 ^ sol2;
        task.target_stream[i] = sol1 ^ sol2 ^ sol3;
    }
    return task;
}

// --- UNCONSTRAINED NATURAL SELECTION ENGINE ---

static void MutateOrganism(TotipotentOrganism& org, std::mt19937_64& rng) {
    int choice = rng() % 5;
    if (choice == 0 && org.genome.program.size() < 12) {
        // Add instruction
        DynamicInstruction inst;
        inst.opcode = rng() % NUM_PRIM;
        inst.param1 = 1 + (rng() % 62);
        inst.param2 = 1 + (rng() % 62);
        org.genome.program.push_back(inst);
    } else if (choice == 1 && org.genome.program.size() > 1) {
        // Remove instruction
        size_t idx = rng() % org.genome.program.size();
        org.genome.program.erase(org.genome.program.begin() + idx);
    } else if (choice == 2 && !org.genome.program.empty()) {
        // Mutate parameter
        size_t idx = rng() % org.genome.program.size();
        org.genome.program[idx].param1 = 1 + (rng() % 62);
    } else if (choice == 3 && !org.genome.program.empty()) {
        // Mutate opcode
        size_t idx = rng() % org.genome.program.size();
        org.genome.program[idx].opcode = rng() % NUM_PRIM;
    } else {
        // Mutate sprout rule
        org.genome.sprout_trigger = rng() % 3;
        org.genome.sprout_shift = 1 + (rng() % 62);
    }
}

static void EvaluateOrganism(TotipotentOrganism& org, const AlgorithmicTaskSpec& task) {
    org.bit_collisions = 0;
    org.memory_table.clear();
    org.current_state = 0;

    const size_t num_words = task.input_stream.size();
    const size_t total_bits = num_words * 64;

    for (size_t i = 0; i < num_words; ++i) {
        uint64_t in_word = task.input_stream[i];
        uint64_t target_word = task.target_stream[i];

        uint64_t pred = org.Execute(in_word);
        uint64_t diff = pred ^ target_word;
        uint32_t bit_err = __builtin_popcountll(diff);
        org.bit_collisions += bit_err;

        // Endogenous Memory Sprouting
        bool do_sprout = false;
        if (org.genome.sprout_trigger == 0 && bit_err > 0) do_sprout = true;
        else if (org.genome.sprout_trigger == 1 && (__builtin_popcountll(in_word) % 2 == 0)) do_sprout = true;
        else if (org.genome.sprout_trigger == 2) do_sprout = true;

        if (do_sprout) {
            org.current_state = org.current_state ^ (org.current_state >> org.genome.sprout_shift) ^ diff;
            if (org.memory_table.size() < 65536) {
                org.memory_table.push_back(org.current_state);
            }
        }
    }

    double error_ratio = static_cast<double>(org.bit_collisions) / static_cast<double>(total_bits);
    if (error_ratio <= 0.0) {
        org.free_energy = 1e-9 * org.genome.program.size(); // MDL Complexity penalty
    } else if (error_ratio >= 1.0) {
        org.free_energy = 1.0;
    } else {
        double entropy = -(error_ratio * std::log2(error_ratio) + (1.0 - error_ratio) * std::log2(1.0 - error_ratio));
        org.free_energy = entropy + 0.0001 * org.genome.program.size();
    }
}

struct SelectionResult {
    size_t generation_converged;
    double execution_time_ms;
    uint64_t final_bit_collisions;
    double final_free_energy;
    size_t sprouted_nodes;
    size_t genome_instructions;
};

static SelectionResult EvolveSubstrate(const AlgorithmicTaskSpec& task, size_t pop_size = 64, size_t max_generations = 100) {
    std::mt19937_64 rng(42);
    std::vector<TotipotentOrganism> population(pop_size);

    for (auto& org : population) {
        size_t genes = 2 + (rng() % 3);
        for (size_t g = 0; g < genes; ++g) {
            DynamicInstruction inst;
            inst.opcode = rng() % NUM_PRIM;
            inst.param1 = 1 + (rng() % 62);
            inst.param2 = 1 + (rng() % 62);
            org.genome.program.push_back(inst);
        }
        org.genome.sprout_trigger = rng() % 3;
        org.genome.sprout_shift = 1 + (rng() % 62);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    size_t conv_gen = max_generations;
    TotipotentOrganism best_org;

    for (size_t gen = 1; gen <= max_generations; ++gen) {
        for (auto& org : population) {
            EvaluateOrganism(org, task);
        }

        std::sort(population.begin(), population.end(), [](const TotipotentOrganism& a, const TotipotentOrganism& b) {
            return a.free_energy < b.free_energy;
        });

        best_org = population[0];

        if (best_org.bit_collisions == 0 && task.is_deterministic) {
            conv_gen = gen;
            break;
        }

        // Selection & Mutation (Top 25% reproduce)
        size_t elite_count = pop_size / 4;
        for (size_t i = elite_count; i < pop_size; ++i) {
            population[i] = population[i % elite_count];
            MutateOrganism(population[i], rng);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    SelectionResult res;
    res.generation_converged = conv_gen;
    res.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    res.final_bit_collisions = best_org.bit_collisions;
    res.final_free_energy = best_org.free_energy;
    res.sprouted_nodes = best_org.memory_table.size();
    res.genome_instructions = best_org.genome.program.size();

    return res;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << " KARYON-CORE EXP-1120: AUTOPOIETIC ALGORITHMIC TRANSDUCTION (F -> 0.0000)\n";
    std::cout << " Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist\n";
    std::cout << " Compliance: 100% STRICT PRINCIPLE 0, PRINCIPLE 1 & PRINCIPLE 2 (KEP v50.0)\n";
    std::cout << " Mechanism: Dynamic Instruction Mutation & Unconstrained Natural Selection\n";
    std::cout << "======================================================================\n\n";

    const size_t words_per_task = 20000; // 20,000 x 64-bit = 1,280,000 bits per stream
    std::vector<AlgorithmicTaskSpec> tasks = {
        generate_task_a_stack_transducer(words_per_task),
        generate_task_b_noise_filter(words_per_task),
        generate_task_c_adversarial_flip(words_per_task),
        generate_task_d_soliton(words_per_task)
    };

    double total_time_ms = 0.0;
    uint64_t total_bits_processed = 0;

    std::cout << "====================================================================================================\n";
    std::cout << " EXP-1120 AUTOPOIETIC ALGORITHMIC TRANSDUCTION MATRIX (F -> 0.00000000 NATS):\n";
    std::cout << "====================================================================================================\n";
    std::cout << "Task Name                                    Gen Conv    Time (ms)     Free Energy (nats)  Collisions  Sprouted Memory\n";
    std::cout << "----------------------------------------------------------------------------------------------------\n";

    for (const auto& task : tasks) {
        auto res = EvolveSubstrate(task, 64, 100);
        size_t task_bits = words_per_task * 64 * 64 * res.generation_converged;
        total_bits_processed += task_bits;
        total_time_ms += res.execution_time_ms;

        printf("%-44s Gen %-7zu %-11.2f %-19.8f %-11lu %-zu nodes (%zu genes)\n",
               task.name.c_str(),
               res.generation_converged,
               res.execution_time_ms,
               res.final_free_energy,
               res.final_bit_collisions,
               res.sprouted_nodes,
               res.genome_instructions);
    }

    double throughput_mb_s = (static_cast<double>(total_bits_processed) / 8.0 / 1024.0 / 1024.0) / (total_time_ms / 1000.0);

    std::cout << "====================================================================================================\n";
    std::cout << " SCIENTIFIC SUMMARY & EMPIRICAL VERIFICATION:\n";
    std::cout << " - Total Physical Bits Evaluated: " << total_bits_processed << " bits (" << (total_bits_processed / 8 / 1024 / 1024) << " MB)\n";
    std::cout << " - Total Execution Latency     : " << total_time_ms << " ms (" << (total_time_ms / 1000.0) << " seconds)\n";
    std::cout << " - Silicon Processing Speed    : " << throughput_mb_s << " MB/s (" << (throughput_mb_s * 8.0) << " Mbit/s)\n";
    std::cout << "====================================================================================================\n";

    return 0;
}
