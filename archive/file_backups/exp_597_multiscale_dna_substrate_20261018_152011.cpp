// ======================================================================================
// EXP-597: Multiscale Self-Assembling Substrate with Epigenetic DNA/RNA Resilience
// --------------------------------------------------------------------------------------
// KEP Protocol Master Benchmark:
// 1. Multiscale Separation of Mutation:
//    - Low Level (Micro-Engine): Stable, cache-aligned, ultra-fast deterministic
//      arithmetic & logic cells with sovereign local noise control. Micro-ops almost
//      NEVER mutate randomly, preserving causal computing stability.
//    - High Level (Macro-Router): Dynamic self-assembling routing, operator sprouting,
//      and topological restructuring driven by Free Energy minimization.
// 2. Sovereign Epigenetic DNA/RNA Blueprint & Catastrophic Regeneration:
//    - High-efficacy functional circuits are transcribed into an invariant DNA/RNA registry.
//    - Catastrophic event injection (50% state corruption / sudden distribution shock).
//    - Mode A: With DNA/RNA epigenetic transcription and automated regeneration.
//    - Mode B: Without DNA/RNA (ablated baseline), suffering catastrophic amnesia.
// 3. Ultra-Fast Native Execution:
//    - Cache-aligned, SIMD-friendly flat memory layout.
//    - Target: High throughput (100k+ steps/sec) with rigorous predictive convergence.
// ======================================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <random>
#include <iomanip>

constexpr size_t MICRO_CELLS     = 128;
constexpr size_t MACRO_NODES     = 16;
constexpr size_t VOCAB_SIZE      = 258;
constexpr size_t DNA_CAPACITY    = 32;

// Fast deterministic PRNG for non-deterministic controlled thermal noise
struct FastPRNG {
    uint64_t s[2];
    FastPRNG(uint64_t seed = 42) {
        s[0] = seed ^ 0x9E3779B97F4A7C15ULL;
        s[1] = (seed << 31) | (seed >> 33);
        if (s[0] == 0 && s[1] == 0) s[0] = 1;
    }
    inline uint64_t next() {
        uint64_t s1 = s[0];
        const uint64_t s0 = s[1];
        s[0] = s0;
        s1 ^= s1 << 23;
        s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        return s[1] + s0;
    }
    inline double next_double() {
        return (next() >> 11) * (1.0 / 9007199254740992.0);
    }
};

// 1. Micro-Level Stable Execution Unit (Does NOT mutate randomly)
struct alignas(64) MicroCell {
    uint64_t state;             // Discrete 64-bit register
    double   analog_val;        // Continuous relaxation state
    double   noise_temp;        // Sovereignly controlled noise temperature [0, 1]
    double   causal_trace;      // Sensitivity credit trace
};

// DNA/RNA Invariant Genetic Blueprint for low-level node reconstruction
struct DnaNodeGene {
    uint8_t  node_id;
    uint8_t  src_a;
    uint8_t  src_b;
    uint8_t  op_mode;           // 0: linear/diff, 1: bilinear/gate, 2: resonant/wave, 3: memory
    double   frozen_timescale;  // Invariant dynamic timescale
    double   frozen_conduct;    // Invariant baseline conductance
    uint8_t  valid;
};

// 2. Macro-Level Functional Node (Subject to macro-evolution & routing assembly)
struct MacroNode {
    uint8_t  id;
    uint8_t  src_a;
    uint8_t  src_b;
    uint8_t  op_mode;           // Structural computation type
    double   timescale;         // Decay / integration rate alpha
    double   conductance;       // Transmission weight
    double   causal_credit;     // Macro credit assignment
    uint8_t  active;
    uint8_t  epigenetic_locked;
    double   buffer[8];         // Local node state buffer
};

// Universal Multiscale Substrate
struct MultiscaleSubstrate {
    MicroCell   micro_cells[MICRO_CELLS];
    MacroNode   macro_nodes[MACRO_NODES];
    DnaNodeGene dna_registry[DNA_CAPACITY];
    size_t      dna_count;
    FastPRNG    prng;
    bool        enable_dna_regeneration;

    MultiscaleSubstrate(bool with_dna, uint64_t seed = 2026) 
        : dna_count(0), prng(seed), enable_dna_regeneration(with_dna) 
    {
        // 1. Initialize Micro Cells (Ultra-stable, zero random code mutations)
        for (size_t c = 0; c < MICRO_CELLS; ++c) {
            micro_cells[c].state = prng.next();
            micro_cells[c].analog_val = 0.0;
            micro_cells[c].noise_temp = 0.2; // Baseline low exploration
            micro_cells[c].causal_trace = 0.0;
        }

        // 2. Initialize Macro Nodes (High-level self-assembling routing)
        for (size_t n = 0; n < MACRO_NODES; ++n) {
            macro_nodes[n].id = static_cast<uint8_t>(n);
            macro_nodes[n].src_a = static_cast<uint8_t>(n % 4);
            macro_nodes[n].src_b = static_cast<uint8_t>((n + 1) % 4);
            macro_nodes[n].op_mode = static_cast<uint8_t>(n % 4);
            macro_nodes[n].timescale = 0.3 + 0.1 * (n % 6);
            macro_nodes[n].conductance = 0.5;
            macro_nodes[n].causal_credit = 0.0;
            macro_nodes[n].active = (n < 6) ? 1 : 0; // Start with 6 active macro nodes
            macro_nodes[n].epigenetic_locked = (n < 2) ? 1 : 0; // Lock sensory/motor
            std::memset(macro_nodes[n].buffer, 0, sizeof(macro_nodes[n].buffer));
        }

        std::memset(dna_registry, 0, sizeof(dna_registry));
    }

    // High-speed step execution: Micro level executes stable physics,
    // Macro level routes flows between nodes.
    inline void execute_step(uint8_t input_byte) {
        // Micro-level sensory input injection
        micro_cells[0].analog_val = (static_cast<double>(input_byte) - 128.0) * (1.0 / 128.0);
        micro_cells[0].state = (micro_cells[0].state << 8) | input_byte;

        // Execute Macro-Nodes
        for (size_t n = 0; n < MACRO_NODES; ++n) {
            if (!macro_nodes[n].active) continue;

            MacroNode& node = macro_nodes[n];
            double in_a = micro_cells[node.src_a].analog_val;
            double in_b = micro_cells[node.src_b].analog_val;

            double res = 0.0;
            switch (node.op_mode) {
                // Mode 0: Continuous Differential Wave (Laplacian difference)
                case 0: {
                    res = (in_a + in_b) * 0.5 - node.buffer[0] * 0.2;
                    node.buffer[0] = node.buffer[0] * node.timescale + res * (1.0 - node.timescale);
                    break;
                }
                // Mode 1: Bilinear Gating / Multiplicative Resonance
                case 1: {
                    double gate = 1.0 / (1.0 + std::exp(-in_b * 2.0));
                    res = in_a * gate;
                    node.buffer[0] = node.buffer[0] * node.timescale + res * (1.0 - node.timescale);
                    break;
                }
                // Mode 2: Modern Attractor Relaxation
                case 2: {
                    double diff = in_a - node.buffer[0];
                    res = node.buffer[0] + std::tanh(diff * 3.0) * 0.5;
                    node.buffer[0] = node.buffer[0] * node.timescale + res * (1.0 - node.timescale);
                    break;
                }
                // Mode 3: Recursive State Memory (Hyper-Accumulator)
                case 3: {
                    res = node.buffer[0] * node.timescale + (in_a + in_b * 0.5);
                    node.buffer[0] = std::tanh(res);
                    break;
                }
            }

            // Sovereign Noise Modulation at the Micro Target Cell:
            // The destination micro-cell receives the result + sovereign directed noise
            size_t dst_micro = (n + 4) % MICRO_CELLS;
            double noise = 0.0;
            if (micro_cells[dst_micro].noise_temp > 0.001) {
                noise = (prng.next_double() - 0.5) * micro_cells[dst_micro].noise_temp;
            }

            micro_cells[dst_micro].analog_val = node.buffer[0] * node.conductance + noise;
            micro_cells[dst_micro].state = (micro_cells[dst_micro].state << 4) ^ static_cast<uint64_t>(std::abs(node.buffer[0]) * 1000.0);
        }
    }

    // Direct Readout of predicted byte probability distribution
    inline uint8_t predict_byte(double* out_logits) {
        for (size_t v = 0; v < VOCAB_SIZE; ++v) out_logits[v] = 0.0;

        for (size_t n = 0; n < MACRO_NODES; ++n) {
            if (!macro_nodes[n].active) continue;
            size_t cell_idx = (n + 4) % MICRO_CELLS;
            double val = micro_cells[cell_idx].analog_val;
            uint8_t bin = static_cast<uint8_t>(std::clamp((val + 1.0) * 128.0, 0.0, 255.0));
            out_logits[bin] += macro_nodes[n].conductance * 2.0;
        }

        // Softmax
        double max_l = out_logits[0];
        for (size_t v = 1; v < VOCAB_SIZE; ++v) {
            if (out_logits[v] > max_l) max_l = out_logits[v];
        }

        double sum_exp = 0.0;
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            out_logits[v] = std::exp(out_logits[v] - max_l);
            sum_exp += out_logits[v];
        }

        double inv_sum = 1.0 / (sum_exp + 1e-12);
        uint8_t best_b = 0;
        double best_p = 0.0;
        for (size_t v = 0; v < VOCAB_SIZE; ++v) {
            out_logits[v] *= inv_sum;
            if (out_logits[v] > best_p) {
                best_p = out_logits[v];
                best_b = static_cast<uint8_t>(v);
            }
        }
        return best_b;
    }

    // Macro-Level Morphogenesis & Epigenetic DNA Transcription
    void macro_morphogenesis(double fe_step) {
        // 1. Update Macro Causal Credit
        for (size_t n = 0; n < MACRO_NODES; ++n) {
            if (!macro_nodes[n].active) continue;
            size_t cell_idx = (n + 4) % MICRO_CELLS;
            double activity = std::abs(micro_cells[cell_idx].analog_val);

            // Reward nodes active during low Free Energy
            double reward = (fe_step < 3.0) ? (3.0 - fe_step) * activity : -(fe_step - 3.0) * 0.2;
            macro_nodes[n].causal_credit = 0.98 * macro_nodes[n].causal_credit + 0.02 * reward;

            // Sovereign micro-cell temperature cooling / excitation
            if (macro_nodes[n].causal_credit > 0.1) {
                micro_cells[cell_idx].noise_temp = std::max(micro_cells[cell_idx].noise_temp * 0.95, 0.01);
            } else if (fe_step > 4.5) {
                micro_cells[cell_idx].noise_temp = std::min(micro_cells[cell_idx].noise_temp + 0.02, 0.8);
            }

            // 2. Epigenetic Invariant Locking and DNA Transcription
            if (!macro_nodes[n].epigenetic_locked && macro_nodes[n].causal_credit > 0.4) {
                macro_nodes[n].epigenetic_locked = 1;

                // Transcribe node configuration into DNA Registry
                if (dna_count < DNA_CAPACITY) {
                    dna_registry[dna_count].node_id = macro_nodes[n].id;
                    dna_registry[dna_count].src_a = macro_nodes[n].src_a;
                    dna_registry[dna_count].src_b = macro_nodes[n].src_b;
                    dna_registry[dna_count].op_mode = macro_nodes[n].op_mode;
                    dna_registry[dna_count].frozen_timescale = macro_nodes[n].timescale;
                    dna_registry[dna_count].frozen_conduct = macro_nodes[n].conductance;
                    dna_registry[dna_count].valid = 1;
                    dna_count++;
                }
            }

            // 3. Macro Pruning of degraded nodes
            if (!macro_nodes[n].epigenetic_locked && macro_nodes[n].causal_credit < -0.3) {
                macro_nodes[n].active = 0;
            }
        }

        // 4. Macro Sprouting when Free Energy requires structural capacity
        if (fe_step > 3.5 && (prng.next() % 20 == 0)) {
            for (size_t n = 0; n < MACRO_NODES; ++n) {
                if (!macro_nodes[n].active) {
                    macro_nodes[n].src_a = static_cast<uint8_t>(prng.next() % MICRO_CELLS);
                    macro_nodes[n].src_b = static_cast<uint8_t>(prng.next() % MICRO_CELLS);
                    macro_nodes[n].op_mode = static_cast<uint8_t>(prng.next() % 4);
                    macro_nodes[n].timescale = 0.1 + 0.8 * prng.next_double();
                    macro_nodes[n].conductance = 0.2 + 0.6 * prng.next_double();
                    macro_nodes[n].causal_credit = 0.05;
                    macro_nodes[n].active = 1;
                    macro_nodes[n].epigenetic_locked = 0;
                    break;
                }
            }
        }
    }

    // CATASTROPHIC SHOCK INJECTION: Simulates severe damage / distribution collapse
    void inject_catastrophic_shock() {
        // Corrupt 70% of micro cells and wipe active non-locked macro nodes
        for (size_t c = 0; c < MICRO_CELLS; ++c) {
            micro_cells[c].analog_val = (prng.next_double() - 0.5) * 5.0; // Violent noise burst
            micro_cells[c].state = prng.next();
            micro_cells[c].noise_temp = 0.9; // Flash excitation
        }

        for (size_t n = 0; n < MACRO_NODES; ++n) {
            if (!macro_nodes[n].epigenetic_locked) {
                macro_nodes[n].active = 0;
                macro_nodes[n].conductance = 0.0;
            }
            std::memset(macro_nodes[n].buffer, 0, sizeof(macro_nodes[n].buffer));
        }

        // If DNA regeneration is enabled: restore low-level nodes from DNA blueprint
        if (enable_dna_regeneration && dna_count > 0) {
            for (size_t i = 0; i < dna_count; ++i) {
                if (!dna_registry[i].valid) continue;
                uint8_t target_id = dna_registry[i].node_id;
                macro_nodes[target_id].src_a = dna_registry[i].src_a;
                macro_nodes[target_id].src_b = dna_registry[i].src_b;
                macro_nodes[target_id].op_mode = dna_registry[i].op_mode;
                macro_nodes[target_id].timescale = dna_registry[i].frozen_timescale;
                macro_nodes[target_id].conductance = dna_registry[i].frozen_conduct;
                macro_nodes[target_id].active = 1;
                macro_nodes[target_id].epigenetic_locked = 1;
                macro_nodes[target_id].causal_credit = 0.5; // Restored vitality
            }
        }
    }
};

// Benchmark comparison runner
void run_comparative_experiment(bool with_dna) {
    std::cout << "\n========================================================================\n";
    std::cout << ">>> RUNNING MODE: " << (with_dna ? "WITH DNA/RNA REGENERATION" : "WITHOUT DNA (ABLATED)") << "\n";
    std::cout << "========================================================================\n";

    MultiscaleSubstrate substrate(with_dna, 2026);

    const std::string text_corpus = 
        "Intelligence is multiscale self-assembly without top-down prescriptive rules. "
        "Micro cells provide stable execution while macro nodes reorganize connectivity. "
        "DNA/RNA genetic blueprints preserve proven functional circuits against shock. "
        "When catastrophic disruption occurs, the organism regenerates its core topology. ";

    std::vector<uint8_t> stream;
    for (int rep = 0; rep < 300; ++rep) {
        for (char c : text_corpus) stream.push_back(static_cast<uint8_t>(c));
    }

    alignas(64) double logits[VOCAB_SIZE];
    double smooth_fe = 5.55;
    double min_fe = 999.0;
    size_t correct_hits = 0;
    size_t total_evals = 0;

    auto t_start = std::chrono::high_resolution_clock::now();
    size_t N_STEPS = stream.size() - 1;
    size_t shock_step = N_STEPS / 2; // Shock occurs at 50% training mark

    double fe_pre_shock = 0.0;
    double fe_post_shock_immediate = 0.0;
    double fe_final = 0.0;

    for (size_t t = 0; t < N_STEPS; ++t) {
        // CATASTROPHIC SHOCK EVENT
        if (t == shock_step) {
            fe_pre_shock = smooth_fe;
            std::cout << "\n[!] INJECTING CATASTROPHIC SHOCK AT STEP " << t << "...\n";
            substrate.inject_catastrophic_shock();
        }

        uint8_t in_byte = stream[t];
        uint8_t target_byte = stream[t + 1];

        substrate.execute_step(in_byte);

        uint8_t pred = substrate.predict_byte(logits);
        if (pred == target_byte) correct_hits++;
        total_evals++;

        double p_target = std::max(logits[target_byte], 1e-12);
        double fe_step = -std::log(p_target);
        smooth_fe = 0.995 * smooth_fe + 0.005 * fe_step;
        if (smooth_fe < min_fe) min_fe = smooth_fe;

        if (t == shock_step + 1000) {
            fe_post_shock_immediate = smooth_fe;
        }

        substrate.macro_morphogenesis(fe_step);

        if ((t + 1) % 30000 == 0 || t == N_STEPS - 1) {
            size_t active_nodes = 0;
            size_t locked_nodes = 0;
            for (size_t n = 0; n < MACRO_NODES; ++n) {
                if (substrate.macro_nodes[n].active) active_nodes++;
                if (substrate.macro_nodes[n].epigenetic_locked) locked_nodes++;
            }
            std::cout << "  [Step " << (t + 1) << "] FE: " << std::fixed << std::setprecision(4) << smooth_fe
                      << " | HitRate: " << std::setprecision(2) << (double)correct_hits / total_evals * 100.0 << "%"
                      << " | Active: " << active_nodes 
                      << " | Locked: " << locked_nodes
                      << " | DNA Genes: " << substrate.dna_count << "\n";
            std::cout.flush();
        }
    }

    fe_final = smooth_fe;
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    double steps_per_sec = (double)N_STEPS / elapsed_sec;

    std::cout << "\n>>> MODE SUMMARY (" << (with_dna ? "WITH DNA" : "ABLATED WITHOUT DNA") << "):\n";
    std::cout << "  Throughput            : " << steps_per_sec << " steps/sec\n";
    std::cout << "  Pre-Shock FE          : " << fe_pre_shock << " nats\n";
    std::cout << "  Immediate Post-Shock  : " << fe_post_shock_immediate << " nats\n";
    std::cout << "  Final FE              : " << fe_final << " nats\n";
    std::cout << "  Min FE Achieved       : " << min_fe << " nats\n";
    std::cout << "  Overall Hit Rate      : " << (double)correct_hits / total_evals * 100.0 << " %\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-597: MULTISCALE SEPARATION & DNA/RNA REGENERATION BENCHMARK  ===\n";
    std::cout << "========================================================================\n";

    // 1. Run Baseline without DNA/RNA (Catastrophic Amnesia)
    run_comparative_experiment(false);

    // 2. Run with DNA/RNA Regeneration
    run_comparative_experiment(true);

    return 0;
}
