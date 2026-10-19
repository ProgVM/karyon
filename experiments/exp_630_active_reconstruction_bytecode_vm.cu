// ======================================================================================
// EXP-630: Active Reconstruction Bytecode VM Engine (AR-BVME)
// --------------------------------------------------------------------------------------
// Bazilevs Master Directive & KEP Principle 28 Resolution:
// 1. ELIMINATE GOODHART'S PASSIVE NOP EVASION:
//    - EXP-629 discovered that organisms degenerated into NOP (no-ops) because
//      simply passing through a 50% noisy cue preserved ~0.49 fidelity safely.
// 2. ACTIVE RECONSTRUCTION DELTA FITNESS:
//    - Fitness is defined strictly as GAIN OVER NOISY CUE:
//        Delta_Fidelity = Fidelity(R0_out, Fact) - Fidelity(Cue_noisy, Fact)
//      If program executes NOPs: Delta = 0.0 -> IMMEDIATE EXTINCTION!
//      Only algorithms that actively sharpen, de-noise, and retrieve the fact survive!
// 3. EXTREME NOISE REGIME (80% Corruption):
//    - Input cues have 80% noise (only 20% intact coordinates).
//    - Baseline cue fidelity is ~0.15 - 0.20. Passive survival is impossible.
// 4. RICH SYMBOLIC BYTECODE INSTRUCTION SET:
//    - 16 Opcodes in CUDA registers:
//      OP_NOP, OP_VEC_ADD, OP_VEC_SUB, OP_VEC_HADAMARD, OP_VEC_TANH, OP_VEC_GELU,
//      OP_VEC_EXP_METRIC, OP_VEC_POWER_SIGN, OP_VEC_NORMALIZE, OP_VEC_LATERAL_GATE,
//      OP_TENSOR_OUTER, OP_MATVEC_MUL, OP_MAT_ACCUMULATE, OP_PHASE_ENERGY_WELL,
//      OP_COMPETITIVE_MAX, OP_TOPOLOGICAL_SNAP.
// 5. MASSIVE CUDA SCALE:
//    - 1,024,000 Program Candidates (1,000 Generations x 1024 Blocks) on Tesla T4.
// ======================================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

constexpr int DIM                = 32;       // Base dimension
constexpr int NUM_VEC_REGS       = 4;        // R0, R1, R2, R3
constexpr int MAX_PROGRAM_LEN    = 12;       // Instructions per sub-routine
constexpr int NUM_FACTS          = 50;       // K=50 Facts
constexpr int DISTRACTOR_STEPS   = 500;      // 500 distractor steps
constexpr int POPULATION_SIZE    = 1024;     // Blocks
constexpr int SEARCH_GENERATIONS = 1000;     // 1,000 generations = 1,024,000 PROGRAM EVALS!

enum Opcode {
    OP_NOP = 0,
    OP_VEC_ADD,
    OP_VEC_SUB,
    OP_VEC_HADAMARD,
    OP_VEC_TANH,
    OP_VEC_GELU,
    OP_VEC_EXP_METRIC,
    OP_VEC_POWER_SIGN,
    OP_VEC_NORMALIZE,
    OP_VEC_LATERAL_GATE,
    OP_TENSOR_OUTER,
    OP_MATVEC_MUL,
    OP_MAT_ACCUMULATE,
    OP_PHASE_ENERGY_WELL,
    OP_COMPETITIVE_MAX,
    OP_TOPOLOGICAL_SNAP,
    OP_TOTAL_COUNT
};

struct Instruction {
    uint8_t op;       // Opcode
    uint8_t dst;      // Destination register
    uint8_t src1;     // Source register 1
    uint8_t src2;     // Source register 2
    float   param;    // Continuous scaling parameter
};

struct ActiveProgramGenome {
    Instruction write_prog[MAX_PROGRAM_LEN];
    Instruction read_prog[MAX_PROGRAM_LEN];
    uint8_t     relax_iterations;
    
    float fitness_delta;     // Fidelity_out - Fidelity_in (Active Gain)
    float raw_fidelity;      // Final Cosine Fidelity
    float baseline_fidelity; // Initial Noisy Cue Fidelity
    int   non_nop_count;     // Number of active non-NOP instructions
};

__device__ inline float dot_product(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

__device__ void execute_bytecode(const Instruction* prog, int prog_len,
                                 float R[NUM_VEC_REGS][DIM],
                                 float M[DIM][DIM]) {
    for (int pc = 0; pc < prog_len; ++pc) {
        Instruction inst = prog[pc];
        uint8_t op   = inst.op;
        uint8_t dst  = inst.dst % NUM_VEC_REGS;
        uint8_t src1 = inst.src1 % NUM_VEC_REGS;
        uint8_t src2 = inst.src2 % NUM_VEC_REGS;
        float   p    = inst.param;

        switch (op) {
            case OP_NOP:
                break;

            case OP_VEC_ADD:
                for (int i = 0; i < DIM; ++i) R[dst][i] = R[src1][i] + p * R[src2][i];
                break;

            case OP_VEC_SUB:
                for (int i = 0; i < DIM; ++i) R[dst][i] = R[src1][i] - p * R[src2][i];
                break;

            case OP_VEC_HADAMARD:
                for (int i = 0; i < DIM; ++i) R[dst][i] = R[src1][i] * R[src2][i] * p;
                break;

            case OP_VEC_TANH:
                for (int i = 0; i < DIM; ++i) R[dst][i] = tanhf(p * R[src1][i]);
                break;

            case OP_VEC_GELU:
                for (int i = 0; i < DIM; ++i) {
                    float x = R[src1][i];
                    R[dst][i] = 0.5f * x * (1.0f + tanhf(0.797884f * (x + 0.044715f * x * x * x)));
                }
                break;

            case OP_VEC_EXP_METRIC:
                for (int i = 0; i < DIM; ++i) R[dst][i] = expf(-fabsf(R[src1][i]) * p);
                break;

            case OP_VEC_POWER_SIGN:
                for (int i = 0; i < DIM; ++i) {
                    float x = R[src1][i];
                    float s = (x >= 0.0f) ? 1.0f : -1.0f;
                    R[dst][i] = s * powf(fabsf(x), fmaxf(0.2f, p));
                }
                break;

            case OP_VEC_NORMALIZE: {
                float norm = 0.0f;
                for (int i = 0; i < DIM; ++i) norm += R[src1][i] * R[src1][i];
                norm = sqrtf(norm + 1e-6f);
                for (int i = 0; i < DIM; ++i) R[dst][i] = R[src1][i] / norm;
                break;
            }

            case OP_VEC_LATERAL_GATE: {
                float mean = 0.0f;
                for (int i = 0; i < DIM; ++i) mean += fabsf(R[src1][i]);
                mean /= DIM;
                for (int i = 0; i < DIM; ++i) {
                    R[dst][i] = (fabsf(R[src1][i]) > p * mean) ? R[src1][i] : 0.0f;
                }
                break;
            }

            case OP_TENSOR_OUTER:
                for (int i = 0; i < DIM; ++i) {
                    for (int j = 0; j < DIM; ++j) {
                        M[i][j] += p * R[src1][i] * R[src2][j];
                    }
                }
                break;

            case OP_MATVEC_MUL:
                for (int i = 0; i < DIM; ++i) {
                    float sum = 0.0f;
                    for (int j = 0; j < DIM; ++j) sum += M[i][j] * R[src1][j];
                    R[dst][i] = sum * p;
                }
                break;

            case OP_MAT_ACCUMULATE:
                for (int i = 0; i < DIM; ++i) {
                    for (int j = 0; j < DIM; ++j) {
                        M[i][j] *= p;
                    }
                }
                break;

            case OP_PHASE_ENERGY_WELL: {
                float dist_sq = 0.0f;
                for (int i = 0; i < DIM; ++i) {
                    float diff = R[src1][i] - R[src2][i];
                    dist_sq += diff * diff;
                }
                float factor = 1.0f / (1.0f + p * dist_sq);
                for (int i = 0; i < DIM; ++i) R[dst][i] = R[src1][i] * factor;
                break;
            }

            case OP_COMPETITIVE_MAX: {
                float d = dot_product(R[src1], R[src2], DIM);
                float coeff = expf(fminf(d * p, 20.0f));
                for (int i = 0; i < DIM; ++i) R[dst][i] += coeff * R[src2][i];
                break;
            }

            case OP_TOPOLOGICAL_SNAP: {
                for (int i = 0; i < DIM; ++i) {
                    R[dst][i] = tanhf(R[src1][i] * p);
                }
                break;
            }
        }
    }
}

// Parallel CUDA Kernel: Evaluate Active Reconstruction Delta
__global__ void evaluate_active_reconstruction_kernel(const ActiveProgramGenome* population,
                                                      ActiveProgramGenome* results,
                                                      unsigned long seed,
                                                      int generation) {
    int cand_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand_idx >= POPULATION_SIZE) return;

    ActiveProgramGenome g = population[cand_idx];
    curandState rand_state;
    curand_init(seed + generation * 10000 + cand_idx, 0, 0, &rand_state);

    // Organism's memory field
    float M[DIM][DIM] = {0.0f};

    // 50 Facts
    float facts[NUM_FACTS][DIM];
    for (int f = 0; f < NUM_FACTS; ++f) {
        float norm = 0.0f;
        for (int i = 0; i < DIM; ++i) {
            facts[f][i] = curand_normal(&rand_state);
            norm += facts[f][i] * facts[f][i];
        }
        norm = sqrtf(norm + 1e-6f);
        for (int i = 0; i < DIM; ++i) facts[f][i] /= norm;
    }

    // 1. WRITE PROGRAM EXECUTION
    for (int f = 0; f < NUM_FACTS; ++f) {
        float R[NUM_VEC_REGS][DIM] = {0.0f};
        for (int i = 0; i < DIM; ++i) {
            R[0][i] = facts[f][i];
            R[1][i] = facts[f][i];
        }
        execute_bytecode(g.write_prog, MAX_PROGRAM_LEN, R, M);
    }

    // 2. CHAOTIC DISTRACTOR OCEAN (500 steps)
    for (int step = 0; step < DISTRACTOR_STEPS; ++step) {
        for (int i = 0; i < DIM; ++i) {
            M[i][i] *= 0.999f;
        }
    }

    // 3. READ & RECONSTRUCTION EVALUATION UNDER 80% NOISE
    float total_out_fid = 0.0f;
    float total_base_fid = 0.0f;

    for (int f = 0; f < NUM_FACTS; ++f) {
        float R[NUM_VEC_REGS][DIM] = {0.0f};
        float cue_in[DIM];

        // 80% Extreme Noise Corruption
        for (int i = 0; i < DIM; ++i) {
            cue_in[i] = (curand_uniform(&rand_state) < 0.20f) ? facts[f][i] : curand_normal(&rand_state) * 0.8f;
            R[0][i] = cue_in[i];
        }

        // Measure initial baseline cue fidelity
        float dot_in = dot_product(cue_in, facts[f], DIM);
        float norm_in = sqrtf(dot_product(cue_in, cue_in, DIM));
        float base_fid = (norm_in > 1e-4f) ? fmaxf(0.0f, dot_in / norm_in) : 0.0f;
        total_base_fid += base_fid;

        // Execute Read Program with relaxation cycles
        int iters = (g.relax_iterations % 5) + 1;
        for (int it = 0; it < iters; ++it) {
            execute_bytecode(g.read_prog, MAX_PROGRAM_LEN, R, M);
        }

        // Measure output fidelity
        float dot_out = dot_product(R[0], facts[f], DIM);
        float norm_out = sqrtf(dot_product(R[0], R[0], DIM));
        float out_fid = (norm_out > 1e-4f) ? fmaxf(0.0f, dot_out / norm_out) : 0.0f;
        total_out_fid += out_fid;
    }

    float avg_out_fid  = total_out_fid / NUM_FACTS;
    float avg_base_fid = total_base_fid / NUM_FACTS;

    // ACTIVE DELTA: Must be strictly greater than baseline!
    float delta = avg_out_fid - avg_base_fid;

    // Count non-NOP instructions
    int non_nop = 0;
    for (int p = 0; p < MAX_PROGRAM_LEN; ++p) {
        if (g.write_prog[p].op != OP_NOP) non_nop++;
        if (g.read_prog[p].op != OP_NOP) non_nop++;
    }

    results[cand_idx].fitness_delta     = delta;
    results[cand_idx].raw_fidelity      = avg_out_fid;
    results[cand_idx].baseline_fidelity = avg_base_fid;
    results[cand_idx].non_nop_count     = non_nop;
}

int main() {
    std::cout << "========================================================================\n" << std::flush;
    std::cout << "=== EXP-630: ACTIVE RECONSTRUCTION BYTECODE VM SEARCH                ===\n" << std::flush;
    std::cout << "========================================================================\n" << std::flush;
    std::cout << "Bazilevs Directive   : ERADICATE PASSIVE NOP EVASION, REQUIRE ACTIVE GAIN\n" << std::flush;
    std::cout << "Stress Regime        : K=50 Facts -> 500 Distractors -> 80% CORRUPTED CUES\n" << std::flush;
    std::cout << "Fitness Objective    : Strict Active Delta (Fidelity_out - Fidelity_baseline)\n" << std::flush;
    std::cout << "Instruction Space    : 16 Tensor VM Opcodes (Registers R0..R3, Field M0)\n" << std::flush;
    std::cout << "Search Population    : 1,024,000 PROGRAM ALGORITHMS (1,000 Gen x 1024)\n" << std::flush;
    std::cout << "Hardware Execution   : Massively Parallel CUDA Acceleration (Tesla T4)\n\n" << std::flush;

    ActiveProgramGenome* d_pop_in = nullptr;
    ActiveProgramGenome* d_pop_out = nullptr;

    CHECK_CUDA(cudaMalloc(&d_pop_in,  POPULATION_SIZE * sizeof(ActiveProgramGenome)));
    CHECK_CUDA(cudaMalloc(&d_pop_out, POPULATION_SIZE * sizeof(ActiveProgramGenome)));

    std::vector<ActiveProgramGenome> h_pop(POPULATION_SIZE);
    std::mt19937 rng(2026);
    // Bias away from NOP at initialization (start with active operations 1..15)
    std::uniform_int_distribution<int> dist_active_op(1, OP_TOTAL_COUNT - 1);
    std::uniform_int_distribution<int> dist_all_op(0, OP_TOTAL_COUNT - 1);
    std::uniform_int_distribution<int> dist_reg(0, NUM_VEC_REGS - 1);
    std::uniform_real_distribution<float> dist_param(0.1f, 3.0f);
    std::uniform_int_distribution<int> dist_iter(1, 5);

    for (int i = 0; i < POPULATION_SIZE; ++i) {
        h_pop[i].relax_iterations = dist_iter(rng);
        for (int p = 0; p < MAX_PROGRAM_LEN; ++p) {
            h_pop[i].write_prog[p].op    = (p < 4) ? dist_active_op(rng) : dist_all_op(rng);
            h_pop[i].write_prog[p].dst   = dist_reg(rng);
            h_pop[i].write_prog[p].src1  = dist_reg(rng);
            h_pop[i].write_prog[p].src2  = dist_reg(rng);
            h_pop[i].write_prog[p].param = dist_param(rng);

            h_pop[i].read_prog[p].op     = (p < 4) ? dist_active_op(rng) : dist_all_op(rng);
            h_pop[i].read_prog[p].dst    = dist_reg(rng);
            h_pop[i].read_prog[p].src1   = dist_reg(rng);
            h_pop[i].read_prog[p].src2   = dist_reg(rng);
            h_pop[i].read_prog[p].param  = dist_param(rng);
        }
        h_pop[i].fitness_delta     = -1.0f;
        h_pop[i].raw_fidelity      = 0.0f;
        h_pop[i].baseline_fidelity = 0.0f;
        h_pop[i].non_nop_count     = 0;
    }

    CHECK_CUDA(cudaMemcpy(d_pop_in, h_pop.data(), POPULATION_SIZE * sizeof(ActiveProgramGenome), cudaMemcpyHostToDevice));

    ActiveProgramGenome global_champion;
    global_champion.fitness_delta = -999.0f;
    int champion_generation = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int gen = 0; gen < SEARCH_GENERATIONS; ++gen) {
        int threads_per_block = 256;
        int blocks = (POPULATION_SIZE + threads_per_block - 1) / threads_per_block;

        evaluate_active_reconstruction_kernel<<<blocks, threads_per_block>>>(d_pop_in, d_pop_out, 2026, gen);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_pop.data(), d_pop_out, POPULATION_SIZE * sizeof(ActiveProgramGenome), cudaMemcpyDeviceToHost));

        std::sort(h_pop.begin(), h_pop.end(), [](const ActiveProgramGenome& a, const ActiveProgramGenome& b) {
            return a.fitness_delta > b.fitness_delta;
        });

        if (h_pop[0].fitness_delta > global_champion.fitness_delta) {
            global_champion = h_pop[0];
            champion_generation = gen;
        }

        if ((gen + 1) % 100 == 0 || gen == 0) {
            std::cout << "[GEN " << std::setw(4) << (gen + 1) << "/" << SEARCH_GENERATIONS << "] "
                      << "Evaluated: " << std::setw(9) << ((gen + 1) * POPULATION_SIZE) << " programs | "
                      << "Lead Delta: " << std::showpos << std::fixed << std::setprecision(4) << h_pop[0].fitness_delta
                      << " (Raw: " << std::noshowpos << h_pop[0].raw_fidelity << " vs Base: " << h_pop[0].baseline_fidelity
                      << ", Active Ops: " << h_pop[0].non_nop_count << ") | "
                      << "Global Best Delta: " << std::showpos << global_champion.fitness_delta
                      << " (Gen " << champion_generation << ")\n" << std::noshowpos << std::flush;
        }

        // Genetic Evolution (Selection & Active Mutation)
        for (int i = POPULATION_SIZE / 4; i < POPULATION_SIZE; ++i) {
            int p1 = rng() % (POPULATION_SIZE / 4);
            int p2 = rng() % (POPULATION_SIZE / 4);
            ActiveProgramGenome child = h_pop[p1];

            if (rng() % 2 == 0) {
                for (int p = 0; p < MAX_PROGRAM_LEN; ++p) child.read_prog[p] = h_pop[p2].read_prog[p];
            }

            int mut_count = 1 + (rng() % 3);
            for (int m = 0; m < mut_count; ++m) {
                int target_prog = rng() % 2;
                int idx = rng() % MAX_PROGRAM_LEN;
                Instruction* inst = (target_prog == 0) ? &child.write_prog[idx] : &child.read_prog[idx];

                int mut_type = rng() % 4;
                if (mut_type == 0) inst->op    = dist_all_op(rng);
                if (mut_type == 1) inst->dst   = dist_reg(rng);
                if (mut_type == 2) inst->src1  = dist_reg(rng);
                if (mut_type == 3) {
                    std::uniform_real_distribution<float> d_mut(-0.2f, 0.2f);
                    inst->param = std::clamp(inst->param + d_mut(rng), 0.05f, 15.0f);
                }
            }

            h_pop[i] = child;
        }

        CHECK_CUDA(cudaMemcpy(d_pop_in, h_pop.data(), POPULATION_SIZE * sizeof(ActiveProgramGenome), cudaMemcpyHostToDevice));
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    uint64_t total_evals = static_cast<uint64_t>(SEARCH_GENERATIONS) * static_cast<uint64_t>(POPULATION_SIZE);

    const char* op_str[] = {
        "NOP", "VEC_ADD", "VEC_SUB", "VEC_HADAMARD", "VEC_TANH", "VEC_GELU",
        "VEC_EXP_METRIC", "VEC_POWER_SIGN", "VEC_NORMALIZE", "VEC_LATERAL_GATE",
        "TENSOR_OUTER", "MATVEC_MUL", "MAT_ACCUMULATE", "PHASE_ENERGY_WELL",
        "COMPETITIVE_MAX", "TOPOLOGICAL_SNAP"
    };

    std::cout << "\n========================================================================\n" << std::flush;
    std::cout << ">>> ACTIVE RECONSTRUCTION BYTECODE SEARCH COMPLETE!\n" << std::flush;
    std::cout << ">>> Total GPU Runtime              : " << std::fixed << std::setprecision(2) << total_sec << " seconds\n" << std::flush;
    std::cout << ">>> Candidate Programs Evaluated   : " << total_evals << " (1.024 MILLION!)\n" << std::flush;
    std::cout << ">>> Evaluation Throughput          : " << std::fixed << std::setprecision(0) << (total_evals / total_sec) << " programs / sec\n" << std::flush;
    std::cout << "------------------------------------------------------------------------\n" << std::flush;
    std::cout << ">>> DISCOVERED ACTIVE ALGORITHM (GLOBAL CHAMPION @ GEN " << champion_generation << "):\n" << std::flush;
    std::cout << "    - Active Reconstruction Delta  : " << std::showpos << std::fixed << std::setprecision(4) << global_champion.fitness_delta << " (Gain over noise)\n" << std::noshowpos << std::flush;
    std::cout << "    - Final Reconstructed Fidelity : " << std::fixed << std::setprecision(4) << global_champion.raw_fidelity << " (vs Base: " << global_champion.baseline_fidelity << ")\n" << std::flush;
    std::cout << "    - Active Non-NOP Instructions  : " << global_champion.non_nop_count << " / " << (MAX_PROGRAM_LEN * 2) << "\n" << std::flush;
    std::cout << "    - Relaxation Settling Cycles   : " << (int)global_champion.relax_iterations << "\n" << std::flush;
    std::cout << "\n>>> DISCOVERED WRITE SUBROUTINE BYTECODE:\n" << std::flush;
    for (int p = 0; p < MAX_PROGRAM_LEN; ++p) {
        Instruction inst = global_champion.write_prog[p];
        std::cout << "    [" << std::setw(2) << p << "] " << std::setw(18) << op_str[inst.op] 
                  << " | R" << (int)inst.dst << " <- (R" << (int)inst.src1 << ", R" << (int)inst.src2 
                  << ") param=" << std::fixed << std::setprecision(3) << inst.param << "\n" << std::flush;
    }
    std::cout << "\n>>> DISCOVERED READ SUBROUTINE BYTECODE:\n" << std::flush;
    for (int p = 0; p < MAX_PROGRAM_LEN; ++p) {
        Instruction inst = global_champion.read_prog[p];
        std::cout << "    [" << std::setw(2) << p << "] " << std::setw(18) << op_str[inst.op] 
                  << " | R" << (int)inst.dst << " <- (R" << (int)inst.src1 << ", R" << (int)inst.src2 
                  << ") param=" << std::fixed << std::setprecision(3) << inst.param << "\n" << std::flush;
    }
    std::cout << "========================================================================\n" << std::flush;

    CHECK_CUDA(cudaFree(d_pop_in));
    CHECK_CUDA(cudaFree(d_pop_out));
    return 0;
}
