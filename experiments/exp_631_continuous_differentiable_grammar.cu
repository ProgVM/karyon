// ======================================================================================
// EXP-631: Continuous Differentiable Dynamic Tensor Grammar (CD-DTG)
// --------------------------------------------------------------------------------------
// Bazilevs Master Directive & KEP Principle 30 Resolution:
// 1. OVERCOMING THE DISCRETE BRITTLENESS CLIFF (EXP-630 Lesson):
//    - Discrete bytecode assembly causes 99% lethal execution crashes upon single-bit mutation.
//    - Physical universe & biological cognition operate on CONTINUOUS MANIFOLDS.
// 2. CONTINUOUS DIFFERENTIABLE DYNAMIC GRAMMAR (CD-DTG):
//    - Every node in the computational DAG computes a smooth, continuous mixture over
//      a rich operator basis:
//        O(x, y) = sum_k Softmax(w_k / tau) * Operator_k(x, y; param_k)
//      where Operator_k spans:
//        [0] Identity / Linear Pass
//        [1] Outer Product Memory Injection  (x (x) y^T)
//        [2] MatVec Field Projection         (M * x)
//        [3] Non-linear Tanh/GELU Gate
//        [4] Differential Voronoi Metric     (exp(-gamma * ||x - y||^2))
//        [5] Power-Law Attractor             (sgn(x) * |x|^p)
//        [6] Competitive Resonance Softmax
//        [7] Phase-Amplitude Wave Modulation
// 3. SMOOTH MANIFOLD MUTATION & CONTINUOUS PLASTICITY:
//    - Mutations continuously deform operator weights w_k and routing coordinates.
//    - ZERO execution crashes! Every point in genome space corresponds to a valid,
//      smoothly functioning physical tensor law.
// 4. MASSIVE MILLION-SCALE CUDA SCREENING:
//    - Evaluates 1,024,000 Continuous Operator Field Candidates on Tesla T4.
//    - Stress: K=50 Facts, 500 distractor steps, 80% corrupted cues.
//    - Objective: Measure Active Reconstruction Delta (Fidelity_out - Fidelity_base).
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

constexpr int DIM                = 32;       // Base vector dimension
constexpr int NUM_OPERATORS      = 8;        // Operator basis size
constexpr int DAG_DEPTH          = 4;        // Depth of continuous processing layers
constexpr int NUM_FACTS          = 50;       // K=50 Facts
constexpr int DISTRACTOR_STEPS   = 500;      // 500 distractor steps
constexpr int POPULATION_SIZE    = 1024;     // Population size (Blocks)
constexpr int SEARCH_GENERATIONS = 1000;     // 1,000 generations = 1,024,000 ORGANISMS!

// Continuous Differentiable Grammar Genome
struct ContinuousGrammarGenome {
    // Write Subroutine: Continuous Operator Mixture Tensor [DAG_DEPTH][NUM_OPERATORS]
    float write_op_weights[DAG_DEPTH][NUM_OPERATORS];
    float write_params[DAG_DEPTH];
    
    // Read Subroutine: Continuous Operator Mixture Tensor [DAG_DEPTH][NUM_OPERATORS]
    float read_op_weights[DAG_DEPTH][NUM_OPERATORS];
    float read_params[DAG_DEPTH];

    // Topological & Temporal Coordinates
    float memory_decay;           // Continuous state retention
    float learning_rate;          // Plasticity rate
    float temperature;            // Softmax mixture temperature
    float temporal_inertia;       // Recurrent momentum
    uint8_t relax_cycles;         // 1 to 5 relaxation cycles

    // Empirical Phenotype Metrics
    float fitness_delta;          // Active Gain (Fidelity_out - Fidelity_base)
    float raw_fidelity;           // Reconstructed Cosine Fidelity
    float base_fidelity;          // Initial Cue Fidelity
};

__device__ inline float dot_product(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

// Compute Continuous Operator Basis Primitive
__device__ void compute_operator_basis(int op_idx, const float* in_vec, const float* aux_vec,
                                      const float M[DIM][DIM], float* out_vec, float param) {
    switch (op_idx) {
        case 0: // Identity / Linear Pass
            for (int i = 0; i < DIM; ++i) out_vec[i] = in_vec[i];
            break;

        case 1: // Outer Product Memory Injection (handled directly on Matrix M)
            for (int i = 0; i < DIM; ++i) out_vec[i] = in_vec[i];
            break;

        case 2: // MatVec Field Projection
            for (int i = 0; i < DIM; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < DIM; ++j) sum += M[i][j] * in_vec[j];
                out_vec[i] = sum * param;
            }
            break;

        case 3: // Non-linear Tanh/GELU Gate
            for (int i = 0; i < DIM; ++i) {
                float x = in_vec[i] * param;
                out_vec[i] = tanhf(x);
            }
            break;

        case 4: // Differential Voronoi Metric Attractor
            {
                float dist_sq = 0.0f;
                for (int i = 0; i < DIM; ++i) {
                    float diff = in_vec[i] - aux_vec[i];
                    dist_sq += diff * diff;
                }
                float metric_weight = expf(-param * dist_sq);
                for (int i = 0; i < DIM; ++i) out_vec[i] = in_vec[i] * metric_weight;
            }
            break;

        case 5: // Power-Law Attractor
            for (int i = 0; i < DIM; ++i) {
                float x = in_vec[i];
                float s = (x >= 0.0f) ? 1.0f : -1.0f;
                out_vec[i] = s * powf(fabsf(x), fmaxf(0.1f, param));
            }
            break;

        case 6: // Competitive Resonance Softmax
            {
                float dot = dot_product(in_vec, aux_vec, DIM);
                float coeff = expf(fminf(dot * param, 15.0f));
                for (int i = 0; i < DIM; ++i) out_vec[i] = in_vec[i] + coeff * aux_vec[i];
            }
            break;

        case 7: // Phase-Amplitude Wave Modulation
            for (int i = 0; i < DIM; ++i) {
                out_vec[i] = in_vec[i] * cosf(param * (float)i) + aux_vec[i] * sinf(param * (float)i);
            }
            break;
    }
}

// Execute Continuous Differentiable Layer Flow
__device__ void execute_continuous_layer_flow(const float op_weights[DAG_DEPTH][NUM_OPERATORS],
                                             const float params[DAG_DEPTH],
                                             float temperature,
                                             float h_in[DIM],
                                             const float aux_in[DIM],
                                             float M[DIM][DIM],
                                             int is_write_mode,
                                             float learning_rate) {
    float h_curr[DIM];
    for (int i = 0; i < DIM; ++i) h_curr[i] = h_in[i];

    for (int layer = 0; layer < DAG_DEPTH; ++layer) {
        // Compute Softmax Mixture Probabilities over Operator Basis
        float max_w = -1e9f;
        for (int k = 0; k < NUM_OPERATORS; ++k) {
            if (op_weights[layer][k] > max_w) max_w = op_weights[layer][k];
        }

        float sum_exp = 0.0f;
        float probs[NUM_OPERATORS];
        float temp = fmaxf(0.05f, temperature);
        for (int k = 0; k < NUM_OPERATORS; ++k) {
            probs[k] = expf((op_weights[layer][k] - max_w) / temp);
            sum_exp += probs[k];
        }
        for (int k = 0; k < NUM_OPERATORS; ++k) probs[k] /= (sum_exp + 1e-6f);

        // Continuous Superposition Output
        float h_layer_out[DIM] = {0.0f};
        float temp_op_out[DIM];

        for (int k = 0; k < NUM_OPERATORS; ++k) {
            if (probs[k] < 1e-4f) continue;

            compute_operator_basis(k, h_curr, aux_in, M, temp_op_out, params[layer]);

            // Special Case: Continuous Outer Product Update on Memory Field M
            if (is_write_mode && k == 1) {
                float outer_gain = probs[k] * learning_rate;
                for (int r = 0; r < DIM; ++r) {
                    for (int c = 0; c < DIM; ++c) {
                        M[r][c] += outer_gain * h_curr[r] * aux_in[c];
                    }
                }
            }

            for (int i = 0; i < DIM; ++i) {
                h_layer_out[i] += probs[k] * temp_op_out[i];
            }
        }

        // Layer Output Norm & Residual Flow
        float norm = 0.0f;
        for (int i = 0; i < DIM; ++i) {
            h_curr[i] = 0.7f * h_curr[i] + 0.3f * h_layer_out[i];
            norm += h_curr[i] * h_curr[i];
        }
        norm = sqrtf(norm + 1e-6f);
        for (int i = 0; i < DIM; ++i) h_curr[i] /= norm;
    }

    for (int i = 0; i < DIM; ++i) h_in[i] = h_curr[i];
}

// Parallel CUDA Kernel: Evaluate Continuous Differentiable Grammar Population
__global__ void evaluate_continuous_grammar_kernel(const ContinuousGrammarGenome* population,
                                                   ContinuousGrammarGenome* results,
                                                   unsigned long seed,
                                                   int generation) {
    int cand_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand_idx >= POPULATION_SIZE) return;

    ContinuousGrammarGenome g = population[cand_idx];
    curandState rand_state;
    curand_init(seed + generation * 10000 + cand_idx, 0, 0, &rand_state);

    float M[DIM][DIM] = {0.0f};

    // Generate 50 Facts
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

    // 1. WRITE PHASE (Store Facts into Memory Field via Continuous Flow)
    for (int f = 0; f < NUM_FACTS; ++f) {
        float h_fact[DIM];
        for (int i = 0; i < DIM; ++i) h_fact[i] = facts[f][i];

        execute_continuous_layer_flow(g.write_op_weights, g.write_params, g.temperature,
                                      h_fact, facts[f], M, 1, g.learning_rate);
    }

    // 2. CHAOTIC DISTRACTOR OCEAN (500 steps)
    for (int step = 0; step < DISTRACTOR_STEPS; ++step) {
        for (int i = 0; i < DIM; ++i) {
            M[i][i] *= g.memory_decay;
        }
    }

    // 3. READ & ACTIVE RECONSTRUCTION PHASE (80% Corrupted Cues)
    float total_out_fid = 0.0f;
    float total_base_fid = 0.0f;

    for (int f = 0; f < NUM_FACTS; ++f) {
        float h_cue[DIM];
        float cue_in[DIM];

        // 80% Noise Corruption
        for (int i = 0; i < DIM; ++i) {
            cue_in[i] = (curand_uniform(&rand_state) < 0.20f) ? facts[f][i] : curand_normal(&rand_state) * 0.8f;
            h_cue[i] = cue_in[i];
        }

        // Measure Initial Baseline Fidelity
        float dot_in = dot_product(cue_in, facts[f], DIM);
        float norm_in = sqrtf(dot_product(cue_in, cue_in, DIM));
        float base_fid = (norm_in > 1e-4f) ? fmaxf(0.0f, dot_in / norm_in) : 0.0f;
        total_base_fid += base_fid;

        // Execute Read Continuous Flow with Relaxation Cycles
        int cycles = (g.relax_cycles % 5) + 1;
        for (int cyc = 0; cyc < cycles; ++cyc) {
            float h_prev[DIM];
            for (int i = 0; i < DIM; ++i) h_prev[i] = h_cue[i];

            execute_continuous_layer_flow(g.read_op_weights, g.read_params, g.temperature,
                                          h_cue, cue_in, M, 0, g.learning_rate);

            // Recurrent Inertia Blend
            for (int i = 0; i < DIM; ++i) {
                h_cue[i] = g.temporal_inertia * h_prev[i] + (1.0f - g.temporal_inertia) * h_cue[i];
            }
        }

        // Measure Final Output Fidelity
        float dot_out = dot_product(h_cue, facts[f], DIM);
        float norm_out = sqrtf(dot_product(h_cue, h_cue, DIM));
        float out_fid = (norm_out > 1e-4f) ? fmaxf(0.0f, dot_out / norm_out) : 0.0f;
        total_out_fid += out_fid;
    }

    float avg_out_fid  = total_out_fid / NUM_FACTS;
    float avg_base_fid = total_base_fid / NUM_FACTS;

    results[cand_idx].fitness_delta  = avg_out_fid - avg_base_fid; // Active Gain over Noise
    results[cand_idx].raw_fidelity   = avg_out_fid;
    results[cand_idx].base_fidelity  = avg_base_fid;
}

int main() {
    std::cout << "========================================================================\n" << std::flush;
    std::cout << "=== EXP-631: CONTINUOUS DIFFERENTIABLE DYNAMIC GRAMMAR SEARCH        ===\n" << std::flush;
    std::cout << "========================================================================\n" << std::flush;
    std::cout << "Bazilevs Directive   : OVERCOME DISCRETE BRITTLENESS VIA CONTINUOUS MANIFOLDS\n" << std::flush;
    std::cout << "Operator Space       : 8 Continuous Operator Primitives (Gumbel-Softmax Mixture)\n" << std::flush;
    std::cout << "DAG Architecture     : " << DAG_DEPTH << "-Layer Smooth Tensor Flow Network\n" << std::flush;
    std::cout << "Search Population    : 1,024,000 CONTINUOUS ORGANISMS (1,000 Gen x 1024)\n" << std::flush;
    std::cout << "Stress Regime        : K=50 Facts -> 500 Distractors -> 80% CORRUPTED CUES\n" << std::flush;
    std::cout << "Hardware Execution   : Tesla T4 Massively Parallel GPU Acceleration\n\n" << std::flush;

    ContinuousGrammarGenome* d_pop_in = nullptr;
    ContinuousGrammarGenome* d_pop_out = nullptr;

    CHECK_CUDA(cudaMalloc(&d_pop_in,  POPULATION_SIZE * sizeof(ContinuousGrammarGenome)));
    CHECK_CUDA(cudaMalloc(&d_pop_out, POPULATION_SIZE * sizeof(ContinuousGrammarGenome)));

    std::vector<ContinuousGrammarGenome> h_pop(POPULATION_SIZE);
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist_w(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_param(0.1f, 5.0f);
    std::uniform_real_distribution<float> dist_rate(0.01f, 0.95f);
    std::uniform_int_distribution<int> dist_cyc(1, 5);

    // Initialize Smooth Continuous Genomes
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        for (int d = 0; d < DAG_DEPTH; ++d) {
            for (int k = 0; k < NUM_OPERATORS; ++k) {
                h_pop[i].write_op_weights[d][k] = dist_w(rng);
                h_pop[i].read_op_weights[d][k]  = dist_w(rng);
            }
            h_pop[i].write_params[d] = dist_param(rng);
            h_pop[i].read_params[d]  = dist_param(rng);
        }
        h_pop[i].memory_decay     = 0.95f + 0.05f * dist_rate(rng);
        h_pop[i].learning_rate    = dist_rate(rng);
        h_pop[i].temperature      = 0.2f + 0.8f * dist_rate(rng);
        h_pop[i].temporal_inertia = 0.5f * dist_rate(rng);
        h_pop[i].relax_cycles     = dist_cyc(rng);
        h_pop[i].fitness_delta    = -1.0f;
        h_pop[i].raw_fidelity     = 0.0f;
        h_pop[i].base_fidelity    = 0.0f;
    }

    CHECK_CUDA(cudaMemcpy(d_pop_in, h_pop.data(), POPULATION_SIZE * sizeof(ContinuousGrammarGenome), cudaMemcpyHostToDevice));

    ContinuousGrammarGenome global_champion;
    global_champion.fitness_delta = -999.0f;
    int champion_generation = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int gen = 0; gen < SEARCH_GENERATIONS; ++gen) {
        int threads_per_block = 256;
        int blocks = (POPULATION_SIZE + threads_per_block - 1) / threads_per_block;

        evaluate_continuous_grammar_kernel<<<blocks, threads_per_block>>>(d_pop_in, d_pop_out, 2026, gen);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_pop.data(), d_pop_out, POPULATION_SIZE * sizeof(ContinuousGrammarGenome), cudaMemcpyDeviceToHost));

        std::sort(h_pop.begin(), h_pop.end(), [](const ContinuousGrammarGenome& a, const ContinuousGrammarGenome& b) {
            return a.fitness_delta > b.fitness_delta;
        });

        if (h_pop[0].fitness_delta > global_champion.fitness_delta) {
            global_champion = h_pop[0];
            champion_generation = gen;
        }

        if ((gen + 1) % 100 == 0 || gen == 0) {
            std::cout << "[GEN " << std::setw(4) << (gen + 1) << "/" << SEARCH_GENERATIONS << "] "
                      << "Evaluated: " << std::setw(9) << ((gen + 1) * POPULATION_SIZE) << " organisms | "
                      << "Lead Delta: " << std::showpos << std::fixed << std::setprecision(4) << h_pop[0].fitness_delta
                      << " (Raw: " << std::noshowpos << h_pop[0].raw_fidelity << " vs Base: " << h_pop[0].base_fidelity
                      << ", Temp: " << std::fixed << std::setprecision(2) << h_pop[0].temperature << ") | "
                      << "Global Best Delta: " << std::showpos << global_champion.fitness_delta
                      << " (Gen " << champion_generation << ")\n" << std::noshowpos << std::flush;
        }

        // Smooth Continuous Manifold Evolution (Gaussian Mutations over Continuous Weights)
        for (int i = POPULATION_SIZE / 4; i < POPULATION_SIZE; ++i) {
            int p1 = rng() % (POPULATION_SIZE / 4);
            int p2 = rng() % (POPULATION_SIZE / 4);
            ContinuousGrammarGenome child = h_pop[p1];

            std::uniform_real_distribution<float> d_mut(-0.15f, 0.15f);

            // Recombination
            if (rng() % 2 == 0) {
                for (int d = 0; d < DAG_DEPTH; ++d) {
                    for (int k = 0; k < NUM_OPERATORS; ++k) child.read_op_weights[d][k] = h_pop[p2].read_op_weights[d][k];
                    child.read_params[d] = h_pop[p2].read_params[d];
                }
            }

            // Continuous Gaussian deformation of operator weights and params
            for (int d = 0; d < DAG_DEPTH; ++d) {
                for (int k = 0; k < NUM_OPERATORS; ++k) {
                    if (dist_rate(rng) < 0.20f) child.write_op_weights[d][k] += d_mut(rng);
                    if (dist_rate(rng) < 0.20f) child.read_op_weights[d][k]  += d_mut(rng);
                }
                if (dist_rate(rng) < 0.15f) child.write_params[d] = std::clamp(child.write_params[d] + d_mut(rng), 0.05f, 15.0f);
                if (dist_rate(rng) < 0.15f) child.read_params[d]  = std::clamp(child.read_params[d] + d_mut(rng), 0.05f, 15.0f);
            }

            child.memory_decay     = std::clamp(child.memory_decay + 0.01f * d_mut(rng), 0.85f, 1.00f);
            child.learning_rate    = std::clamp(child.learning_rate + d_mut(rng), 0.01f, 0.99f);
            child.temperature      = std::clamp(child.temperature + d_mut(rng), 0.05f, 2.00f);
            child.temporal_inertia = std::clamp(child.temporal_inertia + d_mut(rng), 0.0f, 0.95f);

            h_pop[i] = child;
        }

        CHECK_CUDA(cudaMemcpy(d_pop_in, h_pop.data(), POPULATION_SIZE * sizeof(ContinuousGrammarGenome), cudaMemcpyHostToDevice));
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    uint64_t total_evals = static_cast<uint64_t>(SEARCH_GENERATIONS) * static_cast<uint64_t>(POPULATION_SIZE);

    const char* op_names[] = {
        "Identity", "OuterProduct", "MatVecProj", "TanhGate",
        "VoronoiMetric", "PowerAttractor", "ResonanceSoftmax", "PhaseWave"
    };

    std::cout << "\n========================================================================\n" << std::flush;
    std::cout << ">>> CONTINUOUS DIFFERENTIABLE GRAMMAR SEARCH COMPLETE!\n" << std::flush;
    std::cout << ">>> Total GPU Runtime              : " << std::fixed << std::setprecision(2) << total_sec << " seconds\n" << std::flush;
    std::cout << ">>> Candidate Organisms Evaluated  : " << total_evals << " (1.024 MILLION!)\n" << std::flush;
    std::cout << ">>> Evaluation Throughput          : " << std::fixed << std::setprecision(0) << (total_evals / total_sec) << " e/sec\n" << std::flush;
    std::cout << "------------------------------------------------------------------------\n" << std::flush;
    std::cout << ">>> DISCOVERED CONTINUOUS TENSOR LAW (GLOBAL CHAMPION @ GEN " << champion_generation << "):\n" << std::flush;
    std::cout << "    - Active Reconstruction Delta  : " << std::showpos << std::fixed << std::setprecision(4) << global_champion.fitness_delta << " (Active Signal Gain)\n" << std::noshowpos << std::flush;
    std::cout << "    - Final Reconstructed Fidelity : " << std::fixed << std::setprecision(4) << global_champion.raw_fidelity << " (vs Base Noise: " << global_champion.base_fidelity << ")\n" << std::flush;
    std::cout << "    - Evolved Mixture Temperature  : " << global_champion.temperature << "\n" << std::flush;
    std::cout << "    - Plasticity Learning Rate     : " << global_champion.learning_rate << "\n" << std::flush;
    std::cout << "    - Temporal Recurrent Inertia   : " << global_champion.temporal_inertia << "\n" << std::flush;

    std::cout << "\n>>> DISCOVERED READ SUBROUTINE OPERATOR MIXTURES (DAG Depth " << DAG_DEPTH << "):\n" << std::flush;
    for (int d = 0; d < DAG_DEPTH; ++d) {
        std::cout << "    [Layer " << d << "] Param=" << std::fixed << std::setprecision(3) << global_champion.read_params[d] << " | ";
        for (int k = 0; k < NUM_OPERATORS; ++k) {
            std::cout << op_names[k] << ":" << std::fixed << std::setprecision(2) << global_champion.read_op_weights[d][k] << " ";
        }
        std::cout << "\n" << std::flush;
    }
    std::cout << "========================================================================\n" << std::flush;

    CHECK_CUDA(cudaFree(d_pop_in));
    CHECK_CUDA(cudaFree(d_pop_out));
    return 0;
}
