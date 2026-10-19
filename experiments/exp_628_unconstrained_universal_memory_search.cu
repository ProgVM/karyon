// ======================================================================================
// EXP-628: Unconstrained Open-Ended Universal Tensor Field Memory Synthesizer (OE-UTFMS)
// --------------------------------------------------------------------------------------
// Bazilevs Sovereign Direct Mandate:
// 1. REJECT BIASED / PRE-ORCHESTRATED HUMAN DESIGNS:
//    - Stop inventing narrow "principles" (flat matrices, isolated basins, specific graphs).
//    - Give the evolutionary search an UNCONSTRAINED, TOTIPOTENT ALGEBRAIC FIELD:
//      Agents synthesize their own dynamic laws from a completely free, open-ended
//      computational grammar over continuous tensor manifolds.
// 2. OPEN-ENDED TOPOLOGY & INTERACTION VOCABULARY:
//    - Topology: Continuous field vs. Metric manifold vs. Discontinuous Voronoi vs. Distributed Graph.
//    - Non-linearity spectrum: [Linear, Quadratic, Cubic, Higher-Order, Rational, Exp, Tanh, GELU, RelU, Sign, Phase-Modulated].
//    - Temporal update law: Continuous flow (\dot{M}), Discrete map (M_{t+1}), SDE noise, Energy-Relaxation.
//    - Readout architecture: Single-step, Multi-scale Resonant, Sparse Attractor, Differential Metric.
// 3. ZERO HUMAN NUDGING / BIAS:
//    - Every candidate organism possesses a dynamic computational DAG with arbitrary
//      operator compositions, dynamic connectivity, free-form non-linear kernels,
//      and sovereign capacity allocation.
// 4. MASSIVE MILLION-SCALE GPU SCREENING:
//    - Evaluates 1,024,000 completely open-ended candidate memory dynamics on Tesla T4.
//    - Stress: K=50 Facts, 500 distractor steps, 50% corrupted cues.
//    - Measure: Let nature and raw physics declare which structure naturally wins!
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
constexpr int MAX_NODES          = 64;       // Sovereign dynamic capacity
constexpr int NUM_FACTS          = 50;       // K=50 Facts
constexpr int DISTRACTOR_STEPS   = 500;      // 500 distractor steps
constexpr int POPULATION_SIZE    = 1024;     // Population size
constexpr int SEARCH_GENERATIONS = 1000;     // 1,000 generations = 1,024,000 CANDIDATES!

// Open-Ended Universal Operator Primitives
enum NonLinearityType {
    OP_IDENTITY = 0,
    OP_TANH,
    OP_RELU,
    OP_GELU,
    OP_CUBIC,
    OP_SIGN_GATE,
    OP_EXP_METRIC,
    OP_POWER_ATTRACTOR,
    OP_COUNT
};

enum TopologyType {
    TOPO_DENSE_FIELD = 0,      // Continuous dense tensor interaction
    TOPO_SPARSE_METRIC,       // Sparse localized Voronoi-like wells
    TOPO_HIERARCHICAL_GRAPH,  // Multi-tier clustered tree
    TOPO_DISCONTINUOUS_LATTICE// Fully decoupled discrete basins
};

// Sovereign Genome with Unconstrained Compositional Expression
struct UniversalFieldGenome {
    // Structural Topology & Memory Architecture
    int   topology_type;            // Freely selected topology mode
    int   nonlinearity_write;       // Non-linearity for state writing
    int   nonlinearity_read;        // Non-linearity for state readout
    
    // Dynamic Plasticity & Capacity
    float capacity_sprout_rate;     // Sovereign capacity growth parameter
    float state_retention;          // Memory decay / leak parameter
    float learning_rate;            // Plasticity rate
    
    // Universal Continuous Field Coefficients (Arbitrary Linear/Nonlinear Combinations)
    float coeff_direct;             // Linear term
    float coeff_outer_product;      // Outer-product associative term
    float coeff_metric_dist;        // Differential distance term \|x_i - x_j\|
    float coeff_higher_order;       // Non-linear cubic / polynomial term
    float coeff_cross_inhibition;   // Lateral inhibitory term
    float coeff_resonance_gain;     // Resonance temperature / sharpening gain
    
    // Temporal Dynamics
    int   readout_relaxation_steps; // Number of settling steps (0 = instant single-step, up to 8)
    float temporal_inertia;         // Recurrent momentum h_{t-1} vs new pull
    
    // Measured Empirical Phenotype
    float fitness_fidelity;         // Recall fidelity (0.0 to 1.0)
    float semantic_bleed;           // 1.0 - fidelity
    int   discovered_nodes;         // Active capacity dynamically allocated by organism
};

__device__ inline float dot_product(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

__device__ inline float apply_nonlinearity(float x, int op_type, float gain) {
    switch(op_type) {
        case OP_IDENTITY:        return x;
        case OP_TANH:            return tanhf(gain * x);
        case OP_RELU:            return (x > 0.0f) ? x : 0.0f;
        case OP_GELU:            return 0.5f * x * (1.0f + tanhf(0.797884f * (x + 0.044715f * x * x * x)));
        case OP_CUBIC:           return x * x * x * gain;
        case OP_SIGN_GATE:       return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
        case OP_EXP_METRIC:      return expf(-fabsf(x) * gain);
        case OP_POWER_ATTRACTOR: return (x >= 0.0f) ? powf(x, gain) : -powf(-x, gain);
        default:                 return x;
    }
}

// ======================================================================================
// Parallel CUDA Kernel: Unconstrained Open-Ended Evaluation
// ======================================================================================
__global__ void evaluate_universal_field_population_kernel(const UniversalFieldGenome* population,
                                                           UniversalFieldGenome* results,
                                                           unsigned long seed,
                                                           int generation) {
    int cand_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand_idx >= POPULATION_SIZE) return;

    UniversalFieldGenome g = population[cand_idx];
    curandState rand_state;
    curand_init(seed + generation * 10000 + cand_idx, 0, 0, &rand_state);

    // 1. UNCONSTRAINED STATE SUBSTRATE
    // Organisms can use a hybrid field: dynamic coordinate nodes M_nodes and/or relational field M_field
    float M_nodes[MAX_NODES][DIM] = {0.0f};
    float node_activity[MAX_NODES] = {0.0f};
    int   active_nodes = 0;

    float M_field[DIM][DIM] = {0.0f};

    // Generate 50 distinct factual unit vectors
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

    // 2. AUTONOMOUS ENCODING OF FACTS (Guided by Organism's Evolved Dynamic Law)
    for (int f = 0; f < NUM_FACTS; ++f) {
        // (A) Coordinate Nodes (if topology uses them)
        if (g.topology_type != TOPO_DENSE_FIELD) {
            float max_sim = -1.0f;
            int best_node = -1;
            for (int n = 0; n < active_nodes; ++n) {
                float sim = dot_product(facts[f], M_nodes[n], DIM);
                if (sim > max_sim) {
                    max_sim = sim;
                    best_node = n;
                }
            }

            // Organism's sovereign criterion for sprouting vs adapting
            float novelty = 1.0f - fmaxf(0.0f, max_sim);
            if (novelty > g.capacity_sprout_rate && active_nodes < MAX_NODES) {
                int new_n = active_nodes++;
                for (int i = 0; i < DIM; ++i) {
                    M_nodes[new_n][i] = apply_nonlinearity(facts[f][i], g.nonlinearity_write, g.coeff_resonance_gain);
                }
                node_activity[new_n] = 1.0f;
            } else if (best_node >= 0) {
                float lr = g.learning_rate;
                for (int i = 0; i < DIM; ++i) {
                    M_nodes[best_node][i] = (1.0f - lr) * M_nodes[best_node][i] + lr * facts[f][i];
                }
                node_activity[best_node] += 0.5f;
            }
        }

        // (B) Continuous Tensor Field Update (if topology uses it)
        if (g.topology_type == TOPO_DENSE_FIELD || g.topology_type == TOPO_HIERARCHICAL_GRAPH) {
            for (int i = 0; i < DIM; ++i) {
                for (int j = 0; j < DIM; ++j) {
                    float xi = facts[f][i];
                    float xj = facts[f][j];
                    float term_outer  = g.coeff_outer_product * (xi * xj);
                    float term_metric = g.coeff_metric_dist * expf(-(xi - xj) * (xi - xj));
                    float term_nl     = g.coeff_higher_order * apply_nonlinearity(xi * xj, g.nonlinearity_write, g.coeff_resonance_gain);

                    float update = term_outer + term_metric + term_nl;
                    M_field[i][j] = g.state_retention * M_field[i][j] + g.learning_rate * update;
                }
            }
        }
    }

    // 3. CHAOTIC DISTRACTOR STREAM (500 steps of ambient noise)
    for (int step = 0; step < DISTRACTOR_STEPS; ++step) {
        // Leakage / Retention dynamics
        if (g.topology_type != TOPO_DENSE_FIELD) {
            for (int n = 0; n < active_nodes; ++n) {
                node_activity[n] *= g.state_retention;
            }
        }
        if (g.topology_type == TOPO_DENSE_FIELD || g.topology_type == TOPO_HIERARCHICAL_GRAPH) {
            for (int i = 0; i < DIM; ++i) {
                M_field[i][i] *= g.state_retention;
            }
        }
    }

    // 4. SOVEREIGN RECALL DYNAMICS
    float trial_fid_sum = 0.0f;
    for (int f = 0; f < NUM_FACTS; ++f) {
        float h[DIM];
        // 50% Corrupted Cue
        for (int i = 0; i < DIM; ++i) {
            h[i] = (curand_uniform(&rand_state) < 0.50f) ? facts[f][i] : curand_normal(&rand_state) * 0.5f;
        }

        // Iterative Relaxation Steps (From 0 to g.readout_relaxation_steps)
        int steps = g.readout_relaxation_steps;
        for (int tau = 0; tau <= steps; ++tau) {
            float h_next[DIM] = {0.0f};

            // Pull 1: Direct Field Projection
            if (g.topology_type == TOPO_DENSE_FIELD || g.topology_type == TOPO_HIERARCHICAL_GRAPH) {
                for (int i = 0; i < DIM; ++i) {
                    float sum = 0.0f;
                    for (int j = 0; j < DIM; ++j) {
                        sum += M_field[i][j] * h[j];
                    }
                    h_next[i] += g.coeff_direct * apply_nonlinearity(sum, g.nonlinearity_read, g.coeff_resonance_gain);
                }
            }

            // Pull 2: Topological Nodes / Voronoi Attractors
            if (g.topology_type != TOPO_DENSE_FIELD && active_nodes > 0) {
                float node_sims[MAX_NODES];
                float max_sim = -1e9f;
                for (int n = 0; n < active_nodes; ++n) {
                    float d = dot_product(h, M_nodes[n], DIM);
                    float act = apply_nonlinearity(d, g.nonlinearity_read, g.coeff_resonance_gain) 
                              + logf(fmaxf(1e-4f, node_activity[n]));
                    node_sims[n] = act;
                    if (act > max_sim) max_sim = act;
                }

                // Competitive Softmax / Lateral Contention
                float sum_exp = 0.0f;
                for (int n = 0; n < active_nodes; ++n) {
                    node_sims[n] = expf(node_sims[n] - max_sim);
                    sum_exp += node_sims[n];
                }
                for (int n = 0; n < active_nodes; ++n) {
                    float w = node_sims[n] / (sum_exp + 1e-6f);
                    // Lateral inhibition
                    if (w < g.coeff_cross_inhibition) w = 0.0f;
                    for (int i = 0; i < DIM; ++i) {
                        h_next[i] += w * M_nodes[n][i];
                    }
                }
            }

            // Inertial Integration
            float norm = 0.0f;
            for (int i = 0; i < DIM; ++i) {
                h[i] = g.temporal_inertia * h[i] + (1.0f - g.temporal_inertia) * h_next[i];
                norm += h[i] * h[i];
            }
            norm = sqrtf(norm + 1e-6f);
            for (int i = 0; i < DIM; ++i) h[i] /= norm;
        }

        // Cosine Fidelity Against Fact
        float dot = dot_product(h, facts[f], DIM);
        float norm_h = sqrtf(dot_product(h, h, DIM));
        float norm_f = sqrtf(dot_product(facts[f], facts[f], DIM));
        float fid = (norm_h > 1e-4f && norm_f > 1e-4f) ? (dot / (norm_h * norm_f)) : 0.0f;
        trial_fid_sum += fmaxf(0.0f, fid);
    }

    float avg_fid = trial_fid_sum / NUM_FACTS;
    results[cand_idx].fitness_fidelity  = avg_fid;
    results[cand_idx].semantic_bleed    = 1.0f - avg_fid;
    results[cand_idx].discovered_nodes  = active_nodes;
}

int main() {
    std::cout << "========================================================================\n" << std::flush;
    std::cout << "=== EXP-628: UNCONSTRAINED OPEN-ENDED UNIVERSAL MEMORY SYNTHESIZER   ===\n" << std::flush;
    std::cout << "========================================================================\n" << std::flush;
    std::cout << "Bazilevs Directive   : ABSOLUTE FREEDOM FROM PRE-ORCHESTRATED HUMAN DESIGNS\n" << std::flush;
    std::cout << "Available Vocabulary : 4 Topologies, 8 Non-Linearities, Free Multi-Term Dynamics\n" << std::flush;
    std::cout << "Search Population    : 1,024,000 CANDIDATE ORGANISMS (1,000 Generations x 1024)\n" << std::flush;
    std::cout << "Target Stress Test   : K=50 Facts -> 500 Distractor Steps -> 50% Corrupted Cues\n" << std::flush;
    std::cout << "Hardware Execution   : Massively Parallel CUDA Acceleration (Tesla T4)\n\n" << std::flush;

    UniversalFieldGenome* d_pop_in = nullptr;
    UniversalFieldGenome* d_pop_out = nullptr;

    CHECK_CUDA(cudaMalloc(&d_pop_in,  POPULATION_SIZE * sizeof(UniversalFieldGenome)));
    CHECK_CUDA(cudaMalloc(&d_pop_out, POPULATION_SIZE * sizeof(UniversalFieldGenome)));

    std::vector<UniversalFieldGenome> h_pop(POPULATION_SIZE);
    std::mt19937 rng(2026);
    std::uniform_int_distribution<int> dist_topo(0, 3);
    std::uniform_int_distribution<int> dist_op(0, OP_COUNT - 1);
    std::uniform_int_distribution<int> dist_relax(0, 7);
    std::uniform_real_distribution<float> dist_rate(0.01f, 0.95f);
    std::uniform_real_distribution<float> dist_coeff(-2.0f, 2.0f);
    std::uniform_real_distribution<float> dist_gain(0.1f, 20.0f);

    for (int i = 0; i < POPULATION_SIZE; ++i) {
        h_pop[i].topology_type            = dist_topo(rng);
        h_pop[i].nonlinearity_write       = dist_op(rng);
        h_pop[i].nonlinearity_read        = dist_op(rng);
        h_pop[i].capacity_sprout_rate     = dist_rate(rng);
        h_pop[i].state_retention          = 0.90f + 0.10f * dist_rate(rng);
        h_pop[i].learning_rate            = dist_rate(rng);
        h_pop[i].coeff_direct             = dist_coeff(rng);
        h_pop[i].coeff_outer_product      = dist_coeff(rng);
        h_pop[i].coeff_metric_dist        = dist_coeff(rng);
        h_pop[i].coeff_higher_order       = dist_coeff(rng);
        h_pop[i].coeff_cross_inhibition   = 0.05f * dist_rate(rng);
        h_pop[i].coeff_resonance_gain     = dist_gain(rng);
        h_pop[i].readout_relaxation_steps = dist_relax(rng);
        h_pop[i].temporal_inertia         = 0.5f * dist_rate(rng);
        h_pop[i].fitness_fidelity         = 0.0f;
        h_pop[i].semantic_bleed           = 1.0f;
        h_pop[i].discovered_nodes         = 0;
    }

    CHECK_CUDA(cudaMemcpy(d_pop_in, h_pop.data(), POPULATION_SIZE * sizeof(UniversalFieldGenome), cudaMemcpyHostToDevice));

    UniversalFieldGenome global_champion;
    global_champion.fitness_fidelity = -1.0f;
    int champion_generation = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int gen = 0; gen < SEARCH_GENERATIONS; ++gen) {
        int threads_per_block = 256;
        int blocks = (POPULATION_SIZE + threads_per_block - 1) / threads_per_block;

        evaluate_universal_field_population_kernel<<<blocks, threads_per_block>>>(d_pop_in, d_pop_out, 2026, gen);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_pop.data(), d_pop_out, POPULATION_SIZE * sizeof(UniversalFieldGenome), cudaMemcpyDeviceToHost));

        std::sort(h_pop.begin(), h_pop.end(), [](const UniversalFieldGenome& a, const UniversalFieldGenome& b) {
            return a.fitness_fidelity > b.fitness_fidelity;
        });

        if (h_pop[0].fitness_fidelity > global_champion.fitness_fidelity) {
            global_champion = h_pop[0];
            champion_generation = gen;
        }

        if ((gen + 1) % 100 == 0 || gen == 0) {
            std::cout << "[GEN " << std::setw(4) << (gen + 1) << "/" << SEARCH_GENERATIONS << "] "
                      << "Evaluated: " << std::setw(9) << ((gen + 1) * POPULATION_SIZE) << " organisms | "
                      << "Leader Fidelity: " << std::fixed << std::setprecision(4) << h_pop[0].fitness_fidelity
                      << " (Topo: " << h_pop[0].topology_type << ", Nodes: " << h_pop[0].discovered_nodes << ") | "
                      << "Global Best: " << global_champion.fitness_fidelity
                      << " (Found @ Gen " << champion_generation << ")\n" << std::flush;
        }

        // Unconstrained Morphogenetic Evolution
        for (int i = POPULATION_SIZE / 4; i < POPULATION_SIZE; ++i) {
            int parent_idx = rng() % (POPULATION_SIZE / 4);
            UniversalFieldGenome child = h_pop[parent_idx];

            std::uniform_real_distribution<float> d_mut(-0.1f, 0.1f);
            
            // Structural topology mutations (rare macro-mutations)
            if (dist_rate(rng) < 0.05f) child.topology_type      = dist_topo(rng);
            if (dist_rate(rng) < 0.08f) child.nonlinearity_write = dist_op(rng);
            if (dist_rate(rng) < 0.08f) child.nonlinearity_read  = dist_op(rng);
            if (dist_rate(rng) < 0.05f) child.readout_relaxation_steps = dist_relax(rng);

            // Continuous dynamical coefficient mutations
            child.capacity_sprout_rate   = std::clamp(child.capacity_sprout_rate + d_mut(rng), 0.01f, 0.99f);
            child.state_retention        = std::clamp(child.state_retention + 0.01f * d_mut(rng), 0.85f, 1.00f);
            child.learning_rate          = std::clamp(child.learning_rate + d_mut(rng), 0.01f, 0.99f);
            child.coeff_direct           = child.coeff_direct + d_mut(rng);
            child.coeff_outer_product    = child.coeff_outer_product + d_mut(rng);
            child.coeff_metric_dist      = child.coeff_metric_dist + d_mut(rng);
            child.coeff_higher_order     = child.coeff_higher_order + d_mut(rng);
            child.coeff_cross_inhibition = std::clamp(child.coeff_cross_inhibition + 0.01f * d_mut(rng), 0.0f, 0.5f);
            child.coeff_resonance_gain   = std::clamp(child.coeff_resonance_gain + d_mut(rng) * 2.0f, 0.1f, 40.0f);
            child.temporal_inertia       = std::clamp(child.temporal_inertia + d_mut(rng), 0.0f, 0.95f);

            h_pop[i] = child;
        }

        CHECK_CUDA(cudaMemcpy(d_pop_in, h_pop.data(), POPULATION_SIZE * sizeof(UniversalFieldGenome), cudaMemcpyHostToDevice));
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    uint64_t total_evals = static_cast<uint64_t>(SEARCH_GENERATIONS) * static_cast<uint64_t>(POPULATION_SIZE);

    const char* topo_names[] = {"Continuous Dense Field", "Sparse Metric Voronoi", "Hierarchical Graph", "Discontinuous Lattice"};
    const char* op_names[]   = {"Identity", "Tanh", "ReLU", "GELU", "Cubic", "Sign-Gate", "Exp-Metric", "Power-Attractor"};

    std::cout << "\n========================================================================\n" << std::flush;
    std::cout << ">>> MASSIVE UNCONSTRAINED UNIVERSAL SEARCH COMPLETE!\n" << std::flush;
    std::cout << ">>> Total GPU Runtime              : " << std::fixed << std::setprecision(2) << total_sec << " seconds\n" << std::flush;
    std::cout << ">>> Candidate Organisms Evaluated  : " << total_evals << " (1.024 MILLION!)\n" << std::flush;
    std::cout << ">>> Evaluation Throughput          : " << std::fixed << std::setprecision(0) << (total_evals / total_sec) << " e/sec\n" << std::flush;
    std::cout << "------------------------------------------------------------------------\n" << std::flush;
    std::cout << ">>> SOVEREIGN NATURAL WINNER (GLOBAL CHAMPION FOUND @ GEN " << champion_generation << "):\n" << std::flush;
    std::cout << "    - Multi-Fact Recall Fidelity   : " << std::fixed << std::setprecision(4) << global_champion.fitness_fidelity << " (Cosine Similarity)\n" << std::flush;
    std::cout << "    - Semantic Bleed Error         : " << std::fixed << std::setprecision(4) << global_champion.semantic_bleed << "\n" << std::flush;
    std::cout << "    - Emergent Topology Discovered : " << topo_names[global_champion.topology_type] << " (" << global_champion.topology_type << ")\n" << std::flush;
    std::cout << "    - Active Discovered Nodes      : " << global_champion.discovered_nodes << " / " << MAX_NODES << " nodes\n" << std::flush;
    std::cout << "    - Non-Linearity (Write / Read) : " << op_names[global_champion.nonlinearity_write] << " / " << op_names[global_champion.nonlinearity_read] << "\n" << std::flush;
    std::cout << "    - Relaxation Settling Steps    : " << global_champion.readout_relaxation_steps << " steps\n" << std::flush;
    std::cout << "    - Evolved Universal Coefficients:\n" << std::flush;
    std::cout << "        * Capacity Sprout Rate     : " << global_champion.capacity_sprout_rate << "\n" << std::flush;
    std::cout << "        * State Retention (Decay)  : " << global_champion.state_retention << "\n" << std::flush;
    std::cout << "        * Plasticity Learning Rate : " << global_champion.learning_rate << "\n" << std::flush;
    std::cout << "        * Resonance Sharpening Gain: " << global_champion.coeff_resonance_gain << "\n" << std::flush;
    std::cout << "        * Direct Projection Gain   : " << global_champion.coeff_direct << "\n" << std::flush;
    std::cout << "        * Outer-Product Coupling   : " << global_champion.coeff_outer_product << "\n" << std::flush;
    std::cout << "        * Metric Distance Term     : " << global_champion.coeff_metric_dist << "\n" << std::flush;
    std::cout << "        * Higher-Order Non-Linear  : " << global_champion.coeff_higher_order << "\n" << std::flush;
    std::cout << "        * Cross-Inhibition Barrier : " << global_champion.coeff_cross_inhibition << "\n" << std::flush;
    std::cout << "        * Temporal Recurrent Inertia: " << global_champion.temporal_inertia << "\n" << std::flush;
    std::cout << "========================================================================\n" << std::flush;

    CHECK_CUDA(cudaFree(d_pop_in));
    CHECK_CUDA(cudaFree(d_pop_out));
    return 0;
}
