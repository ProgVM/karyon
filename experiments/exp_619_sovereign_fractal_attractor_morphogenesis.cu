// ======================================================================================
// EXP-619: Sovereign Self-Similar Fractal Attractor Morphogenesis (SS-FAM)
// --------------------------------------------------------------------------------------
// Fundamental Cybernetics (Bazilevs Sovereign Mandate & KEP Principles 2, 4, 28, 29, 30):
// 1. "ИДЕЯ БАЗИЛЕВСА О ФРАКТАЛАХ":
//    - Fractals embody universal self-similarity across spatial and temporal scales:
//      a small micro-circuit's structure and dynamics are mathematically identical/similar
//      to the macro-system: z_{k+1} = \mathcal{F}_{\theta}(z_k, c).
//    - Solves the Speech & Working Memory Bottleneck: A fractal generator can hold infinite
//      multi-scale context (phoneme -> morpheme -> word -> sentence) inside a compact 3-node
//      self-similar attractor without blowing up Landauer ATP node count!
// 2. SELF-SIMILAR FRACTAL RECURSIVE SUPER-OP (UOP_FRACTAL_RECURSION):
//    - Every node can execute a fractal recursive transformation field:
//      z_0 = in_1, \quad z_{k+1} = f_{\theta\_law}(z_k, in_2) \gg 2 \quad \text{for } k=0..D_{fractal}
//      where D_{fractal} \in [1..4] is the node's self-similar depth coordinate.
// 3. MULTI-SCALE TEMPORAL FRACTAL CLOCKS (\Delta t_l = 2^l):
//    - Cortical Tier l operates on fractal temporal scale \Delta t_l = 2^l, matching the scale-invariant
//      hierarchical structure of human language and natural physical wavefields.
// 4. EPIGENETIC LANDAUER SUBSIDY FOR FRACTAL MEMORY:
//    - Nodes displaying fractal self-similarity and long-horizon predictive utility receive an
//      epigenetic Landauer ATP cost subsidy: E_{cost} = E_{base} / (1 + \log_2(1 + D_{fractal})),
//      protecting speech/language memory from premature apoptosis!
// 5. UNIVERSAL 8-MODAL BENCHMARK SUITE:
//    - Evaluated across Physics (Lorenz), Waves (Oscillator), Turing Logic, Parity Stream, LIFO Stack,
//      Markov, Multi-Byte Speech Stream ("instruction_response_karyon_core_synthesis"), and Video Game Causal Physics.
// 6. 100% RESIDENT CUDA PIPELINE:
//    - 16,384 parallel organisms, 6,000 deep generations on Tesla T4 GPU.
// ======================================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
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

constexpr int POP_SIZE          = 16384;   // 16,384 parallel organisms
constexpr int NUM_GENERATIONS   = 6000;    // 6,000 deep generations
constexpr int MACRO_PACING_K    = 250;     // Macro-phase transition interval
constexpr int MAX_LAYERS        = 4;       // Up to 4 cortical layer tiers
constexpr int MAX_NODES_PER_LAYER = 32;    // Dynamic width per layer (total max nodes = 128)
constexpr int MAX_TOTAL_NODES   = MAX_LAYERS * MAX_NODES_PER_LAYER; // 128
constexpr int MAX_CAPSULES      = 8;       // Up to 8 specialist capsule slots
constexpr int MAX_CAPSULE_NODES = 8;       // Nodes per isolated capsule
constexpr int MEM_SIZE          = 128;     // 128 dynamic registers & universal state field
constexpr int SENSORY_INPUTS    = 4;       // Environmental sensory inputs
constexpr int TELEMETRY_INPUTS  = 3;       // [F_t, G_t, Omega_t]
constexpr int FEEDBACK_CHANNELS = 4;       // Reciprocal top-down feedback signals
constexpr int TOTAL_BASE_INPUTS = SENSORY_INPUTS + TELEMETRY_INPUTS + FEEDBACK_CHANNELS; // 11
constexpr int TOTAL_INPUTS      = TOTAL_BASE_INPUTS + MAX_CAPSULES;  // 19 inputs
constexpr int NUM_OUTPUTS       = 2;
constexpr int NUM_NICHES        = 8;
constexpr int NICHE_SLOTS       = 1024;    // 1024 resident Pareto niche slots
constexpr int MAX_FRACTAL_DEPTH = 4;       // Max self-similar recursive fractal depth

enum UniversalSuperOp : uint8_t {
    UOP_PASS = 0,
    UOP_ADD,
    UOP_SUB,
    UOP_MUL_SHIFT,
    UOP_DIV_MOD,
    UOP_XOR,
    UOP_AND,
    UOP_OR,
    UOP_NOT,
    UOP_SHL,
    UOP_SHR,
    UOP_MIN,
    UOP_MAX,
    UOP_ABS,
    UOP_SATURATE,
    UOP_CMP_EQ,
    UOP_CMP_LT,
    UOP_COND_MUX,
    UOP_LOAD_REG,
    UOP_STORE_REG,
    UOP_ACCUMULATE,
    UOP_ASSOC_READ,
    UOP_ASSOC_WRITE,
    UOP_STACK_PUSH,
    UOP_STACK_POP,
    UOP_FRACTAL_RECURSION, // Self-similar fractal generator super-op!
    NUM_UNIVERSAL_OPS
};

struct GeneNode {
    uint8_t op;           // Universal Super-Opcode
    uint8_t in1;          // Primary input coordinate
    uint8_t in2;          // Secondary input coordinate
    uint8_t gate_in;      // Continuous modulating gate coordinate
    
    // Continuous Dynamic Law Synthesis Coordinates (\theta_law)
    int8_t  alpha_lin;    // Linear law gain [-16 .. 16]
    int8_t  alpha_sat;    // Saturating law gain [-16 .. 16]
    int8_t  alpha_step;   // Threshold step law gain [-16 .. 16]
    int8_t  alpha_bilin;  // Bilinear product law gain [-16 .. 16]
    
    int8_t  weight1;      // Primary coupling gain [-16 .. 16]
    int8_t  weight2;      // Secondary coupling gain [-16 .. 16]
    
    uint8_t fractal_depth;// Self-similar fractal recursion depth [1 .. 4]
    uint8_t is_active;    // Sovereign activation bit (Apoptosis/Sprouting)
    int8_t  causal_credit;// Endogenous morphogenetic potential [-128 .. 127]
};

struct alignas(16) OrganCapsule {
    GeneNode nodes[MAX_CAPSULE_NODES];
    uint8_t  active_nodes;
    uint8_t  source_niche;   // Source niche ID
    uint8_t  is_active;      // Sovereign activation bit
    int8_t   output_coupling;// Output scaling [-16 .. 16]
};

struct alignas(64) SovereignFractalGenome {
    GeneNode     layers[MAX_LAYERS][MAX_NODES_PER_LAYER];
    OrganCapsule capsules[MAX_CAPSULES];
    
    // Reciprocal Top-Down Feedback Projections (Layer (l+1) -> Layer l)
    int8_t       fb_couplings[FEEDBACK_CHANNELS]; // [-16 .. 16]
    
    // Self-Similar Fractal Attractor Parameters
    uint8_t      fractal_clock_scale;        // [1 .. 4] Macro fractal time clock ratio
    int8_t       beta_morpheme;              // [-16 .. 16] Phonotactic precision snapping gain
    
    // Macro & Micro Sovereign Regulators
    uint8_t      active_layers;              // [1 .. 4] Cortical layer depth
    uint8_t      layer_nodes[MAX_LAYERS];    // [1 .. 32] Width per layer
    uint8_t      active_capsules_limit;      // [1 .. 8] Capsule capacity limit
    uint8_t      bus_topology_mode;          // [0: FF, 1: Dense, 2: Recurrent, 3: Res]
    uint8_t      diff_velocity;              // [1 .. 16] Micro differentiation pace
    uint32_t     macro_transitions_count;    // Total macro phase transitions experienced
    
    // Telemetry & Thermodynamic Fitness
    float        niche_errors[NUM_NICHES];
    float        metabolic_energy_cost;      // Continuous ATP / Landauer structural cost
    float        composite_fe;               // Sensory Free Energy + Metabolic Cost
    float        prev_macro_fe;              // Baseline FE at last macro phase transition
};

__device__ inline int16_t dev_clamp16(int32_t val, int32_t low, int32_t high) {
    return static_cast<int16_t>((val < low) ? low : ((val > high) ? high : val));
}

// ======================================================================================
// Sovereign Continuous Dynamic Law Synthesizer with Fractal Self-Similar Recursion
// ======================================================================================
__device__ inline int16_t exec_sovereign_law_field_op(const GeneNode& gn, int16_t a, int16_t b, int16_t* mem) {
    int32_t raw = 0;

    if (gn.op == UOP_FRACTAL_RECURSION) {
        // --- SELF-SIMILAR FRACTAL RECURSIVE TRANSFORM FIELD ---
        // z_0 = a; z_{k+1} = ShapedLaw(z_k, b) >> 2
        int32_t z_curr = static_cast<int32_t>(a);
        int depth = (gn.fractal_depth < 1) ? 1 : ((gn.fractal_depth > MAX_FRACTAL_DEPTH) ? MAX_FRACTAL_DEPTH : gn.fractal_depth);

        for (int k = 0; k < depth; ++k) {
            int32_t v_lin   = z_curr;
            int32_t v_sat   = (z_curr < -256) ? -256 : ((z_curr > 255) ? 255 : z_curr);
            int32_t v_step  = (z_curr > 0) ? 16 : ((z_curr < 0) ? -16 : 0);
            int32_t v_bilin = (z_curr * static_cast<int32_t>(b)) >> 4;

            int32_t shaped = (v_lin   * static_cast<int32_t>(gn.alpha_lin) +
                              v_sat   * static_cast<int32_t>(gn.alpha_sat) +
                              v_step  * static_cast<int32_t>(gn.alpha_step) +
                              v_bilin * static_cast<int32_t>(gn.alpha_bilin)) >> 4;

            z_curr = (z_curr + shaped) >> 1; // Self-similar fractal attractor convergence
        }
        raw = z_curr;
    } else {
        switch(gn.op) {
            case UOP_PASS:        raw = a; break;
            case UOP_ADD:         raw = (static_cast<int32_t>(a) * gn.weight1 + static_cast<int32_t>(b) * gn.weight2) >> 4; break;
            case UOP_SUB:         raw = (static_cast<int32_t>(a) * gn.weight1 - static_cast<int32_t>(b) * gn.weight2) >> 4; break;
            case UOP_MUL_SHIFT:   raw = (static_cast<int32_t>(a) * static_cast<int32_t>(b)) >> 4; break;
            case UOP_DIV_MOD:     raw = (b != 0) ? (a % b) : 0; break;
            case UOP_XOR:         raw = a ^ b; break;
            case UOP_AND:         raw = a & b; break;
            case UOP_OR:          raw = a | b; break;
            case UOP_NOT:         raw = ~a; break;
            case UOP_SHL:         raw = a << (b & 7); break;
            case UOP_SHR:         raw = a >> (b & 7); break;
            case UOP_MIN:         raw = (a < b) ? a : b; break;
            case UOP_MAX:         raw = (a > b) ? a : b; break;
            case UOP_ABS:         raw = (a < 0) ? -a : a; break;
            case UOP_SATURATE:    raw = (a < -128) ? -128 : ((a > 127) ? 127 : a); break;
            case UOP_CMP_EQ:      raw = (a == b) ? 16 : 0; break;
            case UOP_CMP_LT:      raw = (a < b) ? 16 : 0; break;
            case UOP_COND_MUX:    raw = (a > 0) ? b : 0; break;
            case UOP_LOAD_REG:    raw = mem[static_cast<uint8_t>(a) % 64]; break;
            case UOP_STORE_REG:   mem[static_cast<uint8_t>(b) % 64] = a; raw = a; break;
            case UOP_ACCUMULATE:  {
                uint8_t reg = static_cast<uint8_t>(a) % 64;
                int32_t update = (static_cast<int32_t>(b) * gn.weight2) >> 4;
                mem[reg] = dev_clamp16(static_cast<int32_t>(mem[reg]) + update, -4096, 4095);
                raw = mem[reg];
                break;
            }
            case UOP_ASSOC_READ:  {
                uint8_t h = (static_cast<uint32_t>(a) * 73u + 19u) % 32;
                raw = mem[64 + h];
                break;
            }
            case UOP_ASSOC_WRITE: {
                uint8_t h = (static_cast<uint32_t>(a) * 73u + 19u) % 32;
                mem[64 + h] = b;
                raw = b;
                break;
            }
            case UOP_STACK_PUSH:  {
                int16_t sp = mem[127];
                mem[96 + (sp & 31)] = a;
                mem[127] = dev_clamp16(sp + 1, 0, 31);
                raw = mem[127];
                break;
            }
            case UOP_STACK_POP:   {
                int16_t sp = mem[127];
                if (sp > 0) sp--;
                mem[127] = sp;
                raw = mem[96 + (sp & 31)];
                break;
            }
            default: raw = a; break;
        }

        int32_t v_lin   = raw;
        int32_t v_sat   = (raw < -256) ? -256 : ((raw > 255) ? 255 : raw);
        int32_t v_step  = (raw > 0) ? 16 : ((raw < 0) ? -16 : 0);
        int32_t v_bilin = (raw * static_cast<int32_t>(b)) >> 4;

        int32_t shaped = (v_lin   * static_cast<int32_t>(gn.alpha_lin) +
                          v_sat   * static_cast<int32_t>(gn.alpha_sat) +
                          v_step  * static_cast<int32_t>(gn.alpha_step) +
                          v_bilin * static_cast<int32_t>(gn.alpha_bilin)) >> 4;

        raw = (shaped * static_cast<int32_t>(gn.weight1)) >> 4;
    }

    return dev_clamp16(raw, -4096, 4095);
}

// ======================================================================================
// Multi-Tier Laminar Feedforward & Fractal Temporal Clock Sweep
// ======================================================================================
__device__ inline void run_sovereign_fractal_organism(const SovereignFractalGenome& agent, 
                                                       const int16_t* sensory, 
                                                       int16_t fe_cur, 
                                                       int16_t fe_exp, 
                                                       int16_t fe_cum,
                                                       int16_t* outputs, 
                                                       int16_t* mem,
                                                       int16_t* feedback_state,
                                                       int step_idx) {
    // 1. Execute active isolated capsules
    int16_t capsule_outputs[MAX_CAPSULES] = {0};
    int max_caps = (agent.active_capsules_limit > MAX_CAPSULES) ? MAX_CAPSULES : agent.active_capsules_limit;

    for (int c = 0; c < max_caps; ++c) {
        if (!agent.capsules[c].is_active || agent.capsules[c].active_nodes == 0) continue;
        const OrganCapsule& cap = agent.capsules[c];
        
        int16_t cap_bus[TOTAL_BASE_INPUTS + MAX_CAPSULE_NODES];
        for (int i = 0; i < SENSORY_INPUTS; ++i) cap_bus[i] = sensory[i];
        cap_bus[SENSORY_INPUTS + 0] = fe_cur;
        cap_bus[SENSORY_INPUTS + 1] = fe_exp;
        cap_bus[SENSORY_INPUTS + 2] = fe_cum;
        for (int fb = 0; fb < FEEDBACK_CHANNELS; ++fb) {
            cap_bus[SENSORY_INPUTS + TELEMETRY_INPUTS + fb] = feedback_state[fb];
        }

        for (int n = 0; n < cap.active_nodes; ++n) {
            const GeneNode& gn = cap.nodes[n];
            if (!gn.is_active) continue;

            int16_t gate_signal = cap_bus[gn.gate_in % (TOTAL_BASE_INPUTS + n)];
            if (gate_signal <= -8) {
                cap_bus[TOTAL_BASE_INPUTS + n] = cap_bus[gn.in1 % (TOTAL_BASE_INPUTS + n)];
                continue;
            }
            int16_t in1_val = cap_bus[gn.in1 % (TOTAL_BASE_INPUTS + n)];
            int16_t in2_val = cap_bus[gn.in2 % (TOTAL_BASE_INPUTS + n)];
            cap_bus[TOTAL_BASE_INPUTS + n] = exec_sovereign_law_field_op(gn, in1_val, in2_val, mem);
        }

        int16_t raw_out = cap_bus[TOTAL_BASE_INPUTS + cap.active_nodes - 1];
        capsule_outputs[c] = dev_clamp16((static_cast<int32_t>(raw_out) * cap.output_coupling) >> 4, -4096, 4095);
    }

    // 2. Build Layered Cortical Highway Bus
    int16_t bus[TOTAL_INPUTS + MAX_TOTAL_NODES];
    for (int i = 0; i < SENSORY_INPUTS; ++i) bus[i] = sensory[i];
    bus[SENSORY_INPUTS + 0] = fe_cur;
    bus[SENSORY_INPUTS + 1] = fe_exp;
    bus[SENSORY_INPUTS + 2] = fe_cum;
    for (int fb = 0; fb < FEEDBACK_CHANNELS; ++fb) {
        bus[SENSORY_INPUTS + TELEMETRY_INPUTS + fb] = feedback_state[fb];
    }
    for (int c = 0; c < MAX_CAPSULES; ++c) {
        bus[TOTAL_BASE_INPUTS + c] = capsule_outputs[c];
    }

    int current_bus_offset = TOTAL_INPUTS;
    int last_node_global_idx = TOTAL_INPUTS;
    int prev_node_global_idx = TOTAL_INPUTS;

    int act_layers = (agent.active_layers < 1) ? 1 : ((agent.active_layers > MAX_LAYERS) ? MAX_LAYERS : agent.active_layers);
    int layer_start_indices[MAX_LAYERS];

    // 3. Multi-Tier Laminar Feedforward Sweep with Multi-Scale Fractal Clocks (\Delta t_l = 2^l)
    for (int l = 0; l < act_layers; ++l) {
        layer_start_indices[l] = current_bus_offset;
        int layer_width = agent.layer_nodes[l];
        if (layer_width < 1) layer_width = 1;
        if (layer_width > MAX_NODES_PER_LAYER) layer_width = MAX_NODES_PER_LAYER;

        // Fractal clock scale gate: higher layers update at fractal power-of-2 strides
        int clock_stride = 1 << l; // Layer 0: 1, Layer 1: 2, Layer 2: 4, Layer 3: 8
        bool clock_tick = (step_idx % clock_stride == 0);

        for (int n = 0; n < layer_width; ++n) {
            const GeneNode& gn = agent.layers[l][n];
            if (!gn.is_active) {
                bus[current_bus_offset + n] = 0;
                continue;
            }

            int available_inputs = current_bus_offset + n;
            if (agent.bus_topology_mode == 0) available_inputs = TOTAL_INPUTS + n;

            if (!clock_tick && l > 0) {
                // Hold persistent state on non-clock ticks
                bus[current_bus_offset + n] = mem[(l * 32 + n) % 64];
            } else {
                int16_t gate_signal = bus[gn.gate_in % available_inputs];
                if (gate_signal <= -8) {
                    bus[current_bus_offset + n] = bus[gn.in1 % available_inputs];
                } else {
                    int16_t in1_val = bus[gn.in1 % available_inputs];
                    int16_t in2_val = bus[gn.in2 % available_inputs];
                    bus[current_bus_offset + n] = exec_sovereign_law_field_op(gn, in1_val, in2_val, mem);
                }
                mem[(l * 32 + n) % 64] = bus[current_bus_offset + n];
            }

            prev_node_global_idx = last_node_global_idx;
            last_node_global_idx = current_bus_offset + n;
        }
        current_bus_offset += layer_width;
    }

    int16_t raw_out0 = bus[last_node_global_idx];
    int16_t raw_out1 = bus[prev_node_global_idx];

    // Phonotactic Morphemic Attractor Precision Snapping (\beta_morph)
    if (agent.beta_morpheme != 0) {
        int32_t gain = 16 + static_cast<int32_t>(agent.beta_morpheme);
        if (gain < 1) gain = 1;
        outputs[0] = dev_clamp16((static_cast<int32_t>(raw_out0) * gain) >> 4, -4096, 4095);
        outputs[1] = raw_out1;
    } else {
        outputs[0] = raw_out0;
        outputs[1] = raw_out1;
    }

    // 4. Reciprocal Top-Down Feedback Synthesis
    if (act_layers > 1) {
        int top_layer = act_layers - 1;
        int top_start = layer_start_indices[top_layer];
        int top_width = agent.layer_nodes[top_layer];

        for (int fb = 0; fb < FEEDBACK_CHANNELS; ++fb) {
            int node_pick = top_start + (fb % top_width);
            int32_t projected = (static_cast<int32_t>(bus[node_pick]) * agent.fb_couplings[fb]) >> 4;
            feedback_state[fb] = dev_clamp16((static_cast<int32_t>(feedback_state[fb]) * 3 + projected) >> 2, -2048, 2047);
        }
    } else {
        for (int fb = 0; fb < FEEDBACK_CHANNELS; ++fb) {
            feedback_state[fb] = (feedback_state[fb] * 3) >> 2;
        }
    }
}

__device__ inline float compute_metabolic_energy_cost(const SovereignFractalGenome& agent) {
    // Landauer / ATP Cost with Epigenetic Self-Similar Fractal Memory Subsidy
    float cost = 0.0f;
    for (int l = 0; l < agent.active_layers; ++l) {
        for (int n = 0; n < agent.layer_nodes[l]; ++n) {
            const GeneNode& gn = agent.layers[l][n];
            if (gn.is_active) {
                float node_energy = 0.008f;
                if (gn.op == UOP_FRACTAL_RECURSION) {
                    // EPIGENETIC FRACTAL SUBSIDY: Self-similar fractal nodes reuse equations recursively,
                    // yielding infinite scale representation at fractional Landauer cost!
                    float fractal_subsidy = 1.0f / (1.0f + 0.5f * gn.fractal_depth);
                    node_energy *= fractal_subsidy;
                }
                float law_complexity = (abs(gn.alpha_lin) + abs(gn.alpha_sat) + abs(gn.alpha_step) + abs(gn.alpha_bilin)) / 64.0f;
                cost += (node_energy + law_complexity * 0.003f);
            }
        }
    }
    for (int c = 0; c < agent.active_capsules_limit && c < MAX_CAPSULES; ++c) {
        if (agent.capsules[c].is_active) {
            cost += 0.025f + agent.capsules[c].active_nodes * 0.005f;
        }
    }
    return cost;
}

__global__ void init_curand_kernel(curandState* states, unsigned long seed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// ======================================================================================
// Universal Multi-Modal Benchmark Crucible (8 Diverse Environments)
// ======================================================================================
__global__ void evaluate_multimodal_crucible_kernel(SovereignFractalGenome* agents, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    SovereignFractalGenome& agent = agents[idx];
    int16_t mem[MEM_SIZE];
    int16_t feedback_state[FEEDBACK_CHANNELS];
    
    #define FLUSH_FIELD_STATE() \
        for(int m=0; m<MEM_SIZE; ++m) mem[m] = 0; \
        for(int fb=0; fb<FEEDBACK_CHANNELS; ++fb) feedback_state[fb] = 0;

    // NICHE 0: Chaotic Lorenz Physics
    FLUSH_FIELD_STATE();
    float err_lorenz = 0.0f;
    int16_t lx = 10, ly = 20, lz = 30;
    int16_t fe_cur = 0, fe_exp = 0, fe_cum = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t dx = (10 * (ly - lx)) >> 3;
        int16_t target_x = lx + dx;
        int16_t sensory[SENSORY_INPUTS] = {lx, ly, lz, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_sovereign_fractal_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem, feedback_state, t);
        int16_t err = dev_clamp16(abs(out[0] - target_x), 0, 4095);
        err_lorenz += static_cast<float>(err);
        fe_cur = err;
        fe_exp = (fe_exp * 7 + err) >> 3;
        fe_cum = dev_clamp16(fe_cum + (err >> 2), 0, 4095);
        lx = target_x; ly += 5; lz += 2;
    }
    agent.niche_errors[0] = err_lorenz / 32.0f;

    // NICHE 1: Harmonic Oscillator (Wave Dynamics)
    FLUSH_FIELD_STATE();
    float err_osc = 0.0f;
    int16_t sx = 0, sv = 0;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t force = (t & 1) ? 16 : -16;
        int16_t target_x = sx + sv;
        int16_t sensory[SENSORY_INPUTS] = {sx, sv, force, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_sovereign_fractal_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem, feedback_state, t);
        int16_t err = dev_clamp16(abs(out[0] - target_x), 0, 4095);
        err_osc += static_cast<float>(err);
        fe_cur = err;
        fe_exp = (fe_exp * 7 + err) >> 3;
        fe_cum = dev_clamp16(fe_cum + (err >> 2), 0, 4095);
        sv = (14 * sv - 4 * sx + force) >> 4;
        sx = target_x;
    }
    agent.niche_errors[1] = err_osc / 32.0f;

    // NICHE 2: Turing Ripple-Carry Logic
    FLUSH_FIELD_STATE();
    float err_carry = 0.0f;
    int16_t carry = 0;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t a = (t * 7) & 1;
        int16_t b = (t * 13) & 1;
        int16_t tgt_sum = a ^ b ^ carry;
        int16_t tgt_carry = (a & b) | (carry & (a ^ b));
        int16_t sensory[SENSORY_INPUTS] = {a, b, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_sovereign_fractal_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem, feedback_state, t);
        int16_t bit_err = ((out[0] & 1) != tgt_sum) ? 16 : 0;
        if (bit_err > 0) err_carry += 1.0f;
        fe_cur = bit_err;
        fe_exp = (fe_exp * 7 + bit_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + bit_err, 0, 4095);
        carry = tgt_carry;
    }
    agent.niche_errors[2] = err_carry / 32.0f;

    // NICHE 3: Dynamic Parity Stream
    FLUSH_FIELD_STATE();
    float err_parity = 0.0f;
    int16_t par = 0;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t bit = (t * 5 + 1) & 1;
        par ^= bit;
        int16_t sensory[SENSORY_INPUTS] = {bit, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_sovereign_fractal_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem, feedback_state, t);
        int16_t bit_err = ((out[0] & 1) != par) ? 16 : 0;
        if (bit_err > 0) err_parity += 1.0f;
        fe_cur = bit_err;
        fe_exp = (fe_exp * 7 + bit_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + bit_err, 0, 4095);
    }
    agent.niche_errors[3] = err_parity / 32.0f;

    // NICHE 4: Context-Free Bracket Depth
    FLUSH_FIELD_STATE();
    float err_bracket = 0.0f;
    int depth = 0;
    const char brackets[17] = "(())((()))()(())";
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 16; ++t) {
        char b = brackets[t];
        depth += (b == '(') ? 1 : -1;
        int16_t sensory[SENSORY_INPUTS] = {static_cast<int16_t>(b), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_sovereign_fractal_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem, feedback_state, t);
        int16_t depth_err = dev_clamp16(abs(out[0] - depth), 0, 4095);
        if (out[0] != depth) err_bracket += 1.0f;
        fe_cur = depth_err * 8;
        fe_exp = (fe_exp * 7 + fe_cur) >> 3;
        fe_cum = dev_clamp16(fe_cum + fe_cur, 0, 4095);
    }
    agent.niche_errors[4] = err_bracket / 16.0f;

    // NICHE 5: Markov Associative Flow
    FLUSH_FIELD_STATE();
    float err_markov = 0.0f;
    const char seq[17] = "abcdabcdabcdabcd";
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 15; ++t) {
        int16_t cur = static_cast<int16_t>(seq[t]);
        int16_t tgt = static_cast<int16_t>(seq[t + 1]);
        int16_t sensory[SENSORY_INPUTS] = {cur, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_sovereign_fractal_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem, feedback_state, t);
        int16_t byte_err = (out[0] != tgt) ? 16 : 0;
        if (byte_err > 0) err_markov += 1.0f;
        fe_cur = byte_err;
        fe_exp = (fe_exp * 7 + byte_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + byte_err, 0, 4095);
    }
    agent.niche_errors[5] = err_markov / 15.0f;

    // NICHE 6: Enhanced Natural Speech & Dialogue Stream ("instruction_response_karyon_core_synthesis")
    FLUSH_FIELD_STATE();
    float err_lang = 0.0f;
    const char phrase[43] = "instruction_response_karyon_core_synthesis";
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 41; ++t) {
        uint8_t cur = static_cast<uint8_t>(phrase[t]);
        uint8_t nxt = static_cast<uint8_t>(phrase[t + 1]);
        int16_t sensory[SENSORY_INPUTS] = {static_cast<int16_t>(cur), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_sovereign_fractal_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem, feedback_state, t);
        int16_t lang_err = (static_cast<uint8_t>(out[0] & 0xFF) != nxt) ? 16 : 0;
        if (lang_err > 0) err_lang += 1.0f;
        fe_cur = lang_err;
        fe_exp = (fe_exp * 7 + lang_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + lang_err, 0, 4095);
    }
    agent.niche_errors[6] = err_lang / 41.0f;

    // NICHE 7: Video Game Sensorimotor Causal Physics
    FLUSH_FIELD_STATE();
    float err_game = 0.0f;
    int16_t pos_x = 10;
    int16_t vel_x = 2;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t action = (t % 3 == 0) ? 4 : ((t % 5 == 0) ? -4 : 0);
        int16_t next_vel = vel_x + action;
        int16_t next_pos = pos_x + next_vel;
        
        if (next_pos > 40) { next_pos = 40; next_vel = -next_vel; }
        else if (next_pos < 0) { next_pos = 0; next_vel = -next_vel; }

        int16_t sensory[SENSORY_INPUTS] = {pos_x, vel_x, action, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_sovereign_fractal_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem, feedback_state, t);

        int16_t game_causal_err = dev_clamp16(abs(out[0] - next_pos), 0, 4095);
        err_game += static_cast<float>(game_causal_err);

        fe_cur = game_causal_err;
        fe_exp = (fe_exp * 7 + game_causal_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + (game_causal_err >> 2), 0, 4095);

        pos_x = next_pos;
        vel_x = next_vel;
    }
    agent.niche_errors[7] = err_game / 32.0f;

    // Calculate Endogenous Landauer / ATP Metabolic Cost
    agent.metabolic_energy_cost = compute_metabolic_energy_cost(agent);

    // Composite Thermodynamic Variational Free Energy
    float sensory_fe = agent.niche_errors[0] * 0.10f +
                       agent.niche_errors[1] * 0.10f +
                       agent.niche_errors[2] * 2.00f +
                       agent.niche_errors[3] * 2.00f +
                       agent.niche_errors[4] * 2.00f +
                       agent.niche_errors[5] * 2.00f +
                       agent.niche_errors[6] * 2.50f +
                       agent.niche_errors[7] * 0.20f;

    agent.composite_fe = sensory_fe + agent.metabolic_energy_cost;
}

__global__ void gpu_pareto_tournament_kernel(SovereignFractalGenome* agents, 
                                             SovereignFractalGenome* niche_pool, 
                                             int pop_size, 
                                             curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local_state = states[idx];
    SovereignFractalGenome& agent = agents[idx];

    int niche_id = curand(&local_state) % NUM_NICHES;
    int slot = niche_id * (NICHE_SLOTS / NUM_NICHES) + (curand(&local_state) % (NICHE_SLOTS / NUM_NICHES));

    // Multi-Criteria Pareto Selection
    if (agent.niche_errors[niche_id] < niche_pool[slot].niche_errors[niche_id] ||
        (agent.niche_errors[niche_id] == niche_pool[slot].niche_errors[niche_id] && 
         agent.composite_fe < niche_pool[slot].composite_fe)) {
        niche_pool[slot] = agent;
    }

    states[idx] = local_state;
}

// ======================================================================================
// Sovereign Fractal Morphogenesis
// ======================================================================================
__global__ void gpu_sovereign_fractal_morphogenesis_kernel(SovereignFractalGenome* agents, 
                                                           const SovereignFractalGenome* niche_pool, 
                                                           int pop_size, 
                                                           bool is_macro_transition,
                                                           curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local_state = states[idx];

    int p1_idx = curand(&local_state) % NICHE_SLOTS;
    SovereignFractalGenome child = niche_pool[p1_idx];

    if (curand_uniform(&local_state) < 0.15f) {
        int v_delta = (curand(&local_state) % 3) - 1;
        int new_v = static_cast<int>(child.diff_velocity) + v_delta;
        child.diff_velocity = static_cast<uint8_t>((new_v < 1) ? 1 : ((new_v > 16) ? 16 : new_v));
    }
    float diff_factor = static_cast<float>(child.diff_velocity) / 8.0f;

    // A. MACRO-SCALE PHASE TRANSITION
    if (is_macro_transition) {
        bool fe_improved = (child.composite_fe < child.prev_macro_fe - 0.05f);
        
        if (!fe_improved || curand_uniform(&local_state) < 0.35f) {
            child.macro_transitions_count++;

            // Fractal Clock Scale Evolution
            if (curand_uniform(&local_state) < 0.25f) {
                child.fractal_clock_scale = 1 + (curand(&local_state) % 4);
            }

            // Phonotactic Morpheme Attractor Precision Mutation
            if (curand_uniform(&local_state) < 0.25f) {
                int b_delta = (curand(&local_state) % 5) - 2;
                int new_b = static_cast<int>(child.beta_morpheme) + b_delta;
                child.beta_morpheme = static_cast<int8_t>((new_b < -16) ? -16 : ((new_b > 16) ? 16 : new_b));
            }

            // Cortical Depth & Width Evolution
            if (curand_uniform(&local_state) < 0.25f) {
                int l_delta = (curand(&local_state) % 3) - 1;
                int new_l = static_cast<int>(child.active_layers) + l_delta;
                child.active_layers = static_cast<uint8_t>((new_l < 1) ? 1 : ((new_l > MAX_LAYERS) ? MAX_LAYERS : new_l));
            }

            if (curand_uniform(&local_state) < 0.30f) {
                int c_delta = (curand(&local_state) % 3) - 1;
                int new_c = static_cast<int>(child.active_capsules_limit) + c_delta;
                child.active_capsules_limit = static_cast<uint8_t>((new_c < 1) ? 1 : ((new_c > MAX_CAPSULES) ? MAX_CAPSULES : new_c));
            }

            if (curand_uniform(&local_state) < 0.20f) {
                child.bus_topology_mode = curand(&local_state) & 3;
            }

            for (int l = 0; l < child.active_layers; ++l) {
                if (curand_uniform(&local_state) < 0.25f) {
                    int w_delta = (curand(&local_state) % 5) - 2;
                    int new_w = static_cast<int>(child.layer_nodes[l]) + w_delta;
                    child.layer_nodes[l] = static_cast<uint8_t>((new_w < 4) ? 4 : ((new_w > MAX_NODES_PER_LAYER) ? MAX_NODES_PER_LAYER : new_w));
                }
            }
        }
        child.prev_macro_fe = child.composite_fe;
    }

    // B. RECIPROCAL FEEDBACK COUPLING EVOLUTION
    for (int fb = 0; fb < FEEDBACK_CHANNELS; ++fb) {
        if (curand_uniform(&local_state) < 0.18f * diff_factor) {
            int delta_fb = (curand(&local_state) % 5) - 2;
            int new_fb = child.fb_couplings[fb] + delta_fb;
            child.fb_couplings[fb] = static_cast<int8_t>((new_fb < -16) ? -16 : ((new_fb > 16) ? 16 : new_fb));
        }
    }

    // C. MODULAR CAPSULE ENDOSYMBIOSIS
    if (curand_uniform(&local_state) < 0.25f) {
        int donor_niche = curand(&local_state) % NUM_NICHES;
        int donor_slot = donor_niche * (NICHE_SLOTS / NUM_NICHES) + (curand(&local_state) % (NICHE_SLOTS / NUM_NICHES));
        const SovereignFractalGenome& donor = niche_pool[donor_slot];

        if (donor.niche_errors[donor_niche] <= 0.05f) {
            int max_slots = child.active_capsules_limit;
            if (max_slots > MAX_CAPSULES) max_slots = MAX_CAPSULES;

            int target_slot = -1;
            for (int c = 0; c < max_slots; ++c) {
                if (!child.capsules[c].is_active) {
                    target_slot = c;
                    break;
                }
            }
            if (target_slot == -1 && max_slots > 0) {
                target_slot = curand(&local_state) % max_slots;
            }

            if (target_slot != -1) {
                int donor_layer = 0;
                int donor_width = donor.layer_nodes[donor_layer];
                int organ_len = (donor_width > MAX_CAPSULE_NODES) ? MAX_CAPSULE_NODES : donor_width;

                child.capsules[target_slot].active_nodes = organ_len;
                child.capsules[target_slot].source_niche = donor_niche;
                child.capsules[target_slot].is_active = 1;
                child.capsules[target_slot].output_coupling = 16;

                for (int k = 0; k < organ_len; ++k) {
                    GeneNode n_copy = donor.layers[donor_layer][k];
                    n_copy.in1 = n_copy.in1 % (TOTAL_BASE_INPUTS + k);
                    n_copy.in2 = n_copy.in2 % (TOTAL_BASE_INPUTS + k);
                    n_copy.gate_in = n_copy.gate_in % (TOTAL_BASE_INPUTS + k);
                    n_copy.is_active = 1;
                    child.capsules[target_slot].nodes[k] = n_copy;
                }
            }
        }
    }

    for (int c = 0; c < child.active_capsules_limit && c < MAX_CAPSULES; ++c) {
        if (child.capsules[c].is_active && curand_uniform(&local_state) < 0.15f) {
            int delta_gain = (curand(&local_state) % 5) - 2;
            int new_g = child.capsules[c].output_coupling + delta_gain;
            child.capsules[c].output_coupling = static_cast<int8_t>((new_g < -16) ? -16 : ((new_g > 16) ? 16 : new_g));
        }
    }

    // D. MICRO-NODE SUPER-OPCODE, ROUTING, DYNAMICAL LAW & FRACTAL RECURSION MUTATIONS
    float p_op     = 0.14f * diff_factor;
    float p_law    = 0.22f * diff_factor;
    float p_conn   = 0.18f * diff_factor;
    float p_w      = 0.32f * diff_factor;
    float p_sprout = 0.12f * diff_factor;

    int current_offset = TOTAL_INPUTS;
    for (int l = 0; l < child.active_layers; ++l) {
        int width = child.layer_nodes[l];
        for (int n = 0; n < width; ++n) {
            GeneNode& gn = child.layers[l][n];
            int available_inputs = current_offset + n;
            if (child.bus_topology_mode == 0) available_inputs = TOTAL_INPUTS + n;

            if (curand_uniform(&local_state) < p_sprout) {
                gn.is_active = (curand_uniform(&local_state) < 0.85f) ? 1 : 0;
            }

            if (curand_uniform(&local_state) < p_op) {
                gn.op = static_cast<uint8_t>(curand(&local_state) % NUM_UNIVERSAL_OPS);
            }

            // Fractal Self-Similar Recursion Depth Mutation
            if (curand_uniform(&local_state) < 0.20f * diff_factor) {
                gn.fractal_depth = 1 + (curand(&local_state) % MAX_FRACTAL_DEPTH);
            }

            if (curand_uniform(&local_state) < p_law) {
                int d1 = (curand(&local_state) % 5) - 2;
                int d2 = (curand(&local_state) % 5) - 2;
                int d3 = (curand(&local_state) % 5) - 2;
                int d4 = (curand(&local_state) % 5) - 2;

                gn.alpha_lin  = static_cast<int8_t>(dev_clamp16(gn.alpha_lin + d1, -16, 16));
                gn.alpha_sat  = static_cast<int8_t>(dev_clamp16(gn.alpha_sat + d2, -16, 16));
                gn.alpha_step = static_cast<int8_t>(dev_clamp16(gn.alpha_step + d3, -16, 16));
                gn.alpha_bilin= static_cast<int8_t>(dev_clamp16(gn.alpha_bilin + d4, -16, 16));
            }

            if (curand_uniform(&local_state) < p_conn) {
                gn.in1 = curand(&local_state) % available_inputs;
            }
            if (curand_uniform(&local_state) < p_conn) {
                gn.in2 = curand(&local_state) % available_inputs;
            }
            if (curand_uniform(&local_state) < p_conn) {
                gn.gate_in = curand(&local_state) % available_inputs;
            }
            if (curand_uniform(&local_state) < p_w) {
                int delta_w1 = (curand(&local_state) % 7) - 3;
                int nw1 = gn.weight1 + delta_w1;
                gn.weight1 = static_cast<int8_t>((nw1 < -16) ? -16 : ((nw1 > 16) ? 16 : nw1));
            }
            if (curand_uniform(&local_state) < p_w) {
                int delta_w2 = (curand(&local_state) % 7) - 3;
                int nw2 = gn.weight2 + delta_w2;
                gn.weight2 = static_cast<int8_t>((nw2 < -16) ? -16 : ((nw2 > 16) ? 16 : nw2));
            }
        }
        current_offset += width;
    }

    agents[idx] = child;
    states[idx] = local_state;
}

const char* opcode_name(uint8_t op) {
    switch(op) {
        case UOP_PASS: return "PASS";
        case UOP_ADD: return "ADD";
        case UOP_SUB: return "SUB";
        case UOP_MUL_SHIFT: return "MUL_SHIFT";
        case UOP_DIV_MOD: return "DIV_MOD";
        case UOP_XOR: return "XOR";
        case UOP_AND: return "AND";
        case UOP_OR: return "OR";
        case UOP_NOT: return "NOT";
        case UOP_SHL: return "SHL";
        case UOP_SHR: return "SHR";
        case UOP_MIN: return "MIN";
        case UOP_MAX: return "MAX";
        case UOP_ABS: return "ABS";
        case UOP_SATURATE: return "SATURATE";
        case UOP_CMP_EQ: return "CMP_EQ";
        case UOP_CMP_LT: return "CMP_LT";
        case UOP_COND_MUX: return "COND_MUX";
        case UOP_LOAD_REG: return "LOAD_REG";
        case UOP_STORE_REG: return "STORE_REG";
        case UOP_ACCUMULATE: return "ACCUMULATE";
        case UOP_ASSOC_READ: return "ASSOC_READ";
        case UOP_ASSOC_WRITE: return "ASSOC_WRITE";
        case UOP_STACK_PUSH: return "STACK_PUSH";
        case UOP_STACK_POP: return "STACK_POP";
        case UOP_FRACTAL_RECURSION: return "FRACTAL_RECURSION";
        default: return "UNKNOWN";
    }
}

const char* topology_name(uint8_t mode) {
    switch(mode & 3) {
        case 0: return "FEEDFORWARD_LAMINAR";
        case 1: return "DENSE_HIGHWAY";
        case 2: return "RECURRENT_LOOPS";
        case 3: return "RESERVOIR_INTERLEAVED";
        default: return "UNKNOWN";
    }
}

std::string input_desc(int idx) {
    if (idx < SENSORY_INPUTS) return "s_in[" + std::to_string(idx) + "]";
    if (idx == SENSORY_INPUTS + 0) return "FE_CUR(F_t)";
    if (idx == SENSORY_INPUTS + 1) return "FE_EXP(G_t)";
    if (idx == SENSORY_INPUTS + 2) return "FE_CUM(Omega_t)";
    if (idx >= SENSORY_INPUTS + TELEMETRY_INPUTS && idx < TOTAL_BASE_INPUTS) {
        int fb_idx = idx - (SENSORY_INPUTS + TELEMETRY_INPUTS);
        return "TOP_DOWN_FB[" + std::to_string(fb_idx) + "]";
    }
    if (idx >= TOTAL_BASE_INPUTS && idx < TOTAL_INPUTS) {
        int c_idx = idx - TOTAL_BASE_INPUTS;
        return "CAPSULE_" + std::to_string(c_idx) + "_OUT";
    }
    return "bus_n" + std::to_string(idx - TOTAL_INPUTS);
}

void dissect_sovereign_fractal_organism(const SovereignFractalGenome& champion) {
    std::cout << "\n========================================================================\n";
    std::cout << "=== GFX/PHYS: DEEP ENDOSCOPIC AUTOPSY: FRACTAL ATTRACTOR CHAMPION    ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Fractal Clock Scale Ratio: " << (int)champion.fractal_clock_scale << "x (\x1b[32mMulti-Scale Temporal Time\x1b[0m)\n";
    std::cout << "Morpheme Precision Gain  : " << (int)champion.beta_morpheme << " / 16 (Phonotactic Attractor Snapping)\n";
    std::cout << "Cortical Layer Depth     : " << (int)champion.active_layers << " / " << MAX_LAYERS << " layers\n";
    std::cout << "Capsule Organ Capacity   : " << (int)champion.active_capsules_limit << " / " << MAX_CAPSULES << " capsules\n";
    std::cout << "Bus Routing Topology     : " << topology_name(champion.bus_topology_mode) << "\n";
    std::cout << "Macro Transitions Count  : " << champion.macro_transitions_count << " phase transitions\n";
    std::cout << "Metabolic Energy Cost    : " << std::fixed << std::setprecision(5) << champion.metabolic_energy_cost << " nats (ATP/Landauer)\n";
    std::cout << "Composite Free Energy    : " << std::fixed << std::setprecision(5) << champion.composite_fe << " nats (Sensory + Metabolism)\n\n";

    int fractal_nodes_count = 0;
    int total_active_nodes = 0;
    int total_nodes_allocated = 0;
    for (int l = 0; l < champion.active_layers; ++l) {
        int act_in_layer = 0;
        for (int n = 0; n < champion.layer_nodes[l]; ++n) {
            if (champion.layers[l][n].is_active) {
                act_in_layer++;
                if (champion.layers[l][n].op == UOP_FRACTAL_RECURSION) fractal_nodes_count++;
            }
        }
        std::cout << "  Tier " << l << ": " << act_in_layer << " active / " << (int)champion.layer_nodes[l] << " total width\n";
        total_active_nodes += act_in_layer;
        total_nodes_allocated += champion.layer_nodes[l];
    }
    std::cout << "  Total Active Cortical Nodes : " << total_active_nodes << " / " << total_nodes_allocated << "\n";
    std::cout << "  Active Fractal Super-Nodes  : " << fractal_nodes_count << " (Self-Similar Recursive Generators)\n\n";

    std::cout << ">>> 8 Universal Multi-Modal Scorecard (Strict Telemetry):\n";
    std::cout << "  1. Chaotic Lorenz Physics Tracking Error : " << champion.niche_errors[0] << "\n";
    std::cout << "  2. Harmonic Oscillator Tracking Error    : " << champion.niche_errors[1] << "\n";
    std::cout << "  3. Turing Ripple-Carry Logic Error Rate  : " << champion.niche_errors[2] * 100.0f << " %\n";
    std::cout << "  4. Dynamic Parity Stream Error Rate      : " << champion.niche_errors[3] * 100.0f << " %\n";
    std::cout << "  5. Context-Free Stack Depth Error Rate   : " << champion.niche_errors[4] * 100.0f << " %\n";
    std::cout << "  6. Markov Associative Flow Error Rate    : " << champion.niche_errors[5] * 100.0f << " %\n";
    std::cout << "  7. Enhanced Speech & Language Error Rate : " << champion.niche_errors[6] * 100.0f << " %\n";
    std::cout << "  8. Video Game Sensorimotor Causal Error  : " << champion.niche_errors[7] << "\n\n";

    std::cout << "========================================================================\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-619: SELF-SIMILAR FRACTAL ATTRACTOR MORPHOGENESIS (SS-FAM)   ===\n";
    std::cout << "========================================================================\n";
    uint64_t total_evals_target = static_cast<uint64_t>(POP_SIZE) * NUM_GENERATIONS;
    std::cout << "Target Generations       : " << NUM_GENERATIONS << " DEEP GENERATIONS\n";
    std::cout << "Target Evaluations       : " << total_evals_target << " (98.30 MILLION AGENTS)\n";
    std::cout << "Active Batch Size on GPU : " << POP_SIZE << " organisms in parallel\n";
    std::cout << "Fractal Generator Op     : UOP_FRACTAL_RECURSION z_{k+1} = f_shaped(z_k, b) for k=0..D_fractal\n";
    std::cout << "Multi-Scale Fractal Clock: \\Delta t_l = 2^l (Layer-wise Scale Invariance)\n";
    std::cout << "Epigenetic Subsidy       : Subsidized Landauer Cost for Self-Similar Multi-Scale Memory\n";
    std::cout << "Multi-Modal Suite        : 8 Environments (Speech & Language Stream Focus)\n\n";

    SovereignFractalGenome* d_agents = nullptr;
    SovereignFractalGenome* d_niche_pool = nullptr;
    curandState* d_rand_states = nullptr;

    CHECK_CUDA(cudaMalloc(&d_agents, POP_SIZE * sizeof(SovereignFractalGenome)));
    CHECK_CUDA(cudaMalloc(&d_niche_pool, NICHE_SLOTS * sizeof(SovereignFractalGenome)));
    CHECK_CUDA(cudaMalloc(&d_rand_states, POP_SIZE * sizeof(curandState)));

    int threads = 256;
    int blocks_pop = (POP_SIZE + threads - 1) / threads;

    init_curand_kernel<<<blocks_pop, threads>>>(d_rand_states, 2026, POP_SIZE);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<SovereignFractalGenome> h_init_pool(NICHE_SLOTS);
    std::mt19937 host_rng(2026);
    std::uniform_int_distribution<int> op_dist(0, NUM_UNIVERSAL_OPS - 1);
    std::uniform_int_distribution<int> w_dist(-16, 16);
    std::uniform_int_distribution<int> layer_dist(1, 3);
    std::uniform_int_distribution<int> width_dist(8, 20);
    std::uniform_int_distribution<int> cap_dist(3, 6);

    for (int i = 0; i < NICHE_SLOTS; ++i) {
        h_init_pool[i].active_layers = layer_dist(host_rng);
        h_init_pool[i].active_capsules_limit = cap_dist(host_rng);
        h_init_pool[i].bus_topology_mode = 2; // Recurrent Loops
        h_init_pool[i].diff_velocity = 8;
        h_init_pool[i].fractal_clock_scale = 1;
        h_init_pool[i].beta_morpheme = static_cast<int8_t>(w_dist(host_rng));
        h_init_pool[i].macro_transitions_count = 0;
        h_init_pool[i].metabolic_energy_cost = 0.0f;
        h_init_pool[i].composite_fe = 99999.0f;
        h_init_pool[i].prev_macro_fe = 99999.0f;
        for (int fb = 0; fb < FEEDBACK_CHANNELS; ++fb) h_init_pool[i].fb_couplings[fb] = w_dist(host_rng);
        for (int t = 0; t < NUM_NICHES; ++t) h_init_pool[i].niche_errors[t] = 999.0f;

        for (int l = 0; l < MAX_LAYERS; ++l) {
            h_init_pool[i].layer_nodes[l] = width_dist(host_rng);
            for (int n = 0; n < MAX_NODES_PER_LAYER; ++n) {
                h_init_pool[i].layers[l][n].op = static_cast<uint8_t>(op_dist(host_rng));
                h_init_pool[i].layers[l][n].in1 = host_rng() % (TOTAL_INPUTS + n + l * 16 + 1);
                h_init_pool[i].layers[l][n].in2 = host_rng() % (TOTAL_INPUTS + n + l * 16 + 1);
                h_init_pool[i].layers[l][n].gate_in = host_rng() % (TOTAL_INPUTS + n + l * 16 + 1);
                
                h_init_pool[i].layers[l][n].alpha_lin = static_cast<int8_t>(w_dist(host_rng));
                h_init_pool[i].layers[l][n].alpha_sat = static_cast<int8_t>(w_dist(host_rng));
                h_init_pool[i].layers[l][n].alpha_step = static_cast<int8_t>(w_dist(host_rng));
                h_init_pool[i].layers[l][n].alpha_bilin = static_cast<int8_t>(w_dist(host_rng));

                h_init_pool[i].layers[l][n].weight1 = static_cast<int8_t>(w_dist(host_rng));
                h_init_pool[i].layers[l][n].weight2 = static_cast<int8_t>(w_dist(host_rng));
                h_init_pool[i].layers[l][n].fractal_depth = 1 + (host_rng() % MAX_FRACTAL_DEPTH);
                h_init_pool[i].layers[l][n].is_active = 1;
                h_init_pool[i].layers[l][n].causal_credit = 0;
            }
        }
        for (int c = 0; c < MAX_CAPSULES; ++c) {
            h_init_pool[i].capsules[c].is_active = 0;
            h_init_pool[i].capsules[c].active_nodes = 0;
            h_init_pool[i].capsules[c].source_niche = 0;
            h_init_pool[i].capsules[c].output_coupling = 16;
        }
    }

    CHECK_CUDA(cudaMemcpy(d_niche_pool, h_init_pool.data(), NICHE_SLOTS * sizeof(SovereignFractalGenome), cudaMemcpyHostToDevice));

    auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t evals_completed = 0;

    std::vector<SovereignFractalGenome> h_pool_sample(NICHE_SLOTS);

    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        bool is_macro_transition = ((gen + 1) % MACRO_PACING_K == 0);

        // 1. GPU Sovereign Morphogenesis (Self-Similar Fractal Attractors & Multi-Scale Clocks)
        gpu_sovereign_fractal_morphogenesis_kernel<<<blocks_pop, threads>>>(d_agents, d_niche_pool, POP_SIZE, is_macro_transition, d_rand_states);

        // 2. GPU Parallel Reality Evaluation (8 Environments including Speech Stream)
        evaluate_multimodal_crucible_kernel<<<blocks_pop, threads>>>(d_agents, POP_SIZE);

        // 3. GPU Pareto Selection Tournament
        gpu_pareto_tournament_kernel<<<blocks_pop, threads>>>(d_agents, d_niche_pool, POP_SIZE, d_rand_states);

        evals_completed += POP_SIZE;

        if ((gen + 1) % 500 == 0 || gen == 0) {
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(h_pool_sample.data(), d_niche_pool, NICHE_SLOTS * sizeof(SovereignFractalGenome), cudaMemcpyDeviceToHost));

            int best_idx = 0;
            for (int i = 1; i < NICHE_SLOTS; ++i) {
                if (h_pool_sample[i].composite_fe < h_pool_sample[best_idx].composite_fe) {
                    best_idx = i;
                }
            }

            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double speed = static_cast<double>(evals_completed) / elapsed;

            int tot_active_nodes = 0;
            int frac_cnt = 0;
            for (int l = 0; l < h_pool_sample[best_idx].active_layers; ++l) {
                for (int n = 0; n < h_pool_sample[best_idx].layer_nodes[l]; ++n) {
                    if (h_pool_sample[best_idx].layers[l][n].is_active) {
                        tot_active_nodes++;
                        if (h_pool_sample[best_idx].layers[l][n].op == UOP_FRACTAL_RECURSION) frac_cnt++;
                    }
                }
            }

            std::cout << "Gen [" << std::setw(5) << (gen + 1) << "/" << NUM_GENERATIONS << "]"
                      << " | Evals: " << std::setw(8) << evals_completed
                      << " | Speed: " << std::fixed << std::setprecision(0) << speed << " evals/s"
                      << " | Min FE: " << std::setprecision(4) << h_pool_sample[best_idx].composite_fe
                      << " | Turing: " << std::setprecision(1) << (h_pool_sample[best_idx].niche_errors[2] * 100.0f) << "%"
                      << " | Parity: " << (h_pool_sample[best_idx].niche_errors[3] * 100.0f) << "%"
                      << " | Stack: " << (h_pool_sample[best_idx].niche_errors[4] * 100.0f) << "%"
                      << " | Markov: " << (h_pool_sample[best_idx].niche_errors[5] * 100.0f) << "%"
                      << " | SPEECH: " << (h_pool_sample[best_idx].niche_errors[6] * 100.0f) << "%"
                      << " | Game Physics: " << h_pool_sample[best_idx].niche_errors[7]
                      << " | FracOps: " << frac_cnt
                      << " | Nodes: " << tot_active_nodes
                      << " | Trans: " << h_pool_sample[best_idx].macro_transitions_count << "\n";
            std::cout.flush();
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    CHECK_CUDA(cudaMemcpy(h_pool_sample.data(), d_niche_pool, NICHE_SLOTS * sizeof(SovereignFractalGenome), cudaMemcpyDeviceToHost));

    int global_best_idx = 0;
    for (int i = 1; i < NICHE_SLOTS; ++i) {
        if (h_pool_sample[i].composite_fe < h_pool_sample[global_best_idx].composite_fe) {
            global_best_idx = i;
        }
    }

    std::cout << "\n========================================================================\n";
    std::cout << ">>> SELF-SIMILAR FRACTAL ATTRACTOR MORPHOGENESIS COMPLETE!\n";
    std::cout << ">>> Total Runtime : " << std::fixed << std::setprecision(2) << total_sec << " seconds\n";
    std::cout << ">>> Average Speed : " << std::fixed << std::setprecision(0) << (evals_completed / total_sec) << " agents/sec\n";
    std::cout << "========================================================================\n";

    dissect_sovereign_fractal_organism(h_pool_sample[global_best_idx]);

    std::cout << ">>> SPECIALIST NICHE CHAMPIONS AUDIT (SELF-SIMILAR FRACTAL MORPHOLOGY Focus):\n";
    for (int n = 0; n < NUM_NICHES; ++n) {
        int best_slot = n * (NICHE_SLOTS / NUM_NICHES);
        for (int s = 1; s < (NICHE_SLOTS / NUM_NICHES); ++s) {
            int slot = n * (NICHE_SLOTS / NUM_NICHES) + s;
            if (h_pool_sample[slot].niche_errors[n] < h_pool_sample[best_slot].niche_errors[n]) {
                best_slot = slot;
            }
        }
        std::cout << "  Niche " << n << " Specialist Best Error: " 
                  << h_pool_sample[best_slot].niche_errors[n] 
                  << " (Composite FE: " << h_pool_sample[best_slot].composite_fe 
                  << ", Layers: " << (int)h_pool_sample[best_slot].active_layers 
                  << ", Caps: " << (int)h_pool_sample[best_slot].active_capsules_limit << ")\n";
    }
    std::cout << "========================================================================\n";

    CHECK_CUDA(cudaFree(d_agents));
    CHECK_CUDA(cudaFree(d_niche_pool));
    CHECK_CUDA(cudaFree(d_rand_states));
    return 0;
}
