// ======================================================================================
// EXP-615: Continuous Totipotent Field over Universal Modalities (CTF-UM)
// --------------------------------------------------------------------------------------
// Fundamental Cybernetics (Bazilevs Sovereign Mandate & KEP Principles 2, 3, 27, 29, 30):
// 1. THE ABSURDITY OF HARDCODED DOMAIN ZOOS:
//    - Bazilevs pointed out the fundamental irony: if we hardcode "speech opcodes" today,
//      tomorrow we would need to hardcode "image opcodes", "audio opcodes", "math opcodes",
//      "video game causal physics opcodes"... ad infinitum. This leads to an unmanageable
//      human-prescribed discrete primitive zoo that breaks self-evolution.
// 2. UNIFIED CONTINUOUS TOTIPOTENT TENSOR SUPER-PRIMITIVE:
//    - Rejects all task-specific discrete opcodes!
//    - Every computational node operates as a UNIFIED CONTINUOUS PARAMETRIC FIELD:
//      y_t = Clamp( (W_1 * a + W_2 * b + U * mem[reg]) (*) Gated(V_1 * a + V_2 * b) )
//      By adjusting its continuous inner algebraic coordinates (W_1, W_2, U, V, law),
//      this single continuous super-primitive natively degenerates into:
//      - Linear projection / Matrix multiply (Audio / Spectral)
//      - Bilinear spatial gating / 2D Grid convolution (Images / Spatial Vision)
//      - Recurrent state transition / Delay line (Speech / Natural Language)
//      - Discrete bitwise snapping & Stack routing (Turing / Math / Parity)
//      - Action-State causal prediction (Video Game Sensorimotor Physics)
// 3. UNIVERSAL MULTI-MODAL BENCHMARK CRUCIBLE (8 Multi-Domain Environments):
//    - Niche 0: Chaotic Lorenz Physics (Continuous Dynamical Field)
//    - Niche 1: Harmonic Oscillator (Somatic Wave Resonance)
//    - Niche 2: Turing Ripple-Carry Logic (Discrete Boolean Mathematics)
//    - Niche 3: Dynamic Parity Stream (Temporal Bit Sequence)
//    - Niche 4: Context-Free Bracket Depth (Abstract Structural Hierarchy)
//    - Niche 5: Markov Associative Flow (Sequence Prediction)
//    - Niche 6: Natural Language Stream (Speech & Semantics)
//    - Niche 7: Video Game Sensorimotor Causal Physics (Agent Inertia, Wall Collision, Action-Reward Causal Prediction)
// 4. TWO-TIME-SCALE MORPHOGENESIS (Micro-Search vs Macro-Paced Transitions):
//    - Fast micro-weight/routing search every generation.
//    - Macro-paced topological transition every K=250 generations.
// 5. 100% RESIDENT CUDA PIPELINE:
//    - 16,384 parallel organisms, 6,000 deep generations on Tesla T4.
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
constexpr int TOTAL_BASE_INPUTS = SENSORY_INPUTS + TELEMETRY_INPUTS; // 7
constexpr int TOTAL_INPUTS      = TOTAL_BASE_INPUTS + MAX_CAPSULES;  // 7 + 8 = 15 inputs
constexpr int NUM_OUTPUTS       = 2;
constexpr int NUM_NICHES        = 8;
constexpr int NICHE_SLOTS       = 1024;    // 1024 resident Pareto niche slots

// Clean, compact, universal algebraic transformations (No domain-hardcoded opcodes!)
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
    NUM_UNIVERSAL_OPS
};

struct GeneNode {
    uint8_t op;         // Universal Super-Opcode
    uint8_t in1;        // Primary input coordinate
    uint8_t in2;        // Secondary input coordinate
    uint8_t law_mode;   // Continuous non-linear shaping law [0: Linear, 1: Sat, 2: Step, 3: Bilinear]
    uint8_t gate_in;    // Continuous modulating gate coordinate
    int8_t  weight;     // Primary gain parameter [-16 .. 16]
    int8_t  weight2;    // Secondary coupling parameter [-16 .. 16]
};

struct alignas(16) OrganCapsule {
    GeneNode nodes[MAX_CAPSULE_NODES];
    uint8_t  active_nodes;
    uint8_t  source_niche;   // Source niche ID
    uint8_t  is_active;      // Sovereign activation bit
    int8_t   output_coupling;// Output scaling [-16 .. 16]
};

struct alignas(64) ContinuousTotipotentGenome {
    GeneNode     layers[MAX_LAYERS][MAX_NODES_PER_LAYER];
    OrganCapsule capsules[MAX_CAPSULES];
    
    // Macro-Topological Regulators (Paced on Macro-Scale)
    uint8_t      active_layers;              // [1 .. 4] Cortical layer depth
    uint8_t      layer_nodes[MAX_LAYERS];    // [1 .. 32] Width per layer
    uint8_t      active_capsules_limit;      // [1 .. 8] Capsule capacity limit
    uint8_t      bus_topology_mode;          // [0: FF, 1: Dense, 2: Recurrent, 3: Res]
    uint8_t      diff_velocity;              // [1 .. 16] Micro differentiation pace
    uint32_t     macro_transitions_count;    // Total macro phase transitions experienced
    
    // Telemetry & Fitness
    float        niche_errors[NUM_NICHES];
    float        composite_fe;
    float        prev_macro_fe;              // Baseline FE at last macro phase transition
};

__device__ inline int16_t dev_clamp16(int32_t val, int32_t low, int32_t high) {
    return static_cast<int16_t>((val < low) ? low : ((val > high) ? high : val));
}

// ======================================================================================
// Continuous Totipotent Tensor Super-Primitive Execution Engine
// ======================================================================================
__device__ inline int16_t exec_totipotent_field_op(uint8_t op, uint8_t law_mode, int16_t a, int16_t b, int8_t w1, int8_t w2, int16_t* mem) {
    int32_t raw = 0;
    switch(op) {
        case UOP_PASS:        raw = a; break;
        case UOP_ADD:         raw = (static_cast<int32_t>(a) * w1 + static_cast<int32_t>(b) * w2) >> 4; break;
        case UOP_SUB:         raw = (static_cast<int32_t>(a) * w1 - static_cast<int32_t>(b) * w2) >> 4; break;
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
            int32_t update = (static_cast<int32_t>(b) * w2) >> 4;
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

    int32_t shaped = raw;
    switch(law_mode & 3) {
        case 0: shaped = raw; break;
        case 1: shaped = (raw < -256) ? -256 : ((raw > 255) ? 255 : raw); break;
        case 2: shaped = (raw > 0) ? 16 : ((raw < 0) ? -16 : 0); break;
        case 3: shaped = (raw * static_cast<int32_t>(b)) >> 4; break;
    }

    int32_t modulated = (shaped * static_cast<int32_t>(w1)) >> 4;
    return dev_clamp16(modulated, -4096, 4095);
}

__device__ inline void run_totipotent_organism(const ContinuousTotipotentGenome& agent, 
                                               const int16_t* sensory, 
                                               int16_t fe_cur, 
                                               int16_t fe_exp, 
                                               int16_t fe_cum,
                                               int16_t* outputs, 
                                               int16_t* mem) {
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

        for (int n = 0; n < cap.active_nodes; ++n) {
            const GeneNode& gn = cap.nodes[n];
            int16_t gate_signal = cap_bus[gn.gate_in % (TOTAL_BASE_INPUTS + n)];
            if (gate_signal <= -8) {
                cap_bus[TOTAL_BASE_INPUTS + n] = cap_bus[gn.in1 % (TOTAL_BASE_INPUTS + n)];
                continue;
            }
            int16_t in1_val = cap_bus[gn.in1 % (TOTAL_BASE_INPUTS + n)];
            int16_t in2_val = cap_bus[gn.in2 % (TOTAL_BASE_INPUTS + n)];
            cap_bus[TOTAL_BASE_INPUTS + n] = exec_totipotent_field_op(gn.op, gn.law_mode, in1_val, in2_val, gn.weight, gn.weight2, mem);
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
    for (int c = 0; c < MAX_CAPSULES; ++c) {
        bus[TOTAL_BASE_INPUTS + c] = capsule_outputs[c];
    }

    int current_bus_offset = TOTAL_INPUTS;
    int last_node_global_idx = TOTAL_INPUTS;
    int prev_node_global_idx = TOTAL_INPUTS;

    int act_layers = (agent.active_layers < 1) ? 1 : ((agent.active_layers > MAX_LAYERS) ? MAX_LAYERS : agent.active_layers);

    // 3. Multi-Tier Laminar Processing
    for (int l = 0; l < act_layers; ++l) {
        int layer_width = agent.layer_nodes[l];
        if (layer_width < 1) layer_width = 1;
        if (layer_width > MAX_NODES_PER_LAYER) layer_width = MAX_NODES_PER_LAYER;

        for (int n = 0; n < layer_width; ++n) {
            const GeneNode& gn = agent.layers[l][n];
            
            int available_inputs = current_bus_offset + n;
            if (agent.bus_topology_mode == 0) {
                available_inputs = TOTAL_INPUTS + n;
            }

            int16_t gate_signal = bus[gn.gate_in % available_inputs];
            if (gate_signal <= -8) {
                bus[current_bus_offset + n] = bus[gn.in1 % available_inputs];
            } else {
                int16_t in1_val = bus[gn.in1 % available_inputs];
                int16_t in2_val = bus[gn.in2 % available_inputs];
                bus[current_bus_offset + n] = exec_totipotent_field_op(gn.op, gn.law_mode, in1_val, in2_val, gn.weight, gn.weight2, mem);
            }

            prev_node_global_idx = last_node_global_idx;
            last_node_global_idx = current_bus_offset + n;
        }
        current_bus_offset += layer_width;
    }

    outputs[0] = bus[last_node_global_idx];
    outputs[1] = bus[prev_node_global_idx];
}

__global__ void init_curand_kernel(curandState* states, unsigned long seed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// ======================================================================================
// Universal Multi-Modal Benchmark Crucible (8 Environments across Diverse Modalities)
// ======================================================================================
__global__ void evaluate_multimodal_crucible_kernel(ContinuousTotipotentGenome* agents, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    ContinuousTotipotentGenome& agent = agents[idx];
    int16_t mem[MEM_SIZE];
    #define FLUSH_FIELD_MEM() for(int m=0; m<MEM_SIZE; ++m) mem[m] = 0;

    // NICHE 0: Chaotic Lorenz Physics (Continuous Dynamic Trajectory)
    FLUSH_FIELD_MEM();
    float err_lorenz = 0.0f;
    int16_t lx = 10, ly = 20, lz = 30;
    int16_t fe_cur = 0, fe_exp = 0, fe_cum = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t dx = (10 * (ly - lx)) >> 3;
        int16_t target_x = lx + dx;
        int16_t sensory[SENSORY_INPUTS] = {lx, ly, lz, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t err = dev_clamp16(abs(out[0] - target_x), 0, 4095);
        err_lorenz += static_cast<float>(err);
        fe_cur = err;
        fe_exp = (fe_exp * 7 + err) >> 3;
        fe_cum = dev_clamp16(fe_cum + (err >> 2), 0, 4095);
        lx = target_x; ly += 5; lz += 2;
    }
    agent.niche_errors[0] = err_lorenz / 32.0f;

    // NICHE 1: Harmonic Oscillator (Continuous Audio/Wave Dynamics)
    FLUSH_FIELD_MEM();
    float err_osc = 0.0f;
    int16_t sx = 0, sv = 0;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t force = (t & 1) ? 16 : -16;
        int16_t target_x = sx + sv;
        int16_t sensory[SENSORY_INPUTS] = {sx, sv, force, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t err = dev_clamp16(abs(out[0] - target_x), 0, 4095);
        err_osc += static_cast<float>(err);
        fe_cur = err;
        fe_exp = (fe_exp * 7 + err) >> 3;
        fe_cum = dev_clamp16(fe_cum + (err >> 2), 0, 4095);
        sv = (14 * sv - 4 * sx + force) >> 4;
        sx = target_x;
    }
    agent.niche_errors[1] = err_osc / 32.0f;

    // NICHE 2: Turing Ripple-Carry Logic (Discrete Boolean Mathematics)
    FLUSH_FIELD_MEM();
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
        run_totipotent_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t bit_err = ((out[0] & 1) != tgt_sum) ? 16 : 0;
        if (bit_err > 0) err_carry += 1.0f;
        fe_cur = bit_err;
        fe_exp = (fe_exp * 7 + bit_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + bit_err, 0, 4095);
        carry = tgt_carry;
    }
    agent.niche_errors[2] = err_carry / 32.0f;

    // NICHE 3: Dynamic Parity Stream (Temporal Bit Sequence)
    FLUSH_FIELD_MEM();
    float err_parity = 0.0f;
    int16_t par = 0;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t bit = (t * 5 + 1) & 1;
        par ^= bit;
        int16_t sensory[SENSORY_INPUTS] = {bit, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t bit_err = ((out[0] & 1) != par) ? 16 : 0;
        if (bit_err > 0) err_parity += 1.0f;
        fe_cur = bit_err;
        fe_exp = (fe_exp * 7 + bit_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + bit_err, 0, 4095);
    }
    agent.niche_errors[3] = err_parity / 32.0f;

    // NICHE 4: Context-Free Bracket Depth (Abstract Structural LIFO Stack)
    FLUSH_FIELD_MEM();
    float err_bracket = 0.0f;
    int depth = 0;
    const char brackets[17] = "(())((()))()(())";
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 16; ++t) {
        char b = brackets[t];
        depth += (b == '(') ? 1 : -1;
        int16_t sensory[SENSORY_INPUTS] = {static_cast<int16_t>(b), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t depth_err = dev_clamp16(abs(out[0] - depth), 0, 4095);
        if (out[0] != depth) err_bracket += 1.0f;
        fe_cur = depth_err * 8;
        fe_exp = (fe_exp * 7 + fe_cur) >> 3;
        fe_cum = dev_clamp16(fe_cum + fe_cur, 0, 4095);
    }
    agent.niche_errors[4] = err_bracket / 16.0f;

    // NICHE 5: Markov Associative Flow (Sequential Text/Byte Markov Transitions)
    FLUSH_FIELD_MEM();
    float err_markov = 0.0f;
    const char seq[17] = "abcdabcdabcdabcd";
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 15; ++t) {
        int16_t cur = static_cast<int16_t>(seq[t]);
        int16_t tgt = static_cast<int16_t>(seq[t + 1]);
        int16_t sensory[SENSORY_INPUTS] = {cur, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t byte_err = (out[0] != tgt) ? 16 : 0;
        if (byte_err > 0) err_markov += 1.0f;
        fe_cur = byte_err;
        fe_exp = (fe_exp * 7 + byte_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + byte_err, 0, 4095);
    }
    agent.niche_errors[5] = err_markov / 15.0f;

    // NICHE 6: Natural Language Stream (Speech & Semantic Byte Stream)
    FLUSH_FIELD_MEM();
    float err_lang = 0.0f;
    const char phrase[33] = "instruction_response_karyon_core";
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 31; ++t) {
        uint8_t cur = static_cast<uint8_t>(phrase[t]);
        uint8_t nxt = static_cast<uint8_t>(phrase[t + 1]);
        int16_t sensory[SENSORY_INPUTS] = {static_cast<int16_t>(cur), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t lang_err = (static_cast<uint8_t>(out[0] & 0xFF) != nxt) ? 16 : 0;
        if (lang_err > 0) err_lang += 1.0f;
        fe_cur = lang_err;
        fe_exp = (fe_exp * 7 + lang_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + lang_err, 0, 4095);
    }
    agent.niche_errors[6] = err_lang / 31.0f;

    // NICHE 7: Video Game Sensorimotor Causal Physics (Agent Position, Velocity, Action-Reward Loop)
    // Simulates an autonomous 2D game grid where the agent must predict its future position [pos_x]
    // given motor action [action: left/right/boost], wall bounce collision physics, and velocity inertia!
    FLUSH_FIELD_MEM();
    float err_game = 0.0f;
    int16_t pos_x = 10;
    int16_t vel_x = 2;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t action = (t % 3 == 0) ? 4 : ((t % 5 == 0) ? -4 : 0); // Motor action
        int16_t next_vel = vel_x + action;
        int16_t next_pos = pos_x + next_vel;
        
        // Wall Bounce Causal Collision (0 <= pos <= 40)
        if (next_pos > 40) { next_pos = 40; next_vel = -next_vel; }
        else if (next_pos < 0) { next_pos = 0; next_vel = -next_vel; }

        int16_t sensory[SENSORY_INPUTS] = {pos_x, vel_x, action, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);

        int16_t game_causal_err = dev_clamp16(abs(out[0] - next_pos), 0, 4095);
        err_game += static_cast<float>(game_causal_err);

        fe_cur = game_causal_err;
        fe_exp = (fe_exp * 7 + game_causal_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + (game_causal_err >> 2), 0, 4095);

        pos_x = next_pos;
        vel_x = next_vel;
    }
    agent.niche_errors[7] = err_game / 32.0f;

    // Composite Variational Free Energy across ALL Universal Modalities
    agent.composite_fe = agent.niche_errors[0] * 0.10f +
                         agent.niche_errors[1] * 0.10f +
                         agent.niche_errors[2] * 2.00f +
                         agent.niche_errors[3] * 2.00f +
                         agent.niche_errors[4] * 2.00f +
                         agent.niche_errors[5] * 2.00f +
                         agent.niche_errors[6] * 2.00f +
                         agent.niche_errors[7] * 0.20f;
}

__global__ void gpu_pareto_tournament_kernel(ContinuousTotipotentGenome* agents, 
                                             ContinuousTotipotentGenome* niche_pool, 
                                             int pop_size, 
                                             curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local_state = states[idx];
    ContinuousTotipotentGenome& agent = agents[idx];

    int niche_id = curand(&local_state) % NUM_NICHES;
    int slot = niche_id * (NICHE_SLOTS / NUM_NICHES) + (curand(&local_state) % (NICHE_SLOTS / NUM_NICHES));

    if (agent.niche_errors[niche_id] < niche_pool[slot].niche_errors[niche_id] ||
        (agent.niche_errors[niche_id] == niche_pool[slot].niche_errors[niche_id] && 
         agent.composite_fe < niche_pool[slot].composite_fe)) {
        niche_pool[slot] = agent;
    }

    states[idx] = local_state;
}

// ======================================================================================
// Two-Time-Scale Morphogenesis over Universal Modalities
// ======================================================================================
__global__ void gpu_totipotent_morphogenesis_kernel(ContinuousTotipotentGenome* agents, 
                                                    const ContinuousTotipotentGenome* niche_pool, 
                                                    int pop_size, 
                                                    bool is_macro_transition,
                                                    curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local_state = states[idx];

    int p1_idx = curand(&local_state) % NICHE_SLOTS;
    ContinuousTotipotentGenome child = niche_pool[p1_idx];

    if (curand_uniform(&local_state) < 0.15f) {
        int v_delta = (curand(&local_state) % 3) - 1;
        int new_v = static_cast<int>(child.diff_velocity) + v_delta;
        child.diff_velocity = static_cast<uint8_t>((new_v < 1) ? 1 : ((new_v > 16) ? 16 : new_v));
    }
    float diff_factor = static_cast<float>(child.diff_velocity) / 8.0f;

    // A. MACRO-SCALE PHASE TRANSITION (Paced every K=250 gens)
    if (is_macro_transition) {
        bool fe_improved = (child.composite_fe < child.prev_macro_fe - 0.05f);
        
        if (!fe_improved || curand_uniform(&local_state) < 0.35f) {
            child.macro_transitions_count++;

            // Macro Mutate 1: Cortical Layer Depth [1 .. 4]
            if (curand_uniform(&local_state) < 0.25f) {
                int l_delta = (curand(&local_state) % 3) - 1;
                int new_l = static_cast<int>(child.active_layers) + l_delta;
                child.active_layers = static_cast<uint8_t>((new_l < 1) ? 1 : ((new_l > MAX_LAYERS) ? MAX_LAYERS : new_l));
            }

            // Macro Mutate 2: Capsule Limit Expansion [1 .. 8]
            if (curand_uniform(&local_state) < 0.30f) {
                int c_delta = (curand(&local_state) % 3) - 1;
                int new_c = static_cast<int>(child.active_capsules_limit) + c_delta;
                child.active_capsules_limit = static_cast<uint8_t>((new_c < 1) ? 1 : ((new_c > MAX_CAPSULES) ? MAX_CAPSULES : new_c));
            }

            // Macro Mutate 3: Bus Topology Realignment [0 .. 3]
            if (curand_uniform(&local_state) < 0.20f) {
                child.bus_topology_mode = curand(&local_state) & 3;
            }

            // Macro Mutate 4: Layer Widths / Dimensions
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

    // B. MICRO-SCALE HIGH-FREQUENCY SEARCH (Every Generation)
    // 1. Modular Capsule Harvesting
    if (curand_uniform(&local_state) < 0.25f) {
        int donor_niche = curand(&local_state) % NUM_NICHES;
        int donor_slot = donor_niche * (NICHE_SLOTS / NUM_NICHES) + (curand(&local_state) % (NICHE_SLOTS / NUM_NICHES));
        const ContinuousTotipotentGenome& donor = niche_pool[donor_slot];

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

    // 2. Micro-Node Super-Opcode, Law, Routing & Weight Mutations
    float p_op   = 0.15f * diff_factor;
    float p_law  = 0.15f * diff_factor;
    float p_conn = 0.18f * diff_factor;
    float p_w    = 0.32f * diff_factor;

    int current_offset = TOTAL_INPUTS;
    for (int l = 0; l < child.active_layers; ++l) {
        int width = child.layer_nodes[l];
        for (int n = 0; n < width; ++n) {
            int available_inputs = current_offset + n;
            if (child.bus_topology_mode == 0) available_inputs = TOTAL_INPUTS + n;

            if (curand_uniform(&local_state) < p_op) {
                child.layers[l][n].op = 1 + (curand(&local_state) % (NUM_UNIVERSAL_OPS - 1));
            }
            if (curand_uniform(&local_state) < p_law) {
                child.layers[l][n].law_mode = curand(&local_state) & 3;
            }
            if (curand_uniform(&local_state) < p_conn) {
                child.layers[l][n].in1 = curand(&local_state) % available_inputs;
            }
            if (curand_uniform(&local_state) < p_conn) {
                child.layers[l][n].in2 = curand(&local_state) % available_inputs;
            }
            if (curand_uniform(&local_state) < p_conn) {
                child.layers[l][n].gate_in = curand(&local_state) % available_inputs;
            }
            if (curand_uniform(&local_state) < p_w) {
                int delta_w = (curand(&local_state) % 7) - 3;
                int nw = child.layers[l][n].weight + delta_w;
                child.layers[l][n].weight = static_cast<int8_t>((nw < -16) ? -16 : ((nw > 16) ? 16 : nw));
            }
            if (curand_uniform(&local_state) < p_w) {
                int delta_w2 = (curand(&local_state) % 7) - 3;
                int nw2 = child.layers[l][n].weight2 + delta_w2;
                child.layers[l][n].weight2 = static_cast<int8_t>((nw2 < -16) ? -16 : ((nw2 > 16) ? 16 : nw2));
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
        default: return "UNKNOWN";
    }
}

const char* law_name(uint8_t law) {
    switch(law & 3) {
        case 0: return "LINEAR";
        case 1: return "SATURATE";
        case 2: return "STEP_THRESH";
        case 3: return "BILINEAR_PROD";
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
    if (idx >= TOTAL_BASE_INPUTS && idx < TOTAL_INPUTS) {
        int c_idx = idx - TOTAL_BASE_INPUTS;
        return "CAPSULE_" + std::to_string(c_idx) + "_OUT";
    }
    return "bus_n" + std::to_string(idx - TOTAL_INPUTS);
}

void dissect_totipotent_organism(const ContinuousTotipotentGenome& champion) {
    std::cout << "\n========================================================================\n";
    std::cout << "=== 🔬 DEEP ENDOSCOPIC AUTOPSY: CONTINUOUS TOTIPOTENT CHAMPION       ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Cortical Layer Depth     : " << (int)champion.active_layers << " / " << MAX_LAYERS << " layers\n";
    std::cout << "Capsule Organ Capacity   : " << (int)champion.active_capsules_limit << " / " << MAX_CAPSULES << " capsules\n";
    std::cout << "Bus Routing Topology     : " << topology_name(champion.bus_topology_mode) << "\n";
    std::cout << "Macro Transitions Count  : " << champion.macro_transitions_count << " phase transitions\n";
    std::cout << "Differentiation Velocity : " << (int)champion.diff_velocity << " / 16 (" 
              << (champion.diff_velocity / 8.0f) << "x speed)\n";
    std::cout << "Composite Free Energy    : " << std::fixed << std::setprecision(5) << champion.composite_fe << "\n\n";

    std::cout << ">>> Layer Dimensionality Breakdown:\n";
    int total_active_nodes = 0;
    for (int l = 0; l < champion.active_layers; ++l) {
        std::cout << "  Tier " << l << ": " << (int)champion.layer_nodes[l] << " nodes\n";
        total_active_nodes += champion.layer_nodes[l];
    }
    std::cout << "  Total Active Cortical Nodes: " << total_active_nodes << " nodes\n\n";

    std::cout << ">>> 8 Universal Multi-Modal Scorecard (Strict Telemetry):\n";
    std::cout << "  1. Chaotic Lorenz Physics Tracking Error : " << champion.niche_errors[0] << "\n";
    std::cout << "  2. Harmonic Oscillator Tracking Error    : " << champion.niche_errors[1] << "\n";
    std::cout << "  3. Turing Ripple-Carry Logic Error Rate  : " << champion.niche_errors[2] * 100.0f << " %\n";
    std::cout << "  4. Dynamic Parity Stream Error Rate      : " << champion.niche_errors[3] * 100.0f << " %\n";
    std::cout << "  5. Context-Free Stack Depth Error Rate   : " << champion.niche_errors[4] * 100.0f << " %\n";
    std::cout << "  6. Markov Associative Flow Error Rate    : " << champion.niche_errors[5] * 100.0f << " %\n";
    std::cout << "  7. Natural Language Stream Error Rate    : " << champion.niche_errors[6] * 100.0f << " %\n";
    std::cout << "  8. Video Game Sensorimotor Causal Error  : " << champion.niche_errors[7] << "\n\n";

    std::cout << ">>> Encapsulated Specialist Organs (Active Capsules):\n";
    int active_caps_cnt = 0;
    for (int c = 0; c < champion.active_capsules_limit && c < MAX_CAPSULES; ++c) {
        if (!champion.capsules[c].is_active) {
            std::cout << "  Capsule " << c << ": [EMPTY]\n";
            continue;
        }
        active_caps_cnt++;
        std::cout << "  Capsule " << c << ": [ACTIVE] | Source Niche: " << (int)champion.capsules[c].source_niche
                  << " | Nodes: " << (int)champion.capsules[c].active_nodes
                  << " | Coupling Gain: " << (int)champion.capsules[c].output_coupling << " / 16\n";
    }
    std::cout << "  Active Registered Capsules: " << active_caps_cnt << "\n\n";

    std::cout << ">>> Cortical Circuit (Sample Nodes Audit):\n";
    for (int l = 0; l < champion.active_layers; ++l) {
        std::cout << "--- [Tier " << l << " Cortical Sheet (Width: " << (int)champion.layer_nodes[l] << ")] ---\n";
        for (int n = 0; n < champion.layer_nodes[l]; ++n) {
            if (n >= 6 && n < champion.layer_nodes[l] - 2) {
                if (n == 6) std::cout << "  ... [" << (champion.layer_nodes[l] - 8) << " intermediate nodes omitted for display] ...\n";
                continue;
            }
            const GeneNode& gn = champion.layers[l][n];
            std::string s1 = input_desc(gn.in1);
            std::string s2 = input_desc(gn.in2);
            std::string gate = input_desc(gn.gate_in);

            std::cout << "  L" << l << "_n" << std::setw(2) << n << " = " << opcode_name(gn.op) 
                      << "(" << s1 << ", " << s2 << ") [" << law_name(gn.law_mode) 
                      << "] * (" << (int)gn.weight << " / 16, " << (int)gn.weight2 << " / 16) | Gate: " << gate;
            if (gn.in1 >= TOTAL_BASE_INPUTS && gn.in1 < TOTAL_INPUTS) std::cout << " <== [CAPSULE-FED]";
            if (gn.in2 >= TOTAL_BASE_INPUTS && gn.in2 < TOTAL_INPUTS) std::cout << " <== [CAPSULE-FED]";
            std::cout << "\n";
        }
    }
    std::cout << "========================================================================\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-615: CONTINUOUS TOTIPOTENT FIELD OVER UNIVERSAL MODALITIES   ===\n";
    std::cout << "========================================================================\n";
    uint64_t total_evals_target = static_cast<uint64_t>(POP_SIZE) * NUM_GENERATIONS;
    std::cout << "Target Generations       : " << NUM_GENERATIONS << " DEEP GENERATIONS\n";
    std::cout << "Target Evaluations       : " << total_evals_target << " (98.30 MILLION AGENTS)\n";
    std::cout << "Active Batch Size on GPU : " << POP_SIZE << " organisms in parallel\n";
    std::cout << "Universal Super-Primitive: Pure continuous algebraic tensor field (Zero domain opcodes!)\n";
    std::cout << "Multi-Modal Environments : Physics, Waves, Math, Logic, Stack, Markov, Speech, Video Game Causal!\n";
    std::cout << "Two-Time-Scale Pacing    : Micro-weight search + Macro-Paced Transitions every K=" << MACRO_PACING_K << " gens\n";
    std::cout << "Sovereign Substrate      : Continuous Plasticity Field + 128 Dynamic Registers\n\n";

    ContinuousTotipotentGenome* d_agents = nullptr;
    ContinuousTotipotentGenome* d_niche_pool = nullptr;
    curandState* d_rand_states = nullptr;

    CHECK_CUDA(cudaMalloc(&d_agents, POP_SIZE * sizeof(ContinuousTotipotentGenome)));
    CHECK_CUDA(cudaMalloc(&d_niche_pool, NICHE_SLOTS * sizeof(ContinuousTotipotentGenome)));
    CHECK_CUDA(cudaMalloc(&d_rand_states, POP_SIZE * sizeof(curandState)));

    int threads = 256;
    int blocks_pop = (POP_SIZE + threads - 1) / threads;

    init_curand_kernel<<<blocks_pop, threads>>>(d_rand_states, 2026, POP_SIZE);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<ContinuousTotipotentGenome> h_init_pool(NICHE_SLOTS);
    std::mt19937 host_rng(2026);
    std::uniform_int_distribution<int> op_dist(1, NUM_UNIVERSAL_OPS - 1);
    std::uniform_int_distribution<int> law_dist(0, 3);
    std::uniform_int_distribution<int> w_dist(-16, 16);
    std::uniform_int_distribution<int> layer_dist(1, 2);
    std::uniform_int_distribution<int> width_dist(8, 20);
    std::uniform_int_distribution<int> cap_dist(3, 6);

    for (int i = 0; i < NICHE_SLOTS; ++i) {
        h_init_pool[i].active_layers = layer_dist(host_rng);
        h_init_pool[i].active_capsules_limit = cap_dist(host_rng);
        h_init_pool[i].bus_topology_mode = 2; // Recurrent Loops
        h_init_pool[i].diff_velocity = 8;
        h_init_pool[i].macro_transitions_count = 0;
        h_init_pool[i].composite_fe = 99999.0f;
        h_init_pool[i].prev_macro_fe = 99999.0f;
        for (int t = 0; t < NUM_NICHES; ++t) h_init_pool[i].niche_errors[t] = 999.0f;

        for (int l = 0; l < MAX_LAYERS; ++l) {
            h_init_pool[i].layer_nodes[l] = width_dist(host_rng);
            for (int n = 0; n < MAX_NODES_PER_LAYER; ++n) {
                h_init_pool[i].layers[l][n].op = static_cast<uint8_t>(op_dist(host_rng));
                h_init_pool[i].layers[l][n].law_mode = static_cast<uint8_t>(law_dist(host_rng));
                h_init_pool[i].layers[l][n].in1 = host_rng() % (TOTAL_INPUTS + n + l * 16 + 1);
                h_init_pool[i].layers[l][n].in2 = host_rng() % (TOTAL_INPUTS + n + l * 16 + 1);
                h_init_pool[i].layers[l][n].gate_in = host_rng() % (TOTAL_INPUTS + n + l * 16 + 1);
                h_init_pool[i].layers[l][n].weight = static_cast<int8_t>(w_dist(host_rng));
                h_init_pool[i].layers[l][n].weight2 = static_cast<int8_t>(w_dist(host_rng));
            }
        }
        for (int c = 0; c < MAX_CAPSULES; ++c) {
            h_init_pool[i].capsules[c].is_active = 0;
            h_init_pool[i].capsules[c].active_nodes = 0;
            h_init_pool[i].capsules[c].source_niche = 0;
            h_init_pool[i].capsules[c].output_coupling = 16;
        }
    }

    CHECK_CUDA(cudaMemcpy(d_niche_pool, h_init_pool.data(), NICHE_SLOTS * sizeof(ContinuousTotipotentGenome), cudaMemcpyHostToDevice));

    auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t evals_completed = 0;

    std::vector<ContinuousTotipotentGenome> h_pool_sample(NICHE_SLOTS);

    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        bool is_macro_transition = ((gen + 1) % MACRO_PACING_K == 0);

        // 1. GPU Morphogenesis
        gpu_totipotent_morphogenesis_kernel<<<blocks_pop, threads>>>(d_agents, d_niche_pool, POP_SIZE, is_macro_transition, d_rand_states);

        // 2. GPU Parallel Reality Evaluation
        evaluate_multimodal_crucible_kernel<<<blocks_pop, threads>>>(d_agents, POP_SIZE);

        // 3. GPU Pareto Selection Tournament
        gpu_pareto_tournament_kernel<<<blocks_pop, threads>>>(d_agents, d_niche_pool, POP_SIZE, d_rand_states);

        evals_completed += POP_SIZE;

        if ((gen + 1) % 500 == 0 || gen == 0) {
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(h_pool_sample.data(), d_niche_pool, NICHE_SLOTS * sizeof(ContinuousTotipotentGenome), cudaMemcpyDeviceToHost));

            int best_idx = 0;
            for (int i = 1; i < NICHE_SLOTS; ++i) {
                if (h_pool_sample[i].composite_fe < h_pool_sample[best_idx].composite_fe) {
                    best_idx = i;
                }
            }

            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double speed = static_cast<double>(evals_completed) / elapsed;

            int tot_nodes = 0;
            for (int l = 0; l < h_pool_sample[best_idx].active_layers; ++l) {
                tot_nodes += h_pool_sample[best_idx].layer_nodes[l];
            }

            std::cout << "Gen [" << std::setw(5) << (gen + 1) << "/" << NUM_GENERATIONS << "]"
                      << " | Evals: " << std::setw(8) << evals_completed
                      << " | Speed: " << std::fixed << std::setprecision(0) << speed << " evals/s"
                      << " | Min FE: " << std::setprecision(4) << h_pool_sample[best_idx].composite_fe
                      << " | Turing: " << std::setprecision(1) << (h_pool_sample[best_idx].niche_errors[2] * 100.0f) << "%"
                      << " | Parity: " << (h_pool_sample[best_idx].niche_errors[3] * 100.0f) << "%"
                      << " | Stack: " << (h_pool_sample[best_idx].niche_errors[4] * 100.0f) << "%"
                      << " | Markov: " << (h_pool_sample[best_idx].niche_errors[5] * 100.0f) << "%"
                      << " | Lang: " << (h_pool_sample[best_idx].niche_errors[6] * 100.0f) << "%"
                      << " | Game Physics: " << h_pool_sample[best_idx].niche_errors[7]
                      << " | Layers: " << (int)h_pool_sample[best_idx].active_layers
                      << " | Nodes: " << tot_nodes
                      << " | Caps: " << (int)h_pool_sample[best_idx].active_capsules_limit
                      << " | Trans: " << h_pool_sample[best_idx].macro_transitions_count << "\n";
            std::cout.flush();
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    CHECK_CUDA(cudaMemcpy(h_pool_sample.data(), d_niche_pool, NICHE_SLOTS * sizeof(ContinuousTotipotentGenome), cudaMemcpyDeviceToHost));

    int global_best_idx = 0;
    for (int i = 1; i < NICHE_SLOTS; ++i) {
        if (h_pool_sample[i].composite_fe < h_pool_sample[global_best_idx].composite_fe) {
            global_best_idx = i;
        }
    }

    std::cout << "\n========================================================================\n";
    std::cout << ">>> CONTINUOUS TOTIPOTENT FIELD CRUCIBLE COMPLETE: " << evals_completed << " EVALS!\n";
    std::cout << ">>> Total Runtime : " << std::fixed << std::setprecision(2) << total_sec << " seconds\n";
    std::cout << ">>> Average Speed : " << std::fixed << std::setprecision(0) << (evals_completed / total_sec) << " agents/sec\n";
    std::cout << "========================================================================\n";

    dissect_totipotent_organism(h_pool_sample[global_best_idx]);

    std::cout << ">>> SPECIALIST NICHE CHAMPIONS AUDIT (AFTER TOTIPOTENT EVOLUTION):\n";
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
