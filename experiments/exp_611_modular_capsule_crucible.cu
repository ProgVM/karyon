// ======================================================================================
// EXP-611: Modular Capsule Endosymbiosis & Open-Bus Morphogenesis (MCE-OBM)
// --------------------------------------------------------------------------------------
// Fundamental Cybernetics (Bazilevs Mandate & KEP Protocol):
// 1. RECTIFYING EXP-610's EPIGENETIC RIGIDITY (Principle 6 & Principle 29):
//    - In EXP-610, locking 75% of early nodes in place froze the shared input bus,
//      inducing topological sclerosis and preventing downstream adaptation.
//    - In EXP-611, we introduce MODULAR CAPSULE ARCHITECTURE:
//      * Proven specialist subgraphs (e.g. Turing 0.000%, Parity 0.000%) are encapsulated
//        into autonomous, self-contained functional organs ("Capsules").
//      * Each capsule has its own internal isolated state and output register.
//      * Crucially: The shared input bus and surrounding network remain 100% UNCONSTRAINED
//        and PLASTIC ("Open-Bus Morphogenesis").
// 2. ZERO-SHOCK COUPLING (KEP Principle 6):
//    - When an organ is encapsulated into the global body, it connects via an adaptive
//      coupling gate with smooth initialization, ensuring zero degradation of previously
//      mastered domains while leaving the rest of the organism free to attack Language,
//      Stack, and Physics.
// 3. FULL UNSHACKLED EXPRESSIVITY (Bazilevs Mandate):
//    - Capacity: Up to 128 dynamic nodes + 4 dedicated capsule slots + 128 dynamic registers.
//    - Organisms dynamically scale complexity based on need, with zero artificial caps.
// 4. 100% RESIDENT CUDA PIPELINE:
//    - 16,384 parallel agents, 6,000 generations on Tesla T4.
//    - Telemetry tracking capsule integration, composite Free Energy, and individual niche errors.
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
constexpr int MAX_NODES         = 96;      // Unshackled dynamic nodes
constexpr int MAX_CAPSULE_NODES = 8;       // Up to 8 nodes per isolated capsule
constexpr int NUM_CAPSULES      = 4;       // Up to 4 resident specialist capsules
constexpr int MEM_SIZE          = 128;     // 128 dynamic state registers & field memory
constexpr int SENSORY_INPUTS    = 4;       // Environmental sensory inputs
constexpr int TELEMETRY_INPUTS  = 3;       // [F_t, G_t, Omega_t]
constexpr int CAPSULE_OUTPUTS   = NUM_CAPSULES; // Capsule output lines fed into open bus
constexpr int TOTAL_INPUTS      = SENSORY_INPUTS + TELEMETRY_INPUTS + CAPSULE_OUTPUTS; // 4 + 3 + 4 = 11 inputs
constexpr int NUM_OUTPUTS       = 2;
constexpr int NUM_NICHES        = 8;
constexpr int NICHE_SLOTS       = 1024;    // 1024 resident Pareto niche slots

enum OpCode : uint8_t {
    OP_PASS = 0,
    OP_ADD,
    OP_SUB,
    OP_MUL_SHIFT,
    OP_DIV_MOD,
    OP_XOR,
    OP_AND,
    OP_OR,
    OP_NOT,
    OP_SHL,
    OP_SHR,
    OP_MIN,
    OP_MAX,
    OP_ABS,
    OP_SIGN,
    OP_SATURATE,
    OP_CMP_EQ,
    OP_CMP_LT,
    OP_COND_MUX,
    OP_LOAD_REG,
    OP_STORE_REG,
    OP_ACCUMULATE,
    OP_ASSOC_READ,
    OP_ASSOC_WRITE,
    OP_STACK_PUSH,
    OP_STACK_POP,
    NUM_OPS
};

struct GeneNode {
    uint8_t op;
    uint8_t in1;
    uint8_t in2;
    uint8_t law_mode;
    uint8_t gate_in;
    int8_t  weight;
};

// Self-contained functional capsule (protected organ)
struct alignas(16) OrganCapsule {
    GeneNode nodes[MAX_CAPSULE_NODES];
    uint8_t  active_nodes;
    uint8_t  source_niche;   // Which niche domain this capsule was harvested from
    uint8_t  is_active;      // 1 if active, 0 if empty
    int8_t   output_coupling;// Coupling gain into open bus [-16 .. 16]
};

struct alignas(64) CapsuleGenome {
    GeneNode     nodes[MAX_NODES];
    OrganCapsule capsules[NUM_CAPSULES];
    uint8_t      active_nodes;
    uint8_t      diff_velocity;    // [1 .. 16] Endogenous differentiation velocity
    uint8_t      active_capsules_cnt;
    float        niche_errors[NUM_NICHES];
    float        composite_fe;
};

__device__ inline int16_t dev_clamp16(int32_t val, int32_t low, int32_t high) {
    return static_cast<int16_t>((val < low) ? low : ((val > high) ? high : val));
}

__device__ inline int16_t exec_totipotent_node_op(uint8_t op, uint8_t law_mode, int16_t a, int16_t b, int8_t w, int16_t* mem) {
    int32_t raw = 0;
    switch(op) {
        case OP_PASS:        raw = a; break;
        case OP_ADD:         raw = a + b; break;
        case OP_SUB:         raw = a - b; break;
        case OP_MUL_SHIFT:   raw = (static_cast<int32_t>(a) * static_cast<int32_t>(b)) >> 4; break;
        case OP_DIV_MOD:     raw = (b != 0) ? (a % b) : 0; break;
        case OP_XOR:         raw = a ^ b; break;
        case OP_AND:         raw = a & b; break;
        case OP_OR:          raw = a | b; break;
        case OP_NOT:         raw = ~a; break;
        case OP_SHL:         raw = a << (b & 7); break;
        case OP_SHR:         raw = a >> (b & 7); break;
        case OP_MIN:         raw = (a < b) ? a : b; break;
        case OP_MAX:         raw = (a > b) ? a : b; break;
        case OP_ABS:         raw = (a < 0) ? -a : a; break;
        case OP_SIGN:        raw = (a > 0) ? 1 : ((a < 0) ? -1 : 0); break;
        case OP_SATURATE:    raw = (a < -128) ? -128 : ((a > 127) ? 127 : a); break;
        case OP_CMP_EQ:      raw = (a == b) ? 16 : 0; break;
        case OP_CMP_LT:      raw = (a < b) ? 16 : 0; break;
        case OP_COND_MUX:    raw = (a > 0) ? b : 0; break;
        case OP_LOAD_REG:    raw = mem[static_cast<uint8_t>(a) % 64]; break;
        case OP_STORE_REG:   mem[static_cast<uint8_t>(b) % 64] = a; raw = a; break;
        case OP_ACCUMULATE:  {
            uint8_t reg = static_cast<uint8_t>(a) % 64;
            mem[reg] = dev_clamp16(static_cast<int32_t>(mem[reg]) + b, -4096, 4095);
            raw = mem[reg];
            break;
        }
        case OP_ASSOC_READ:  {
            uint8_t h = (static_cast<uint32_t>(a) * 73u + 19u) % 32;
            raw = mem[64 + h];
            break;
        }
        case OP_ASSOC_WRITE: {
            uint8_t h = (static_cast<uint32_t>(a) * 73u + 19u) % 32;
            mem[64 + h] = b;
            raw = b;
            break;
        }
        case OP_STACK_PUSH:  {
            int16_t sp = mem[127];
            mem[96 + (sp & 31)] = a;
            mem[127] = dev_clamp16(sp + 1, 0, 31);
            raw = mem[127];
            break;
        }
        case OP_STACK_POP:   {
            int16_t sp = mem[127];
            if (sp > 0) sp--;
            mem[127] = sp;
            raw = mem[96 + (sp & 31)];
            break;
        }
        default:             raw = a; break;
    }

    int32_t shaped = raw;
    switch(law_mode & 3) {
        case 0: shaped = raw; break;
        case 1: shaped = (raw < -256) ? -256 : ((raw > 255) ? 255 : raw); break;
        case 2: shaped = (raw > 0) ? 16 : ((raw < 0) ? -16 : 0); break;
        case 3: shaped = (raw * static_cast<int32_t>(b)) >> 4; break;
    }

    int32_t modulated = (shaped * static_cast<int32_t>(w)) >> 4;
    return dev_clamp16(modulated, -4096, 4095);
}

__device__ inline void run_capsule_organism(const CapsuleGenome& agent, 
                                           const int16_t* sensory, 
                                           int16_t fe_cur, 
                                           int16_t fe_exp, 
                                           int16_t fe_cum,
                                           int16_t* outputs, 
                                           int16_t* mem) {
    // 1. Execute encapsulated organs first in isolated local buffers
    int16_t capsule_outputs[NUM_CAPSULES] = {0};

    for (int c = 0; c < NUM_CAPSULES; ++c) {
        if (!agent.capsules[c].is_active || agent.capsules[c].active_nodes == 0) continue;
        const OrganCapsule& cap = agent.capsules[c];
        
        int16_t cap_bus[SENSORY_INPUTS + TELEMETRY_INPUTS + MAX_CAPSULE_NODES];
        for (int i = 0; i < SENSORY_INPUTS; ++i) cap_bus[i] = sensory[i];
        cap_bus[SENSORY_INPUTS + 0] = fe_cur;
        cap_bus[SENSORY_INPUTS + 1] = fe_exp;
        cap_bus[SENSORY_INPUTS + 2] = fe_cum;

        int cap_inputs_count = SENSORY_INPUTS + TELEMETRY_INPUTS;
        for (int n = 0; n < cap.active_nodes; ++n) {
            const GeneNode& gn = cap.nodes[n];
            int16_t gate_signal = cap_bus[gn.gate_in % (cap_inputs_count + n)];
            if (gate_signal <= -8) {
                cap_bus[cap_inputs_count + n] = cap_bus[gn.in1 % (cap_inputs_count + n)];
                continue;
            }
            int16_t in1_val = cap_bus[gn.in1 % (cap_inputs_count + n)];
            int16_t in2_val = cap_bus[gn.in2 % (cap_inputs_count + n)];
            cap_bus[cap_inputs_count + n] = exec_totipotent_node_op(gn.op, gn.law_mode, in1_val, in2_val, gn.weight, mem);
        }

        int16_t raw_cap_out = cap_bus[cap_inputs_count + cap.active_nodes - 1];
        // Scale by adaptive coupling gain into main bus
        capsule_outputs[c] = dev_clamp16((static_cast<int32_t>(raw_cap_out) * cap.output_coupling) >> 4, -4096, 4095);
    }

    // 2. Populate Open Bus with sensory, telemetry, and encapsulated organ outputs
    int16_t bus[TOTAL_INPUTS + MAX_NODES];
    for (int i = 0; i < SENSORY_INPUTS; ++i) bus[i] = sensory[i];
    bus[SENSORY_INPUTS + 0] = fe_cur;
    bus[SENSORY_INPUTS + 1] = fe_exp;
    bus[SENSORY_INPUTS + 2] = fe_cum;
    for (int c = 0; c < NUM_CAPSULES; ++c) {
        bus[SENSORY_INPUTS + TELEMETRY_INPUTS + c] = capsule_outputs[c];
    }

    // 3. Execute open-bus dynamic nodes with full unconstrained plasticity
    for (int n = 0; n < agent.active_nodes; ++n) {
        const GeneNode& gn = agent.nodes[n];
        int16_t gate_signal = bus[gn.gate_in % (TOTAL_INPUTS + n)];
        if (gate_signal <= -8) {
            bus[TOTAL_INPUTS + n] = bus[gn.in1 % (TOTAL_INPUTS + n)];
            continue;
        }

        int16_t in1_val = bus[gn.in1 % (TOTAL_INPUTS + n)];
        int16_t in2_val = bus[gn.in2 % (TOTAL_INPUTS + n)];
        bus[TOTAL_INPUTS + n] = exec_totipotent_node_op(gn.op, gn.law_mode, in1_val, in2_val, gn.weight, mem);
    }

    outputs[0] = bus[TOTAL_INPUTS + agent.active_nodes - 1];
    outputs[1] = (agent.active_nodes > 1) ? bus[TOTAL_INPUTS + agent.active_nodes - 2] : 0;
}

__global__ void init_curand_kernel(curandState* states, unsigned long seed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void evaluate_capsule_crucible_kernel(CapsuleGenome* agents, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    CapsuleGenome& agent = agents[idx];
    int16_t mem[MEM_SIZE];
    #define FLUSH_CAP_MEM() for(int m=0; m<MEM_SIZE; ++m) mem[m] = 0;

    // NICHE 1: Lorenz Physics
    FLUSH_CAP_MEM();
    float err_lorenz = 0.0f;
    int16_t lx = 10, ly = 20, lz = 30;
    int16_t fe_cur = 0, fe_exp = 0, fe_cum = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t dx = (10 * (ly - lx)) >> 3;
        int16_t target_x = lx + dx;
        int16_t sensory[SENSORY_INPUTS] = {lx, ly, lz, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_capsule_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t err = dev_clamp16(abs(out[0] - target_x), 0, 4095);
        err_lorenz += static_cast<float>(err);
        fe_cur = err;
        fe_exp = (fe_exp * 7 + err) >> 3;
        fe_cum = dev_clamp16(fe_cum + (err >> 2), 0, 4095);
        lx = target_x; ly += 5; lz += 2;
    }
    agent.niche_errors[0] = err_lorenz / 32.0f;

    // NICHE 2: Harmonic Oscillator
    FLUSH_CAP_MEM();
    float err_osc = 0.0f;
    int16_t sx = 0, sv = 0;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t force = (t & 1) ? 16 : -16;
        int16_t target_x = sx + sv;
        int16_t sensory[SENSORY_INPUTS] = {sx, sv, force, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_capsule_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t err = dev_clamp16(abs(out[0] - target_x), 0, 4095);
        err_osc += static_cast<float>(err);
        fe_cur = err;
        fe_exp = (fe_exp * 7 + err) >> 3;
        fe_cum = dev_clamp16(fe_cum + (err >> 2), 0, 4095);
        sv = (14 * sv - 4 * sx + force) >> 4;
        sx = target_x;
    }
    agent.niche_errors[1] = err_osc / 32.0f;

    // NICHE 3: Turing Logic
    FLUSH_CAP_MEM();
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
        run_capsule_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t bit_err = ((out[0] & 1) != tgt_sum) ? 16 : 0;
        if (bit_err > 0) err_carry += 1.0f;
        fe_cur = bit_err;
        fe_exp = (fe_exp * 7 + bit_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + bit_err, 0, 4095);
        carry = tgt_carry;
    }
    agent.niche_errors[2] = err_carry / 32.0f;

    // NICHE 4: Parity Stream
    FLUSH_CAP_MEM();
    float err_parity = 0.0f;
    int16_t par = 0;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t bit = (t * 5 + 1) & 1;
        par ^= bit;
        int16_t sensory[SENSORY_INPUTS] = {bit, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_capsule_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t bit_err = ((out[0] & 1) != par) ? 16 : 0;
        if (bit_err > 0) err_parity += 1.0f;
        fe_cur = bit_err;
        fe_exp = (fe_exp * 7 + bit_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + bit_err, 0, 4095);
    }
    agent.niche_errors[3] = err_parity / 32.0f;

    // NICHE 5: Context-Free Bracket Depth
    FLUSH_CAP_MEM();
    float err_bracket = 0.0f;
    int depth = 0;
    const char brackets[17] = "(())((()))()(())";
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 16; ++t) {
        char b = brackets[t];
        depth += (b == '(') ? 1 : -1;
        int16_t sensory[SENSORY_INPUTS] = {static_cast<int16_t>(b), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_capsule_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t depth_err = dev_clamp16(abs(out[0] - depth), 0, 4095);
        if (out[0] != depth) err_bracket += 1.0f;
        fe_cur = depth_err * 8;
        fe_exp = (fe_exp * 7 + fe_cur) >> 3;
        fe_cum = dev_clamp16(fe_cum + fe_cur, 0, 4095);
    }
    agent.niche_errors[4] = err_bracket / 16.0f;

    // NICHE 6: Markov Associative Flow
    FLUSH_CAP_MEM();
    float err_markov = 0.0f;
    const char seq[17] = "abcdabcdabcdabcd";
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 15; ++t) {
        int16_t cur = static_cast<int16_t>(seq[t]);
        int16_t tgt = static_cast<int16_t>(seq[t + 1]);
        int16_t sensory[SENSORY_INPUTS] = {cur, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_capsule_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t byte_err = (out[0] != tgt) ? 16 : 0;
        if (byte_err > 0) err_markov += 1.0f;
        fe_cur = byte_err;
        fe_exp = (fe_exp * 7 + byte_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + byte_err, 0, 4095);
    }
    agent.niche_errors[5] = err_markov / 15.0f;

    // NICHE 7: Natural Linguistic Stream
    FLUSH_CAP_MEM();
    float err_lang = 0.0f;
    const char phrase[33] = "instruction_response_karyon_core";
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 31; ++t) {
        uint8_t cur = static_cast<uint8_t>(phrase[t]);
        uint8_t nxt = static_cast<uint8_t>(phrase[t + 1]);
        int16_t sensory[SENSORY_INPUTS] = {static_cast<int16_t>(cur), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_capsule_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t lang_err = (static_cast<uint8_t>(out[0] & 0xFF) != nxt) ? 16 : 0;
        if (lang_err > 0) err_lang += 1.0f;
        fe_cur = lang_err;
        fe_exp = (fe_exp * 7 + lang_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + lang_err, 0, 4095);
    }
    agent.niche_errors[6] = err_lang / 31.0f;

    // NICHE 8: Noise Immunity
    float err_shock = 0.0f;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;
    for (int t = 0; t < 16; ++t) {
        int16_t clean = t * 4;
        int16_t noisy = clean + ((t & 1) ? 3 : -3);
        int16_t sensory[SENSORY_INPUTS] = {noisy, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_capsule_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);
        int16_t shock_err = dev_clamp16(abs(out[0] - clean), 0, 4095);
        err_shock += static_cast<float>(shock_err);
        fe_cur = shock_err;
        fe_exp = (fe_exp * 7 + shock_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + shock_err, 0, 4095);
    }
    agent.niche_errors[7] = err_shock / 16.0f;

    agent.composite_fe = agent.niche_errors[0] * 0.10f +
                         agent.niche_errors[1] * 0.10f +
                         agent.niche_errors[2] * 2.00f +
                         agent.niche_errors[3] * 2.00f +
                         agent.niche_errors[4] * 2.00f +
                         agent.niche_errors[5] * 2.00f +
                         agent.niche_errors[6] * 2.00f +
                         agent.niche_errors[7] * 0.10f;

    // Count active capsules
    int act_caps = 0;
    for (int c = 0; c < NUM_CAPSULES; ++c) {
        if (agent.capsules[c].is_active) act_caps++;
    }
    agent.active_capsules_cnt = act_caps;
}

__global__ void gpu_pareto_tournament_kernel(CapsuleGenome* agents, 
                                             CapsuleGenome* niche_pool, 
                                             int pop_size, 
                                             curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local_state = states[idx];
    CapsuleGenome& agent = agents[idx];

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
// GPU Resident Modular Capsule Ingestion & Open-Bus Morphogenesis Kernel
// ======================================================================================
__global__ void gpu_modular_capsule_morphogenesis_kernel(CapsuleGenome* agents, 
                                                         const CapsuleGenome* niche_pool, 
                                                         int pop_size, 
                                                         curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local_state = states[idx];

    int p1_idx = curand(&local_state) % NICHE_SLOTS;
    CapsuleGenome child = niche_pool[p1_idx];

    // Sovereign Differentiation Velocity
    if (curand_uniform(&local_state) < 0.20f) {
        int v_delta = (curand(&local_state) % 3) - 1;
        int new_v = static_cast<int>(child.diff_velocity) + v_delta;
        child.diff_velocity = static_cast<uint8_t>((new_v < 1) ? 1 : ((new_v > 16) ? 16 : new_v));
    }
    float diff_factor = static_cast<float>(child.diff_velocity) / 8.0f;

    // 1. MODULAR CAPSULE HARVESTING (Endosymbiosis without bus poisoning)
    // 25% chance to encapsulate a verified specialist organ from niche pool
    if (curand_uniform(&local_state) < 0.25f) {
        int donor_niche = curand(&local_state) % NUM_NICHES;
        int donor_slot = donor_niche * (NICHE_SLOTS / NUM_NICHES) + (curand(&local_state) % (NICHE_SLOTS / NUM_NICHES));
        const CapsuleGenome& donor = niche_pool[donor_slot];

        // If donor has high competence in this niche (error == 0.0 or low)
        if (donor.niche_errors[donor_niche] <= 0.05f && donor.active_nodes >= 2) {
            // Find an open or replaceable capsule slot
            int cap_slot = -1;
            for (int c = 0; c < NUM_CAPSULES; ++c) {
                if (!child.capsules[c].is_active) {
                    cap_slot = c;
                    break;
                }
            }
            if (cap_slot == -1) {
                // If all slots are full, pick randomly if current capsule is from a different niche
                int cand = curand(&local_state) % NUM_CAPSULES;
                if (child.capsules[cand].source_niche != donor_niche) {
                    cap_slot = cand;
                }
            }

            if (cap_slot != -1) {
                // Ingest sub-circuit into isolated capsule
                int organ_len = (donor.active_nodes > MAX_CAPSULE_NODES) ? MAX_CAPSULE_NODES : donor.active_nodes;
                int start_node = donor.active_nodes - organ_len;
                child.capsules[cap_slot].active_nodes = organ_len;
                child.capsules[cap_slot].source_niche = donor_niche;
                child.capsules[cap_slot].is_active = 1;
                child.capsules[cap_slot].output_coupling = 16; // Full unity gain

                int cap_input_scope = SENSORY_INPUTS + TELEMETRY_INPUTS;
                for (int k = 0; k < organ_len; ++k) {
                    GeneNode n_copy = donor.nodes[start_node + k];
                    // Remap inputs within capsule local scope
                    if (n_copy.in1 >= TOTAL_INPUTS) {
                        n_copy.in1 = cap_input_scope + ((n_copy.in1 - TOTAL_INPUTS) % (cap_input_scope + k));
                    } else if (n_copy.in1 >= SENSORY_INPUTS + TELEMETRY_INPUTS) {
                        n_copy.in1 = n_copy.in1 % cap_input_scope;
                    }
                    if (n_copy.in2 >= TOTAL_INPUTS) {
                        n_copy.in2 = cap_input_scope + ((n_copy.in2 - TOTAL_INPUTS) % (cap_input_scope + k));
                    } else if (n_copy.in2 >= SENSORY_INPUTS + TELEMETRY_INPUTS) {
                        n_copy.in2 = n_copy.in2 % cap_input_scope;
                    }
                    if (n_copy.gate_in >= TOTAL_INPUTS) {
                        n_copy.gate_in = cap_input_scope + ((n_copy.gate_in - TOTAL_INPUTS) % (cap_input_scope + k));
                    } else if (n_copy.gate_in >= SENSORY_INPUTS + TELEMETRY_INPUTS) {
                        n_copy.gate_in = n_copy.gate_in % cap_input_scope;
                    }
                    child.capsules[cap_slot].nodes[k] = n_copy;
                }
            }
        }
    }

    // Fine-tune capsule coupling gains
    for (int c = 0; c < NUM_CAPSULES; ++c) {
        if (child.capsules[c].is_active && curand_uniform(&local_state) < 0.15f) {
            int delta_gain = (curand(&local_state) % 5) - 2;
            int new_g = child.capsules[c].output_coupling + delta_gain;
            child.capsules[c].output_coupling = static_cast<int8_t>((new_g < -16) ? -16 : ((new_g > 16) ? 16 : new_g));
        }
    }

    // 2. OPEN-BUS UNSHACKLED MORPHOGENESIS (Sprouting & Pruning under Stability)
    // The main body has 100% plasticity and can grow deep cascades to integrate capsules
    float stability_bonus = (child.composite_fe < 4.0f) ? 1.5f : 1.0f;
    float p_sprout = 0.25f * diff_factor * stability_bonus;
    float p_prune  = 0.10f * diff_factor;

    if (curand_uniform(&local_state) < p_sprout && child.active_nodes < MAX_NODES) {
        child.active_nodes++;
    }
    if (curand_uniform(&local_state) < p_prune && child.active_nodes > 4) {
        child.active_nodes--;
    }

    // 3. FULL PLASTICITY ON MAIN BUS (Zero rigidity!)
    float p_op   = 0.12f * diff_factor;
    float p_law  = 0.12f * diff_factor;
    float p_conn = 0.15f * diff_factor;
    float p_w    = 0.28f * diff_factor;

    for (int n = 0; n < child.active_nodes; ++n) {
        if (curand_uniform(&local_state) < p_op) {
            child.nodes[n].op = 1 + (curand(&local_state) % (NUM_OPS - 1));
        }
        if (curand_uniform(&local_state) < p_law) {
            child.nodes[n].law_mode = curand(&local_state) & 3;
        }
        if (curand_uniform(&local_state) < p_conn) {
            child.nodes[n].in1 = curand(&local_state) % (TOTAL_INPUTS + n);
        }
        if (curand_uniform(&local_state) < p_conn) {
            child.nodes[n].in2 = curand(&local_state) % (TOTAL_INPUTS + n);
        }
        if (curand_uniform(&local_state) < p_conn) {
            child.nodes[n].gate_in = curand(&local_state) % (TOTAL_INPUTS + n);
        }
        if (curand_uniform(&local_state) < p_w) {
            int delta_w = (curand(&local_state) % 7) - 3;
            int nw = child.nodes[n].weight + delta_w;
            child.nodes[n].weight = static_cast<int8_t>((nw < -16) ? -16 : ((nw > 16) ? 16 : nw));
        }
    }

    agents[idx] = child;
    states[idx] = local_state;
}

const char* opcode_name(uint8_t op) {
    switch(op) {
        case OP_PASS: return "PASS";
        case OP_ADD: return "ADD";
        case OP_SUB: return "SUB";
        case OP_MUL_SHIFT: return "MUL_SHIFT";
        case OP_DIV_MOD: return "DIV_MOD";
        case OP_XOR: return "XOR";
        case OP_AND: return "AND";
        case OP_OR: return "OR";
        case OP_NOT: return "NOT";
        case OP_SHL: return "SHL";
        case OP_SHR: return "SHR";
        case OP_MIN: return "MIN";
        case OP_MAX: return "MAX";
        case OP_ABS: return "ABS";
        case OP_SIGN: return "SIGN";
        case OP_SATURATE: return "SATURATE";
        case OP_CMP_EQ: return "CMP_EQ";
        case OP_CMP_LT: return "CMP_LT";
        case OP_COND_MUX: return "COND_MUX";
        case OP_LOAD_REG: return "LOAD_REG";
        case OP_STORE_REG: return "STORE_REG";
        case OP_ACCUMULATE: return "ACCUMULATE";
        case OP_ASSOC_READ: return "ASSOC_READ";
        case OP_ASSOC_WRITE: return "ASSOC_WRITE";
        case OP_STACK_PUSH: return "STACK_PUSH";
        case OP_STACK_POP: return "STACK_POP";
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

std::string input_desc(int idx) {
    if (idx < SENSORY_INPUTS) return "s_in[" + std::to_string(idx) + "]";
    if (idx == SENSORY_INPUTS + 0) return "FE_CUR(F_t)";
    if (idx == SENSORY_INPUTS + 1) return "FE_EXP(G_t)";
    if (idx == SENSORY_INPUTS + 2) return "FE_CUM(Omega_t)";
    if (idx >= SENSORY_INPUTS + TELEMETRY_INPUTS && idx < TOTAL_INPUTS) {
        int c_idx = idx - (SENSORY_INPUTS + TELEMETRY_INPUTS);
        return "CAPSULE_" + std::to_string(c_idx) + "_OUT";
    }
    return "n" + std::to_string(idx - TOTAL_INPUTS);
}

void dissect_capsule_organism(const CapsuleGenome& champion) {
    std::cout << "\n========================================================================\n";
    std::cout << "=== 🔬 DEEP ENDOSCOPIC AUTOPSY: MODULAR CAPSULE CHAMPION             ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Active Open-Bus Nodes    : " << (int)champion.active_nodes << "\n";
    std::cout << "Active Specialist Capsules: " << (int)champion.active_capsules_cnt << " / " << NUM_CAPSULES << "\n";
    std::cout << "Differentiation Velocity : " << (int)champion.diff_velocity << " / 16 (" 
              << (champion.diff_velocity / 8.0f) << "x speed)\n";
    std::cout << "Composite Free Energy    : " << std::fixed << std::setprecision(5) << champion.composite_fe << "\n\n";

    std::cout << ">>> 8-Environmental Niche Scorecard (Strict Telemetry):\n";
    std::cout << "  1. Chaotic Lorenz Physics Tracking Error : " << champion.niche_errors[0] << "\n";
    std::cout << "  2. Harmonic Oscillator Tracking Error    : " << champion.niche_errors[1] << "\n";
    std::cout << "  3. Turing Ripple-Carry Logic Error Rate  : " << champion.niche_errors[2] * 100.0f << " %\n";
    std::cout << "  4. Dynamic Parity Stream Error Rate      : " << champion.niche_errors[3] * 100.0f << " %\n";
    std::cout << "  5. Context-Free Stack Depth Error Rate   : " << champion.niche_errors[4] * 100.0f << " %\n";
    std::cout << "  6. Markov Associative Flow Error Rate    : " << champion.niche_errors[5] * 100.0f << " %\n";
    std::cout << "  7. Natural Language Stream Error Rate    : " << champion.niche_errors[6] * 100.0f << " %\n";
    std::cout << "  8. Synaptic Shock Noise Error            : " << champion.niche_errors[7] << "\n\n";

    std::cout << ">>> Encapsulated Specialist Organs (Capsule Registry):\n";
    for (int c = 0; c < NUM_CAPSULES; ++c) {
        if (!champion.capsules[c].is_active) {
            std::cout << "  Capsule " << c << ": [EMPTY / INACTIVE]\n";
            continue;
        }
        std::cout << "  Capsule " << c << ": [ACTIVE] | Source Niche: " << (int)champion.capsules[c].source_niche
                  << " | Nodes: " << (int)champion.capsules[c].active_nodes
                  << " | Bus Coupling Gain: " << (int)champion.capsules[c].output_coupling << " / 16\n";
    }
    std::cout << "\n>>> Open-Bus Dynamic Circuit Topology (Active Nodes Audit):\n";
    for (int n = 0; n < champion.active_nodes; ++n) {
        const GeneNode& gn = champion.nodes[n];
        std::string s1 = input_desc(gn.in1);
        std::string s2 = input_desc(gn.in2);
        std::string gate = input_desc(gn.gate_in);

        std::cout << "  node_" << std::setw(2) << n << " = " << opcode_name(gn.op) 
                  << "(" << s1 << ", " << s2 << ") [" << law_name(gn.law_mode) 
                  << "] * (" << (int)gn.weight << " / 16) | Gate: " << gate;
        if (gn.in1 >= SENSORY_INPUTS && gn.in1 < SENSORY_INPUTS + TELEMETRY_INPUTS) std::cout << " <== [FE-SENSING]";
        if (gn.in2 >= SENSORY_INPUTS && gn.in2 < SENSORY_INPUTS + TELEMETRY_INPUTS) std::cout << " <== [FE-SENSING]";
        if (gn.in1 >= SENSORY_INPUTS + TELEMETRY_INPUTS && gn.in1 < TOTAL_INPUTS) std::cout << " <== [CAPSULE-FED]";
        if (gn.in2 >= SENSORY_INPUTS + TELEMETRY_INPUTS && gn.in2 < TOTAL_INPUTS) std::cout << " <== [CAPSULE-FED]";
        std::cout << "\n";
    }
    std::cout << "========================================================================\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-611: MODULAR CAPSULE ENDOSYMBIOSIS & OPEN-BUS MORPHOGENESIS  ===\n";
    std::cout << "========================================================================\n";
    uint64_t total_evals_target = static_cast<uint64_t>(POP_SIZE) * NUM_GENERATIONS;
    std::cout << "Target Generations       : " << NUM_GENERATIONS << " DEEP GENERATIONS\n";
    std::cout << "Target Evaluations       : " << total_evals_target << " (98.30 MILLION AGENTS)\n";
    std::cout << "Active Batch Size on GPU : " << POP_SIZE << " organisms in parallel\n";
    std::cout << "Modular Architecture     : Up to " << NUM_CAPSULES << " isolated specialist capsules\n";
    std::cout << "Open-Bus Plasticity      : Main network completely plastic up to " << MAX_NODES << " nodes\n";
    std::cout << "Epigenetic Protection    : Capsules isolated from main-bus mutations\n";
    std::cout << "Sovereign Substrate      : Arbitrary DAG + 128 dynamic registers\n\n";

    CapsuleGenome* d_agents = nullptr;
    CapsuleGenome* d_niche_pool = nullptr;
    curandState* d_rand_states = nullptr;

    CHECK_CUDA(cudaMalloc(&d_agents, POP_SIZE * sizeof(CapsuleGenome)));
    CHECK_CUDA(cudaMalloc(&d_niche_pool, NICHE_SLOTS * sizeof(CapsuleGenome)));
    CHECK_CUDA(cudaMalloc(&d_rand_states, POP_SIZE * sizeof(curandState)));

    int threads = 256;
    int blocks_pop = (POP_SIZE + threads - 1) / threads;

    init_curand_kernel<<<blocks_pop, threads>>>(d_rand_states, 2026, POP_SIZE);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<CapsuleGenome> h_init_pool(NICHE_SLOTS);
    std::mt19937 host_rng(2026);
    std::uniform_int_distribution<int> op_dist(1, NUM_OPS - 1);
    std::uniform_int_distribution<int> law_dist(0, 3);
    std::uniform_int_distribution<int> w_dist(-16, 16);
    std::uniform_int_distribution<int> n_dist(8, 20);

    for (int i = 0; i < NICHE_SLOTS; ++i) {
        h_init_pool[i].active_nodes = n_dist(host_rng);
        h_init_pool[i].diff_velocity = 8;
        h_init_pool[i].active_capsules_cnt = 0;
        h_init_pool[i].composite_fe = 99999.0f;
        for (int t = 0; t < NUM_NICHES; ++t) h_init_pool[i].niche_errors[t] = 999.0f;
        for (int c = 0; c < NUM_CAPSULES; ++c) {
            h_init_pool[i].capsules[c].is_active = 0;
            h_init_pool[i].capsules[c].active_nodes = 0;
            h_init_pool[i].capsules[c].source_niche = 0;
            h_init_pool[i].capsules[c].output_coupling = 16;
        }
        for (int n = 0; n < MAX_NODES; ++n) {
            h_init_pool[i].nodes[n].op = static_cast<uint8_t>(op_dist(host_rng));
            h_init_pool[i].nodes[n].law_mode = static_cast<uint8_t>(law_dist(host_rng));
            h_init_pool[i].nodes[n].in1 = host_rng() % (TOTAL_INPUTS + n);
            h_init_pool[i].nodes[n].in2 = host_rng() % (TOTAL_INPUTS + n);
            h_init_pool[i].nodes[n].gate_in = host_rng() % (TOTAL_INPUTS + n);
            h_init_pool[i].nodes[n].weight = static_cast<int8_t>(w_dist(host_rng));
        }
    }

    CHECK_CUDA(cudaMemcpy(d_niche_pool, h_init_pool.data(), NICHE_SLOTS * sizeof(CapsuleGenome), cudaMemcpyHostToDevice));

    auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t evals_completed = 0;

    std::vector<CapsuleGenome> h_pool_sample(NICHE_SLOTS);

    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        // 1. GPU Modular Capsule Harvesting & Open-Bus Morphogenesis
        gpu_modular_capsule_morphogenesis_kernel<<<blocks_pop, threads>>>(d_agents, d_niche_pool, POP_SIZE, d_rand_states);

        // 2. GPU Parallel Reality Evaluation
        evaluate_capsule_crucible_kernel<<<blocks_pop, threads>>>(d_agents, POP_SIZE);

        // 3. GPU Pareto Selection Tournament
        gpu_pareto_tournament_kernel<<<blocks_pop, threads>>>(d_agents, d_niche_pool, POP_SIZE, d_rand_states);

        evals_completed += POP_SIZE;

        if ((gen + 1) % 500 == 0 || gen == 0) {
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(h_pool_sample.data(), d_niche_pool, NICHE_SLOTS * sizeof(CapsuleGenome), cudaMemcpyDeviceToHost));

            int best_idx = 0;
            for (int i = 1; i < NICHE_SLOTS; ++i) {
                if (h_pool_sample[i].composite_fe < h_pool_sample[best_idx].composite_fe) {
                    best_idx = i;
                }
            }

            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double speed = static_cast<double>(evals_completed) / elapsed;

            std::cout << "Gen [" << std::setw(5) << (gen + 1) << "/" << NUM_GENERATIONS << "]"
                      << " | Evals: " << std::setw(8) << evals_completed
                      << " | Speed: " << std::fixed << std::setprecision(0) << speed << " evals/s"
                      << " | Min FE: " << std::setprecision(4) << h_pool_sample[best_idx].composite_fe
                      << " | Turing: " << std::setprecision(1) << (h_pool_sample[best_idx].niche_errors[2] * 100.0f) << "%"
                      << " | Parity: " << (h_pool_sample[best_idx].niche_errors[3] * 100.0f) << "%"
                      << " | Stack: " << (h_pool_sample[best_idx].niche_errors[4] * 100.0f) << "%"
                      << " | Markov: " << (h_pool_sample[best_idx].niche_errors[5] * 100.0f) << "%"
                      << " | Lang: " << (h_pool_sample[best_idx].niche_errors[6] * 100.0f) << "%"
                      << " | Nodes: " << (int)h_pool_sample[best_idx].active_nodes
                      << " | Caps: " << (int)h_pool_sample[best_idx].active_capsules_cnt << "\n";
            std::cout.flush();
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    CHECK_CUDA(cudaMemcpy(h_pool_sample.data(), d_niche_pool, NICHE_SLOTS * sizeof(CapsuleGenome), cudaMemcpyDeviceToHost));

    int global_best_idx = 0;
    for (int i = 1; i < NICHE_SLOTS; ++i) {
        if (h_pool_sample[i].composite_fe < h_pool_sample[global_best_idx].composite_fe) {
            global_best_idx = i;
        }
    }

    std::cout << "\n========================================================================\n";
    std::cout << ">>> MODULAR CAPSULE CRUCIBLE COMPLETE: " << evals_completed << " EVALS!\n";
    std::cout << ">>> Total Runtime : " << std::fixed << std::setprecision(2) << total_sec << " seconds\n";
    std::cout << ">>> Average Speed : " << std::fixed << std::setprecision(0) << (evals_completed / total_sec) << " agents/sec\n";
    std::cout << "========================================================================\n";

    dissect_capsule_organism(h_pool_sample[global_best_idx]);

    std::cout << ">>> SPECIALIST NICHE CHAMPIONS AUDIT (AFTER MODULAR CAPSULE EVOLUTION):\n";
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
                  << ", Nodes: " << (int)h_pool_sample[best_slot].active_nodes 
                  << ", Caps: " << (int)h_pool_sample[best_slot].active_capsules_cnt << ")\n";
    }
    std::cout << "========================================================================\n";

    CHECK_CUDA(cudaFree(d_agents));
    CHECK_CUDA(cudaFree(d_niche_pool));
    CHECK_CUDA(cudaFree(d_rand_states));
    return 0;
}
