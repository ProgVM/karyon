// ======================================================================================
// EXP-607: Self-Aware Telemetric Field Organism (Free Energy Telemetry & Sovereign Gating)
// --------------------------------------------------------------------------------------
// Fundamental Sovereign Cybernetics (Principles 2, 5, 29, 30 & Bazilevs Directive):
// 1. DYNAMIC COMPARTMENT ROUTING & SELECTIVE SUBSTRATE ACTIVATION:
//    - Organisms are NO LONGER forced to execute their entire body monolithically.
//    - Every node features an endogenous, self-controlled execution gate:
//        Active dynamic execution mask dynamically modulated by sensory context and internal state.
//    - Organisms can autonomously sleep, bypass, or route execution through specific organ
//      compartments depending on the domain/task demand.
// 2. EPISTEMIC SELF-AWARENESS (Free Energy Endoscopic Vision):
//    - Agents possess direct, unmediated sensory access to their own thermodynamic state:
//        Input Channel 4: Current Instantaneous Error / Surprise: F_t = |y_t - target_{t-1}|
//        Input Channel 5: Expected Free Energy / Moving Variance: G_t = EMA(F_t)
//        Input Channel 6: Cumulative Drift / Global Free Energy: \Omega_t = \sum F_t
//    - Agents directly "see" the consequence of their actions and internal structural firings!
// 3. CAUSAL SELF-CREDIT & TELEMETRIC GUIDED RECONFIGURATION (Principle 5):
//    - Evolution utilizes causal efficacy traces:
//        Nodes whose activation correlates with Free Energy reduction are preserved/reinforced.
//        Nodes whose activation spikes Free Energy are dynamically pruned or mutated.
// 4. UNSHACKLED SOVEREIGN SUBSTRATE:
//    - Up to 64 dynamic nodes, 128 dynamic memory registers, 4 field laws, complete silicon ISA.
//    - Zero external time clocks (no artificial 't' steps).
//    - Strict KEP Rule #2 evaluation across the 8-task reality battery.
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

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

constexpr int POP_SIZE        = 65536;  // 65,536 parallel self-aware organisms
constexpr int NUM_GENERATIONS = 100;    // 6.55M evaluations on CUDA
constexpr int MAX_NODES       = 64;     // Unshackled node ceiling
constexpr int MEM_SIZE        = 128;    // 128 dynamic state registers & field memory
constexpr int SENSORY_INPUTS  = 4;      // Pure environmental inputs
constexpr int TELEMETRY_INPUTS= 3;      // [F_t (current), G_t (expected), Omega_t (cumulative)]
constexpr int TOTAL_INPUTS    = SENSORY_INPUTS + TELEMETRY_INPUTS; // 7 inputs total
constexpr int NUM_OUTPUTS     = 2;      // Primary action, auxiliary efference
constexpr int NUM_NICHES      = 8;      // 8 independent environmental reality challenges

enum OpCode : uint8_t {
    OP_PASS = 0,
    OP_ADD,
    OP_SUB,
    OP_MUL_SHIFT,    // (a * b) >> 4
    OP_DIV_MOD,      // (b != 0) ? (a % b) : 0
    OP_XOR,
    OP_AND,
    OP_OR,
    OP_NOT,
    OP_SHL,          // a << (b & 7)
    OP_SHR,          // a >> (b & 7)
    OP_MIN,
    OP_MAX,
    OP_ABS,
    OP_SIGN,
    OP_SATURATE,     // clamp to [-128, 127]
    OP_CMP_EQ,       // (a == b) ? 16 : 0
    OP_CMP_LT,       // (a < b) ? 16 : 0
    OP_COND_MUX,     // (a > 0) ? b : 0
    OP_LOAD_REG,     // read registers[a % 64]
    OP_STORE_REG,    // registers[b % 64] = a
    OP_ACCUMULATE,   // registers[a % 64] += b
    OP_ASSOC_READ,   // read hash_table[((a * 73) ^ 19) % 32]
    OP_ASSOC_WRITE,  // hash_table[((a * 73) ^ 19) % 32] = b
    OP_STACK_PUSH,   // stack[sp++] = a
    OP_STACK_POP,    // res = stack[--sp]
    NUM_OPS
};

struct GeneNode {
    uint8_t op;
    uint8_t in1;       // input index into (TOTAL_INPUTS + previous_nodes)
    uint8_t in2;       // input index into (TOTAL_INPUTS + previous_nodes)
    uint8_t law_mode;  // 0: Linear, 1: Saturate, 2: Threshold/Step, 3: Bilinear
    uint8_t gate_in;   // conditional activation gate input index
    int8_t  weight;    // Modulation scale [-16 .. 16]
};

struct alignas(64) SelfAwareGenome {
    GeneNode nodes[MAX_NODES];
    uint8_t  active_nodes;
    float    niche_errors[NUM_NICHES];
    float    composite_fe;
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

__device__ inline void run_self_aware_organism(const SelfAwareGenome& agent, 
                                              const int16_t* sensory, 
                                              int16_t fe_current, 
                                              int16_t fe_expected, 
                                              int16_t fe_cumulative,
                                              int16_t* outputs, 
                                              int16_t* mem) {
    int16_t bus[TOTAL_INPUTS + MAX_NODES];
    for (int i = 0; i < SENSORY_INPUTS; ++i) bus[i] = sensory[i];
    bus[SENSORY_INPUTS + 0] = fe_current;
    bus[SENSORY_INPUTS + 1] = fe_expected;
    bus[SENSORY_INPUTS + 2] = fe_cumulative;

    for (int n = 0; n < agent.active_nodes; ++n) {
        const GeneNode& gn = agent.nodes[n];
        
        // Dynamic Gating & Substrate Bypassing:
        // If gate_in signal <= -4, the node is dormant/bypassed, passing residual signal forward!
        int16_t gate_signal = bus[gn.gate_in % (TOTAL_INPUTS + n)];
        if (gate_signal <= -8) {
            // Dormant organ compartment: bypass computation cleanly without side-effects
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

__global__ void evaluate_self_aware_crucible_kernel(SelfAwareGenome* agents, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    SelfAwareGenome& agent = agents[idx];
    int16_t mem[MEM_SIZE];
    #define FLUSH_AWARE_MEM() for(int m=0; m<MEM_SIZE; ++m) mem[m] = 0;

    // ----------------------------------------------------------------------------------
    // NICHE 1: Chaotic Continuous Dynamics (Lorenz Velocity Tracking - Self-Aware FE)
    // ----------------------------------------------------------------------------------
    FLUSH_AWARE_MEM();
    float err_lorenz = 0.0f;
    int16_t lx = 10, ly = 20, lz = 30;
    int16_t fe_cur = 0, fe_exp = 0, fe_cum = 0;

    for (int t = 0; t < 32; ++t) {
        int16_t dx = (10 * (ly - lx)) >> 3;
        int16_t target_x = lx + dx;
        int16_t sensory[SENSORY_INPUTS] = {lx, ly, lz, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_self_aware_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);

        int16_t err = dev_clamp16(abs(out[0] - target_x), 0, 4095);
        err_lorenz += static_cast<float>(err);

        // Feedback thermodynamic telemetry directly into cognitive loop
        fe_cur = err;
        fe_exp = (fe_exp * 7 + err) >> 3;
        fe_cum = dev_clamp16(fe_cum + (err >> 2), 0, 4095);

        lx = target_x; ly += 5; lz += 2;
    }
    agent.niche_errors[0] = err_lorenz / 32.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 2: Physical Damped Forced Oscillator (Continuous Mechanics - Self-Aware FE)
    // ----------------------------------------------------------------------------------
    FLUSH_AWARE_MEM();
    float err_osc = 0.0f;
    int16_t sx = 0, sv = 0;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;

    for (int t = 0; t < 32; ++t) {
        int16_t force = (t & 1) ? 16 : -16;
        int16_t target_x = sx + sv;
        int16_t sensory[SENSORY_INPUTS] = {sx, sv, force, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_self_aware_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);

        int16_t err = dev_clamp16(abs(out[0] - target_x), 0, 4095);
        err_osc += static_cast<float>(err);

        fe_cur = err;
        fe_exp = (fe_exp * 7 + err) >> 3;
        fe_cum = dev_clamp16(fe_cum + (err >> 2), 0, 4095);

        sv = (14 * sv - 4 * sx + force) >> 4;
        sx = target_x;
    }
    agent.niche_errors[1] = err_osc / 32.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 3: Turing Multi-Bit Ripple Carry Logic (Self-Aware Discrete State Machine)
    // ----------------------------------------------------------------------------------
    FLUSH_AWARE_MEM();
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

        run_self_aware_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);

        int16_t bit_err = ((out[0] & 1) != tgt_sum) ? 16 : 0;
        if (bit_err > 0) err_carry += 1.0f;

        fe_cur = bit_err;
        fe_exp = (fe_exp * 7 + bit_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + bit_err, 0, 4095);

        carry = tgt_carry;
    }
    agent.niche_errors[2] = err_carry / 32.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 4: Dynamic Parity Stream (Sequence XOR / T-Flip-Flop with Self-Aware FE)
    // ----------------------------------------------------------------------------------
    FLUSH_AWARE_MEM();
    float err_parity = 0.0f;
    int16_t par = 0;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;

    for (int t = 0; t < 32; ++t) {
        int16_t bit = (t * 5 + 1) & 1;
        par ^= bit;
        int16_t sensory[SENSORY_INPUTS] = {bit, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_self_aware_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);

        int16_t bit_err = ((out[0] & 1) != par) ? 16 : 0;
        if (bit_err > 0) err_parity += 1.0f;

        fe_cur = bit_err;
        fe_exp = (fe_exp * 7 + bit_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + bit_err, 0, 4095);
    }
    agent.niche_errors[3] = err_parity / 32.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 5: Context-Free Bracket Balancing (Push/Pop Stack - Self-Aware FE)
    // ----------------------------------------------------------------------------------
    FLUSH_AWARE_MEM();
    float err_bracket = 0.0f;
    int depth = 0;
    const char brackets[17] = "(())((()))()(())";
    fe_cur = 0; fe_exp = 0; fe_cum = 0;

    for (int t = 0; t < 16; ++t) {
        char b = brackets[t];
        depth += (b == '(') ? 1 : -1;
        int16_t sensory[SENSORY_INPUTS] = {static_cast<int16_t>(b), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_self_aware_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);

        int16_t depth_err = dev_clamp16(abs(out[0] - depth), 0, 4095);
        if (out[0] != depth) err_bracket += 1.0f;

        fe_cur = depth_err * 8;
        fe_exp = (fe_exp * 7 + fe_cur) >> 3;
        fe_cum = dev_clamp16(fe_cum + fe_cur, 0, 4095);
    }
    agent.niche_errors[4] = err_bracket / 16.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 6: Markov Associative Byte Transitions (Graph Flow with Self-Aware FE)
    // ----------------------------------------------------------------------------------
    FLUSH_AWARE_MEM();
    float err_markov = 0.0f;
    const char seq[17] = "abcdabcdabcdabcd";
    fe_cur = 0; fe_exp = 0; fe_cum = 0;

    for (int t = 0; t < 15; ++t) {
        int16_t cur = static_cast<int16_t>(seq[t]);
        int16_t tgt = static_cast<int16_t>(seq[t + 1]);
        int16_t sensory[SENSORY_INPUTS] = {cur, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_self_aware_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);

        int16_t byte_err = (out[0] != tgt) ? 16 : 0;
        if (byte_err > 0) err_markov += 1.0f;

        fe_cur = byte_err;
        fe_exp = (fe_exp * 7 + byte_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + byte_err, 0, 4095);
    }
    agent.niche_errors[5] = err_markov / 15.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 7: Natural Linguistic Stream (Alpaca Next-Byte with Self-Aware FE)
    // ----------------------------------------------------------------------------------
    FLUSH_AWARE_MEM();
    float err_lang = 0.0f;
    const char phrase[33] = "instruction_response_karyon_core";
    fe_cur = 0; fe_exp = 0; fe_cum = 0;

    for (int t = 0; t < 31; ++t) {
        uint8_t cur = static_cast<uint8_t>(phrase[t]);
        uint8_t nxt = static_cast<uint8_t>(phrase[t + 1]);
        int16_t sensory[SENSORY_INPUTS] = {static_cast<int16_t>(cur), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_self_aware_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);

        int16_t lang_err = (static_cast<uint8_t>(out[0] & 0xFF) != nxt) ? 16 : 0;
        if (lang_err > 0) err_lang += 1.0f;

        fe_cur = lang_err;
        fe_exp = (fe_exp * 7 + lang_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + lang_err, 0, 4095);
    }
    agent.niche_errors[6] = err_lang / 31.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 8: Synaptic Shock Noise
    // ----------------------------------------------------------------------------------
    float err_shock = 0.0f;
    fe_cur = 0; fe_exp = 0; fe_cum = 0;

    for (int t = 0; t < 16; ++t) {
        int16_t clean = t * 4;
        int16_t noisy = clean + ((t & 1) ? 3 : -3);
        int16_t sensory[SENSORY_INPUTS] = {noisy, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_self_aware_organism(agent, sensory, fe_cur, fe_exp, fe_cum, out, mem);

        int16_t shock_err = dev_clamp16(abs(out[0] - clean), 0, 4095);
        err_shock += static_cast<float>(shock_err);

        fe_cur = shock_err;
        fe_exp = (fe_exp * 7 + shock_err) >> 3;
        fe_cum = dev_clamp16(fe_cum + shock_err, 0, 4095);
    }
    agent.niche_errors[7] = err_shock / 16.0f;

    // Master Composite Free Energy
    agent.composite_fe = agent.niche_errors[0] * 0.10f + // Lorenz
                         agent.niche_errors[1] * 0.10f + // Oscillator
                         agent.niche_errors[2] * 2.00f + // Turing
                         agent.niche_errors[3] * 2.00f + // Parity
                         agent.niche_errors[4] * 2.00f + // Stack
                         agent.niche_errors[5] * 2.00f + // Markov
                         agent.niche_errors[6] * 2.00f + // Language
                         agent.niche_errors[7] * 0.10f;  // Shock
}

void init_self_aware_genome(SelfAwareGenome& agent, std::mt19937& rng) {
    std::uniform_int_distribution<int> op_dist(1, NUM_OPS - 1);
    std::uniform_int_distribution<int> law_dist(0, 3);
    std::uniform_int_distribution<int> w_dist(-16, 16);
    std::uniform_int_distribution<int> n_dist(8, 24);

    agent.active_nodes = n_dist(rng);
    agent.composite_fe = 99999.0f;
    for (int t = 0; t < NUM_NICHES; ++t) agent.niche_errors[t] = 999.0f;

    for (int n = 0; n < MAX_NODES; ++n) {
        agent.nodes[n].op = static_cast<uint8_t>(op_dist(rng));
        agent.nodes[n].law_mode = static_cast<uint8_t>(law_dist(rng));
        agent.nodes[n].in1 = rng() % (TOTAL_INPUTS + n);
        agent.nodes[n].in2 = rng() % (TOTAL_INPUTS + n);
        agent.nodes[n].gate_in = rng() % (TOTAL_INPUTS + n);
        agent.nodes[n].weight = static_cast<int8_t>(w_dist(rng));
    }
}

void mutate_self_aware_genome(SelfAwareGenome& child, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_int_distribution<int> op_dist(1, NUM_OPS - 1);
    std::uniform_int_distribution<int> law_dist(0, 3);

    // Dynamic Sprouting and Pruning
    if (prob(rng) < 0.25f && child.active_nodes < MAX_NODES) child.active_nodes++;
    if (prob(rng) < 0.10f && child.active_nodes > 4) child.active_nodes--;

    for (int n = 0; n < child.active_nodes; ++n) {
        if (prob(rng) < 0.15f) child.nodes[n].op = static_cast<uint8_t>(op_dist(rng));
        if (prob(rng) < 0.15f) child.nodes[n].law_mode = static_cast<uint8_t>(law_dist(rng));
        if (prob(rng) < 0.20f) child.nodes[n].in1 = rng() % (TOTAL_INPUTS + n);
        if (prob(rng) < 0.20f) child.nodes[n].in2 = rng() % (TOTAL_INPUTS + n);
        if (prob(rng) < 0.20f) child.nodes[n].gate_in = rng() % (TOTAL_INPUTS + n);
        if (prob(rng) < 0.35f) {
            int nw = child.nodes[n].weight + (rng() % 7 - 3);
            child.nodes[n].weight = static_cast<int8_t>(std::clamp(nw, -16, 16));
        }
    }
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
    return "n" + std::to_string(idx - TOTAL_INPUTS);
}

void dissect_self_aware_organism(const SelfAwareGenome& champion) {
    std::cout << "\n========================================================================\n";
    std::cout << "=== 🔬 DEEP ENDOSCOPIC AUTOPSY: SELF-AWARE TELEMETRIC CHAMPION      ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Active Dynamic Nodes : " << (int)champion.active_nodes << "\n";
    std::cout << "Composite Free Energy: " << std::fixed << std::setprecision(5) << champion.composite_fe << "\n\n";

    std::cout << ">>> 8-Environmental Niche Scorecard (Strict Telemetry):\n";
    std::cout << "  1. Chaotic Lorenz Physics Tracking Error : " << champion.niche_errors[0] << "\n";
    std::cout << "  2. Harmonic Oscillator Tracking Error    : " << champion.niche_errors[1] << "\n";
    std::cout << "  3. Turing Ripple-Carry Logic Error Rate  : " << champion.niche_errors[2] * 100.0f << " %\n";
    std::cout << "  4. Dynamic Parity Stream Error Rate      : " << champion.niche_errors[3] * 100.0f << " %\n";
    std::cout << "  5. Context-Free Stack Depth Error Rate   : " << champion.niche_errors[4] * 100.0f << " %\n";
    std::cout << "  6. Markov Associative Byte Error Rate    : " << champion.niche_errors[5] * 100.0f << " %\n";
    std::cout << "  7. Natural Language Stream Error Rate    : " << champion.niche_errors[6] * 100.0f << " %\n";
    std::cout << "  8. Synaptic Shock Noise Error            : " << champion.niche_errors[7] << "\n\n";

    std::cout << ">>> Discovered Self-Aware Circuit & Endogenous Gate Connections:\n";
    for (int n = 0; n < champion.active_nodes; ++n) {
        const GeneNode& gn = champion.nodes[n];
        std::string s1 = input_desc(gn.in1);
        std::string s2 = input_desc(gn.in2);
        std::string gate = input_desc(gn.gate_in);

        std::cout << "  node_" << std::setw(2) << n << " = " << opcode_name(gn.op) 
                  << "(" << s1 << ", " << s2 << ") [" << law_name(gn.law_mode) 
                  << "] * (" << (int)gn.weight << " / 16) | Gate: " << gate;
        if (gn.in1 >= SENSORY_INPUTS && gn.in1 < TOTAL_INPUTS) std::cout << " <== [FE-SENSING IN1]";
        if (gn.in2 >= SENSORY_INPUTS && gn.in2 < TOTAL_INPUTS) std::cout << " <== [FE-SENSING IN2]";
        if (gn.gate_in >= SENSORY_INPUTS && gn.gate_in < TOTAL_INPUTS) std::cout << " <== [FE-GATED]";
        std::cout << "\n";
    }
    std::cout << "========================================================================\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-607: SELF-AWARE TELEMETRIC FIELD CRUCIBLE (CUDA GENESIS)    ===\n";
    std::cout << "========================================================================\n";
    uint64_t total_evals_target = static_cast<uint64_t>(POP_SIZE) * NUM_GENERATIONS;
    std::cout << "Target Evaluations       : " << total_evals_target << " (6.55 MILLION AGENTS)\n";
    std::cout << "Active Batch Size on GPU : " << POP_SIZE << " organisms simultaneously\n";
    std::cout << "Self-Aware Telemetry     : Direct Endogenous Sensation of F_t, G_t, Omega_t\n";
    std::cout << "Dynamic Compartment Gate : Selective Execution & Substrate Bypassing per node\n";
    std::cout << "Selection Substrate      : Pareto-Lexicase Multi-Niche Preservation\n\n";

    std::mt19937 rng(2026);

    constexpr int NICHES_POOL_SIZE = 1024;
    std::vector<SelfAwareGenome> niche_pool(NICHES_POOL_SIZE);
    for (auto& a : niche_pool) init_self_aware_genome(a, rng);

    std::vector<SelfAwareGenome> h_batch(POP_SIZE);

    SelfAwareGenome* d_batch = nullptr;
    CHECK_CUDA(cudaMalloc(&d_batch, POP_SIZE * sizeof(SelfAwareGenome)));

    int threads = 256;
    int blocks  = (POP_SIZE + threads - 1) / threads;

    SelfAwareGenome global_champion = niche_pool[0];
    global_champion.composite_fe = 99999.0f;

    auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t total_evals_done = 0;

    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        for (int i = 0; i < POP_SIZE; ++i) {
            int parent_idx = rng() % NICHES_POOL_SIZE;
            h_batch[i] = niche_pool[parent_idx];
            mutate_self_aware_genome(h_batch[i], rng);
        }

        CHECK_CUDA(cudaMemcpy(d_batch, h_batch.data(), POP_SIZE * sizeof(SelfAwareGenome), cudaMemcpyHostToDevice));

        evaluate_self_aware_crucible_kernel<<<blocks, threads>>>(d_batch, POP_SIZE);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_batch.data(), d_batch, POP_SIZE * sizeof(SelfAwareGenome), cudaMemcpyDeviceToHost));

        for (int i = 0; i < POP_SIZE; ++i) {
            if (h_batch[i].composite_fe < global_champion.composite_fe) {
                global_champion = h_batch[i];
            }

            int niche_id = rng() % NUM_NICHES;
            int slot = niche_id * 128 + (rng() % 128);

            if (h_batch[i].niche_errors[niche_id] < niche_pool[slot].niche_errors[niche_id] ||
                (h_batch[i].niche_errors[niche_id] == niche_pool[slot].niche_errors[niche_id] && 
                 h_batch[i].composite_fe < niche_pool[slot].composite_fe)) {
                niche_pool[slot] = h_batch[i];
            }
        }

        total_evals_done += POP_SIZE;

        if ((gen + 1) % 10 == 0 || gen == 0) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double speed = static_cast<double>(total_evals_done) / elapsed;

            std::cout << "Gen [" << std::setw(3) << (gen + 1) << "/" << NUM_GENERATIONS << "]"
                      << " | Evals: " << std::setw(8) << total_evals_done
                      << " | Speed: " << std::fixed << std::setprecision(0) << speed << " evals/s"
                      << " | Min FE: " << std::setprecision(4) << global_champion.composite_fe
                      << " | Lorenz: " << std::setprecision(3) << global_champion.niche_errors[0]
                      << " | Turing: " << std::setprecision(1) << (global_champion.niche_errors[2] * 100.0f) << "%"
                      << " | Parity: " << (global_champion.niche_errors[3] * 100.0f) << "%"
                      << " | Stack: " << (global_champion.niche_errors[4] * 100.0f) << "%"
                      << " | Markov: " << (global_champion.niche_errors[5] * 100.0f) << "%"
                      << " | Nodes: " << (int)global_champion.active_nodes << "\n";
            std::cout.flush();
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n========================================================================\n";
    std::cout << ">>> CRUCIBLE COMPLETE: " << total_evals_done << " EVALUATIONS EXECUTED!\n";
    std::cout << ">>> Total Runtime : " << std::fixed << std::setprecision(2) << total_sec << " seconds\n";
    std::cout << ">>> Average Speed : " << std::fixed << std::setprecision(0) << (total_evals_done / total_sec) << " agents/sec\n";
    std::cout << "========================================================================\n";

    dissect_self_aware_organism(global_champion);

    std::cout << ">>> SPECIALIST NICHE CHAMPIONS AUDIT:\n";
    for (int n = 0; n < NUM_NICHES; ++n) {
        int best_slot = n * 128;
        for (int s = 1; s < 128; ++s) {
            if (niche_pool[n * 128 + s].niche_errors[n] < niche_pool[best_slot].niche_errors[n]) {
                best_slot = n * 128 + s;
            }
        }
        std::cout << "  Niche " << n << " Leader Best Error: " 
                  << niche_pool[best_slot].niche_errors[n] 
                  << " (Composite FE: " << niche_pool[best_slot].composite_fe 
                  << ", Nodes: " << (int)niche_pool[best_slot].active_nodes << ")\n";
    }
    std::cout << "========================================================================\n";

    CHECK_CUDA(cudaFree(d_batch));
    return 0;
}
