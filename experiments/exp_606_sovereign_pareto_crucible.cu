// ======================================================================================
// EXP-606: Totipotent Field Organism (Unshackled Substrate & Lexicase Pareto Crucible)
// --------------------------------------------------------------------------------------
// Fundamental Cybernetics (Bazilevs Sovereign Principles 2, 29, 30 & KEP Sovereign Spec):
// 1. ABSOLUTE SUBSTRATE SOVEREIGNTY (Zero Human Arbitrary Caps):
//    - High dynamic node capacity (up to 64 unconstrained nodes).
//    - Arbitrary topological wiring: any node can feed into any subsequent node,
//      can form arbitrary feedforward, residual, wide, multi-tier, or modular structures.
//    - Self-Modulating Dynamic Laws:
//        Each node dynamically synthesizes its own activation curvature & metric scale:
//        law_mode 0: Linear/Residual Identity
//        law_mode 1: Saturated Bounded Dynamic (Tanh/Saturate field)
//        law_mode 2: Hyper-Sharp Discontinuous Logic (Step/Threshold/Quantize)
//        law_mode 3: Gated Bilinear Multiplicative Field (SwiGLU/Product modulation)
//    - Dynamic Variable Memory Substrate (128 dynamic cells):
//        Direct register arrays, Key-Value associative content-addressed hash fields,
//        Bidirectional Push/Pop LIFO stacks, Accumulators.
// 2. PARETO-LEXICASE ENVIRONMENTAL NICHING (Death to Scalar Averaging):
//    - Eliminates the fatal scalar collapse where cheap continuous errors mask discrete failures.
//    - Each reality challenge is an independent epistemic niche:
//        Niche 1: Chaotic Continuous Dynamics (Lorenz Attractor Velocity)
//        Niche 2: Physical Damped Forced Oscillator (Mechanical Resonance)
//        Niche 3: Turing Ripple-Carry Logic (Arithmetic State Machine)
//        Niche 4: Dynamic Parity Stream (Sequence XOR / T-Flip-Flop)
//        Niche 5: Context-Free Bracket Balancing (Push/Pop Stack Memory)
//        Niche 6: Markov Associative Byte Transitions (Graph Flow)
//        Niche 7: Natural Linguistic Stream (Alpaca Next-Byte Prediction)
//        Niche 8: Synaptic Noise Filtering & Shock Recovery
//    - Selection preserves champions of distinct niches, enabling partial discoveries
//      (e.g., initial bracket detector) to survive and cross-pollinate without being pruned.
// 3. ZERO TIME CRUTCHES (Strictly No Step Counters in Inputs).
// 4. STRICT KEP VERDICT CRITERION:
//    - Absolutely NO green verdicts unless Free Energy is strictly near-zero across domains.
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

constexpr int POP_SIZE        = 65536;  // 65,536 parallel sovereign organisms
constexpr int NUM_GENERATIONS = 100;    // 6,553,600 deep evaluations on CUDA
constexpr int MAX_NODES       = 64;     // Unshackled node ceiling (up to 64 nodes)
constexpr int MEM_SIZE        = 128;    // 128 dynamic state registers & field memory
constexpr int NUM_INPUTS      = 4;      // Pure sensory inputs (strictly NO time clocks!)
constexpr int NUM_OUTPUTS     = 2;      
constexpr int NUM_NICHES      = 8;      // 8 independent environmental niches

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
    uint8_t in1;       // input index into (NUM_INPUTS + previous_nodes)
    uint8_t in2;       // input index into (NUM_INPUTS + previous_nodes)
    uint8_t law_mode;  // 0: Linear, 1: Saturate, 2: Threshold/Step, 3: Multiplicative
    int8_t  weight;    // Modulation scale [-16 .. 16]
};

struct alignas(64) SovereignGenome {
    GeneNode nodes[MAX_NODES];
    uint8_t  active_nodes;
    float    niche_errors[NUM_NICHES];
    float    composite_fe;
};

__device__ inline int16_t dev_clamp16(int32_t val, int32_t low, int32_t high) {
    return static_cast<int16_t>((val < low) ? low : ((val > high) ? high : val));
}

__device__ inline int16_t exec_totipotent_field_op(uint8_t op, uint8_t law_mode, int16_t a, int16_t b, int8_t w, int16_t* mem) {
    int32_t raw = 0;
    // Memory Substrate:
    // mem[0..63]   : Direct persistent registers
    // mem[64..95]  : Associative Key-Value hash field
    // mem[96..126] : Bidirectional LIFO stack buffer
    // mem[127]     : Stack pointer (sp)
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

    // Dynamic Law & Curvature Synthesis
    int32_t shaped = raw;
    switch(law_mode & 3) {
        case 0: // Linear/Identity
            shaped = raw;
            break;
        case 1: // Saturated Bounded Dynamic
            shaped = (raw < -256) ? -256 : ((raw > 255) ? 255 : raw);
            break;
        case 2: // Hyper-Sharp Discontinuous Step
            shaped = (raw > 0) ? 16 : ((raw < 0) ? -16 : 0);
            break;
        case 3: // Bilinear Multiplicative Field
            shaped = (raw * static_cast<int32_t>(b)) >> 4;
            break;
    }

    int32_t modulated = (shaped * static_cast<int32_t>(w)) >> 4;
    return dev_clamp16(modulated, -4096, 4095);
}

__device__ inline void run_totipotent_organism(const SovereignGenome& agent, const int16_t* inputs, int16_t* outputs, int16_t* mem) {
    int16_t bus[NUM_INPUTS + MAX_NODES];
    for (int i = 0; i < NUM_INPUTS; ++i) bus[i] = inputs[i];

    for (int n = 0; n < agent.active_nodes; ++n) {
        const GeneNode& gn = agent.nodes[n];
        int16_t in1_val = bus[gn.in1 % (NUM_INPUTS + n)];
        int16_t in2_val = bus[gn.in2 % (NUM_INPUTS + n)];
        bus[NUM_INPUTS + n] = exec_totipotent_field_op(gn.op, gn.law_mode, in1_val, in2_val, gn.weight, mem);
    }

    outputs[0] = bus[NUM_INPUTS + agent.active_nodes - 1];
    outputs[1] = (agent.active_nodes > 1) ? bus[NUM_INPUTS + agent.active_nodes - 2] : 0;
}

__global__ void evaluate_sovereign_crucible_kernel(SovereignGenome* agents, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    SovereignGenome& agent = agents[idx];
    int16_t mem[MEM_SIZE];
    #define FLUSH_SOVEREIGN_MEM() for(int m=0; m<MEM_SIZE; ++m) mem[m] = 0;

    // ----------------------------------------------------------------------------------
    // NICHE 1: Chaotic Continuous Dynamics (Lorenz Attractor Velocity - Zero Clocks)
    // ----------------------------------------------------------------------------------
    FLUSH_SOVEREIGN_MEM();
    float err_lorenz = 0.0f;
    int16_t lx = 10, ly = 20, lz = 30;
    for (int t = 0; t < 32; ++t) {
        int16_t dx = (10 * (ly - lx)) >> 3;
        int16_t target_x = lx + dx;
        int16_t in[NUM_INPUTS] = {lx, ly, lz, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, in, out, mem);
        err_lorenz += fabsf(static_cast<float>(out[0] - target_x));
        lx = target_x; ly += 5; lz += 2;
    }
    agent.niche_errors[0] = err_lorenz / 32.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 2: Physical Damped Forced Oscillator (Continuous Mechanics)
    // ----------------------------------------------------------------------------------
    FLUSH_SOVEREIGN_MEM();
    float err_osc = 0.0f;
    int16_t sx = 0, sv = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t force = (t & 1) ? 16 : -16;
        int16_t target_x = sx + sv;
        int16_t in[NUM_INPUTS] = {sx, sv, force, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, in, out, mem);
        err_osc += fabsf(static_cast<float>(out[0] - target_x));
        sv = (14 * sv - 4 * sx + force) >> 4;
        sx = target_x;
    }
    agent.niche_errors[1] = err_osc / 32.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 3: Turing Multi-Bit Ripple Carry Logic (Discrete ALU with internal carry)
    // ----------------------------------------------------------------------------------
    FLUSH_SOVEREIGN_MEM();
    float err_carry = 0.0f;
    int16_t carry = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t a = (t * 7) & 1;
        int16_t b = (t * 13) & 1;
        int16_t tgt_sum = a ^ b ^ carry;
        int16_t tgt_carry = (a & b) | (carry & (a ^ b));
        int16_t in[NUM_INPUTS] = {a, b, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, in, out, mem);
        if ((out[0] & 1) != tgt_sum) err_carry += 1.0f;
        carry = tgt_carry;
    }
    agent.niche_errors[2] = err_carry / 32.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 4: Dynamic Parity Stream (Sequence XOR / T-Flip-Flop without Step Clocks)
    // ----------------------------------------------------------------------------------
    FLUSH_SOVEREIGN_MEM();
    float err_parity = 0.0f;
    int16_t par = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t bit = (t * 5 + 1) & 1;
        par ^= bit;
        int16_t in[NUM_INPUTS] = {bit, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, in, out, mem);
        if ((out[0] & 1) != par) err_parity += 1.0f;
    }
    agent.niche_errors[3] = err_parity / 32.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 5: Context-Free Bracket Balancing (Push/Pop Stack Memory - Pure Char Input)
    // ----------------------------------------------------------------------------------
    FLUSH_SOVEREIGN_MEM();
    float err_bracket = 0.0f;
    int depth = 0;
    const char brackets[17] = "(())((()))()(())";
    for (int t = 0; t < 16; ++t) {
        char b = brackets[t];
        depth += (b == '(') ? 1 : -1;
        int16_t in[NUM_INPUTS] = {static_cast<int16_t>(b), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, in, out, mem);
        if (out[0] != depth) err_bracket += 1.0f;
    }
    agent.niche_errors[4] = err_bracket / 16.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 6: Markov Associative Byte Transitions (Graph Flow)
    // ----------------------------------------------------------------------------------
    FLUSH_SOVEREIGN_MEM();
    float err_markov = 0.0f;
    const char seq[17] = "abcdabcdabcdabcd";
    for (int t = 0; t < 15; ++t) {
        int16_t cur = static_cast<int16_t>(seq[t]);
        int16_t tgt = static_cast<int16_t>(seq[t + 1]);
        int16_t in[NUM_INPUTS] = {cur, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, in, out, mem);
        if (out[0] != tgt) err_markov += 1.0f;
    }
    agent.niche_errors[5] = err_markov / 15.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 7: Natural Linguistic Stream (Alpaca Raw Text Next-Byte Prediction)
    // ----------------------------------------------------------------------------------
    FLUSH_SOVEREIGN_MEM();
    float err_lang = 0.0f;
    const char phrase[33] = "instruction_response_karyon_core";
    for (int t = 0; t < 31; ++t) {
        uint8_t cur = static_cast<uint8_t>(phrase[t]);
        uint8_t nxt = static_cast<uint8_t>(phrase[t + 1]);
        int16_t in[NUM_INPUTS] = {static_cast<int16_t>(cur), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, in, out, mem);
        if (static_cast<uint8_t>(out[0] & 0xFF) != nxt) err_lang += 1.0f;
    }
    agent.niche_errors[6] = err_lang / 31.0f;

    // ----------------------------------------------------------------------------------
    // NICHE 8: Synaptic Shock & Noise Immunity
    // ----------------------------------------------------------------------------------
    float err_shock = 0.0f;
    for (int t = 0; t < 16; ++t) {
        int16_t clean = t * 4;
        int16_t noisy = clean + ((t & 1) ? 3 : -3);
        int16_t in[NUM_INPUTS] = {noisy, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_totipotent_organism(agent, in, out, mem);
        err_shock += fabsf(static_cast<float>(out[0] - clean));
    }
    agent.niche_errors[7] = err_shock / 16.0f;

    // Normalized Master Free Energy (for tracking global convergence)
    agent.composite_fe = agent.niche_errors[0] * 0.10f + // Lorenz
                         agent.niche_errors[1] * 0.10f + // Oscillator
                         agent.niche_errors[2] * 2.00f + // Turing
                         agent.niche_errors[3] * 2.00f + // Parity
                         agent.niche_errors[4] * 2.00f + // Stack
                         agent.niche_errors[5] * 2.00f + // Markov
                         agent.niche_errors[6] * 2.00f + // Language
                         agent.niche_errors[7] * 0.10f;  // Shock
}

void init_sovereign_genome(SovereignGenome& agent, std::mt19937& rng) {
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
        agent.nodes[n].in1 = rng() % (NUM_INPUTS + n);
        agent.nodes[n].in2 = rng() % (NUM_INPUTS + n);
        agent.nodes[n].weight = static_cast<int8_t>(w_dist(rng));
    }
}

void mutate_sovereign_genome(SovereignGenome& child, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_int_distribution<int> op_dist(1, NUM_OPS - 1);
    std::uniform_int_distribution<int> law_dist(0, 3);

    // Unconstrained Sprouting & Pruning
    if (prob(rng) < 0.25f && child.active_nodes < MAX_NODES) child.active_nodes++;
    if (prob(rng) < 0.10f && child.active_nodes > 4) child.active_nodes--;

    for (int n = 0; n < child.active_nodes; ++n) {
        if (prob(rng) < 0.15f) child.nodes[n].op = static_cast<uint8_t>(op_dist(rng));
        if (prob(rng) < 0.15f) child.nodes[n].law_mode = static_cast<uint8_t>(law_dist(rng));
        if (prob(rng) < 0.20f) child.nodes[n].in1 = rng() % (NUM_INPUTS + n);
        if (prob(rng) < 0.20f) child.nodes[n].in2 = rng() % (NUM_INPUTS + n);
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

void dissect_sovereign_organism(const SovereignGenome& champion) {
    std::cout << "\n========================================================================\n";
    std::cout << "=== 🔬 DEEP ENDOSCOPIC AUTOPSY: UNSHACKLED TOTIPOTENT SOVEREIGN      ===\n";
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

    std::cout << ">>> Discovered Dynamic Field Circuit & Synthesized Laws:\n";
    for (int n = 0; n < champion.active_nodes; ++n) {
        const GeneNode& gn = champion.nodes[n];
        std::string s1 = (gn.in1 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in1) + "]") : ("n" + std::to_string(gn.in1 - NUM_INPUTS));
        std::string s2 = (gn.in2 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in2) + "]") : ("n" + std::to_string(gn.in2 - NUM_INPUTS));

        std::cout << "  node_" << std::setw(2) << n << " = " << opcode_name(gn.op) 
                  << "(" << s1 << ", " << s2 << ") [" << law_name(gn.law_mode) << "] * (" << (int)gn.weight << " / 16)";
        if (gn.op >= OP_LOAD_REG) {
            std::cout << "  <== [FIELD MEMORY OP]";
        }
        std::cout << "\n";
    }
    std::cout << "========================================================================\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-606: TOTIPOTENT FIELD ORGANISM (PARETO-LEXICASE CRUCIBLE)   ===\n";
    std::cout << "========================================================================\n";
    uint64_t total_evals_target = static_cast<uint64_t>(POP_SIZE) * NUM_GENERATIONS;
    std::cout << "Target Evaluations       : " << total_evals_target << " (6.55 MILLION AGENTS)\n";
    std::cout << "Active Batch Size on GPU : " << POP_SIZE << " organisms simultaneously\n";
    std::cout << "Topological Freedom      : Unconstrained Dynamic DAG (up to 64 nodes)\n";
    std::cout << "Law Genesis              : Endogenous Synthesized Activation Modes (4 Field Laws)\n";
    std::cout << "Memory Architecture      : 128 Registers (Scalars + Hash Tables + Stacks)\n";
    std::cout << "Selection Mechanism      : Environmental Niching & Pareto-Lexicase Preservation\n\n";

    std::mt19937 rng(2026);

    // Environmental Niche Champions Pool (Preserving specialist leaders across all 8 niches)
    constexpr int NICHES_POOL_SIZE = 1024;
    std::vector<SovereignGenome> niche_pool(NICHES_POOL_SIZE);
    for (auto& a : niche_pool) init_sovereign_genome(a, rng);

    std::vector<SovereignGenome> h_batch(POP_SIZE);

    SovereignGenome* d_batch = nullptr;
    CHECK_CUDA(cudaMalloc(&d_batch, POP_SIZE * sizeof(SovereignGenome)));

    int threads = 256;
    int blocks  = (POP_SIZE + threads - 1) / threads;

    SovereignGenome global_champion = niche_pool[0];
    global_champion.composite_fe = 99999.0f;

    auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t total_evals_done = 0;

    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        // Reproduce from the multi-objective environmental niche pool
        for (int i = 0; i < POP_SIZE; ++i) {
            int parent_idx = rng() % NICHES_POOL_SIZE;
            h_batch[i] = niche_pool[parent_idx];
            mutate_sovereign_genome(h_batch[i], rng);
        }

        CHECK_CUDA(cudaMemcpy(d_batch, h_batch.data(), POP_SIZE * sizeof(SovereignGenome), cudaMemcpyHostToDevice));

        evaluate_sovereign_crucible_kernel<<<blocks, threads>>>(d_batch, POP_SIZE);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_batch.data(), d_batch, POP_SIZE * sizeof(SovereignGenome), cudaMemcpyDeviceToHost));

        // Pareto-Lexicase Environmental Niche Selection
        for (int i = 0; i < POP_SIZE; ++i) {
            // Track absolute minimum composite Free Energy
            if (h_batch[i].composite_fe < global_champion.composite_fe) {
                global_champion = h_batch[i];
            }

            // Assign into environmental niche slots (128 slots per niche for 8 niches = 1024 pool)
            int niche_id = rng() % NUM_NICHES;
            int slot = niche_id * 128 + (rng() % 128);

            // Niche-specific Pareto dominance: replace if better on THIS specific niche
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

    dissect_sovereign_organism(global_champion);

    // Also inspect niche specialists
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
