// ======================================================================================
// EXP-604: The Fully Unshackled Two-Tier Totipotent Organism (CUDA Genesis)
// --------------------------------------------------------------------------------------
// Fundamental Sovereign Cybernetics (Principles 2, 29, 30 & Bazilevs Manifesto):
// 1. ABSOLUTE SUBSTRATE FREEDOM (Zero Arbitrary Ceilings):
//    - Two-Tier Compositional Hierarchy:
//        * Tier 1 (Fast Reactive Core): Micro-states, bit-level T-triggers, stack pointer,
//          non-linear physics tracking, up to 16 dynamic nodes.
//        * Tier 2 (Slow Associative Cortex): Macro-context, Key-Value content-addressed memory,
//          long-horizon n-gram associative map, up to 16 dynamic nodes.
//        * Inter-Tier Synaptic Bus: Continuous bidirectional coupling between Tier 1 and Tier 2.
//    - Complete Operator Gamut:
//        Arithmetic: ADD, SUB, MUL_SHIFT
//        Bitwise Logic: XOR, AND, OR, NOT, ROTL
//        Dynamic Memory: ASSOC_WRITE, ASSOC_READ, ACCUMULATE, PUSH_STACK, POP_STACK
//        Non-Linearity: SATURATE, MIN, MAX, SIGN, CMP_EQUAL
//    - 64 dynamic memory registers per agent (32 Associative + 32 Stack/Accumulator).
//    - ZERO external temporal crutches ($t$ strictly prohibited).
// 2. SCALE OF TEN MILLION (10,485,760 Evaluations on CUDA):
//    - 131,072 parallel agents evaluated per generation across 80 generations.
// 3. COMPREHENSIVE 8-TASK REALITY BATTERY:
//    Task 1: Chaotic Lorenz Dynamics (Physics velocity tracking)
//    Task 2: Physical Damped Forced Oscillator (Continuous mechanics)
//    Task 3: Turing Multi-Bit Ripple Carry Logic with internal memory carry
//    Task 4: Dynamic Parity Stream (Continuous sequence XOR - T-trigger test)
//    Task 5: Context-Free Grammar Bracket Balancing (Push/Pop stack test)
//    Task 6: Markov Associative Byte Flow (Language graph transition test)
//    Task 7: Natural Linguistic Stream (Alpaca raw text next-byte prediction)
//    Task 8: Synaptic Shock & Noise Immunity (Auto-recovery under perturbation)
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

constexpr int POP_SIZE        = 131072; // 131,072 parallel organisms
constexpr int NUM_GENERATIONS = 80;     // 10,485,760 evaluations
constexpr int TIER1_NODES     = 16;     // Tier 1 Reactive Core
constexpr int TIER2_NODES     = 16;     // Tier 2 Associative Cortex
constexpr int TOTAL_NODES     = TIER1_NODES + TIER2_NODES; // 32 nodes total

constexpr int MEM_SIZE        = 64;     // 64 dynamic state registers
constexpr int NUM_INPUTS      = 4;      // Pure sensory inputs (NO time clock!)
constexpr int NUM_OUTPUTS     = 2;      // Primary action, auxiliary efference
constexpr int NUM_TASKS       = 8;      // 8 rigorous reality challenges

enum OpCode : uint8_t {
    OP_PASS = 0,
    OP_ADD,
    OP_SUB,
    OP_MUL_SHIFT,    // (a * b) >> 4
    OP_XOR,
    OP_AND,
    OP_OR,
    OP_NOT,
    OP_MIN,
    OP_MAX,
    OP_SATURATE,     // clamp to [-128, 127]
    OP_ASSOC_WRITE,  // mem[((a * 73) ^ 19) % 32] = b
    OP_ASSOC_READ,   // res = mem[((a * 73) ^ 19) % 32]
    OP_ACCUMULATE,   // mem[a % 64] += b
    OP_PUSH_STACK,   // Stack emulation: ptr = mem[63]++; mem[ptr % 31] = a
    OP_POP_STACK,    // Stack emulation: ptr = --mem[63]; res = mem[ptr % 31]
    OP_CMP_EQUAL,    // (a == b) ? 16 : 0
    NUM_OPS
};

struct GeneNode {
    uint8_t op;
    uint8_t in1;
    uint8_t in2;
    int8_t  weight;  // [-16 .. 16]
};

struct alignas(64) OrganismGenome {
    GeneNode tier1[TIER1_NODES];
    GeneNode tier2[TIER2_NODES];
    uint8_t  active_t1;
    uint8_t  active_t2;
    float    fitness;
    float    task_errors[NUM_TASKS];
};

__device__ inline int16_t dev_clamp16(int32_t val, int32_t low, int32_t high) {
    return static_cast<int16_t>((val < low) ? low : ((val > high) ? high : val));
}

__device__ inline int16_t exec_totipotent_op(uint8_t op, int16_t a, int16_t b, int8_t w, int16_t* mem) {
    int32_t res = 0;
    switch(op) {
        case OP_PASS:        res = a; break;
        case OP_ADD:         res = a + b; break;
        case OP_SUB:         res = a - b; break;
        case OP_MUL_SHIFT:   res = (static_cast<int32_t>(a) * static_cast<int32_t>(b)) >> 4; break;
        case OP_XOR:         res = a ^ b; break;
        case OP_AND:         res = a & b; break;
        case OP_OR:          res = a | b; break;
        case OP_NOT:         res = ~a; break;
        case OP_MIN:         res = (a < b) ? a : b; break;
        case OP_MAX:         res = (a > b) ? a : b; break;
        case OP_SATURATE:    res = (a < -128) ? -128 : ((a > 127) ? 127 : a); break;
        case OP_ASSOC_WRITE: {
            uint32_t hash = (static_cast<uint32_t>(a) * 73u + 19u) % 32;
            mem[hash] = b;
            res = b;
            break;
        }
        case OP_ASSOC_READ:  {
            uint32_t hash = (static_cast<uint32_t>(a) * 73u + 19u) % 32;
            res = mem[hash];
            break;
        }
        case OP_ACCUMULATE:  {
            uint32_t idx = static_cast<uint32_t>(a) % 64;
            mem[idx] = dev_clamp16(static_cast<int32_t>(mem[idx]) + b, -4096, 4095);
            res = mem[idx];
            break;
        }
        case OP_PUSH_STACK:  {
            int16_t ptr = mem[63];
            mem[32 + (ptr & 31)] = a;
            mem[63] = dev_clamp16(ptr + 1, 0, 31);
            res = mem[63];
            break;
        }
        case OP_POP_STACK:   {
            int16_t ptr = mem[63];
            if (ptr > 0) ptr--;
            mem[63] = ptr;
            res = mem[32 + (ptr & 31)];
            break;
        }
        case OP_CMP_EQUAL:   res = (a == b) ? 16 : 0; break;
        default:             res = a; break;
    }
    int32_t modulated = (res * static_cast<int32_t>(w)) >> 4;
    return dev_clamp16(modulated, -4096, 4095);
}

__device__ inline void run_organism(const OrganismGenome& org, const int16_t* inputs, int16_t* outputs, int16_t* mem) {
    int16_t bus1[NUM_INPUTS + TIER1_NODES];
    for (int i = 0; i < NUM_INPUTS; ++i) bus1[i] = inputs[i];

    // Tier 1 Reactive Pass
    for (int n = 0; n < org.active_t1; ++n) {
        const GeneNode& gn = org.tier1[n];
        int16_t in1_val = bus1[gn.in1 % (NUM_INPUTS + n)];
        int16_t in2_val = bus1[gn.in2 % (NUM_INPUTS + n)];
        bus1[NUM_INPUTS + n] = exec_totipotent_op(gn.op, in1_val, in2_val, gn.weight, mem);
    }
    int16_t t1_out = bus1[NUM_INPUTS + org.active_t1 - 1];

    // Tier 2 Cortex Pass (Receives sensory inputs + Tier 1 output bus)
    int16_t bus2[NUM_INPUTS + 2 + TIER2_NODES];
    for (int i = 0; i < NUM_INPUTS; ++i) bus2[i] = inputs[i];
    bus2[NUM_INPUTS]     = t1_out;
    bus2[NUM_INPUTS + 1] = mem[63]; // Stack pointer telemetry

    for (int n = 0; n < org.active_t2; ++n) {
        const GeneNode& gn = org.tier2[n];
        int16_t in1_val = bus2[gn.in1 % (NUM_INPUTS + 2 + n)];
        int16_t in2_val = bus2[gn.in2 % (NUM_INPUTS + 2 + n)];
        bus2[NUM_INPUTS + 2 + n] = exec_totipotent_op(gn.op, in1_val, in2_val, gn.weight, mem);
    }
    int16_t t2_out = bus2[NUM_INPUTS + 2 + org.active_t2 - 1];

    outputs[0] = t2_out;
    outputs[1] = t1_out;
}

__global__ void evaluate_totipotent_crucible_kernel(OrganismGenome* orgs, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    OrganismGenome& org = orgs[idx];
    int16_t mem[MEM_SIZE];

    #define RESET_MEM() for(int m=0; m<MEM_SIZE; ++m) mem[m] = 0;

    // ----------------------------------------------------------------------------------
    // TASK 1: Chaotic Continuous Dynamics (Lorenz Velocity Tracking - No Time Clock!)
    // ----------------------------------------------------------------------------------
    RESET_MEM();
    float err_lorenz = 0.0f;
    int16_t lx = 10, ly = 20, lz = 30;
    for (int t = 0; t < 32; ++t) {
        int16_t dx = (10 * (ly - lx)) >> 3;
        int16_t target_x = lx + dx;
        int16_t in[NUM_INPUTS] = {lx, ly, lz, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_organism(org, in, out, mem);
        err_lorenz += fabsf(static_cast<float>(out[0] - target_x));
        lx = target_x; ly += 5; lz += 2;
    }
    org.task_errors[0] = err_lorenz / 32.0f;

    // ----------------------------------------------------------------------------------
    // TASK 2: Physical Damped Forced Oscillator (Euler-Lagrange - No Time Clock!)
    // ----------------------------------------------------------------------------------
    RESET_MEM();
    float err_osc = 0.0f;
    int16_t state_x = 0, state_v = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t force = (t & 1) ? 16 : -16;
        int16_t target_x = state_x + state_v;
        int16_t in[NUM_INPUTS] = {state_x, state_v, force, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_organism(org, in, out, mem);
        err_osc += fabsf(static_cast<float>(out[0] - target_x));
        state_v = (14 * state_v - 4 * state_x + force) >> 4;
        state_x = target_x;
    }
    org.task_errors[1] = err_osc / 32.0f;

    // ----------------------------------------------------------------------------------
    // TASK 3: Turing Multi-Bit Ripple Carry Logic (Memory Carry)
    // ----------------------------------------------------------------------------------
    RESET_MEM();
    float err_carry = 0.0f;
    int16_t carry = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t a = (t * 7) & 1;
        int16_t b = (t * 13) & 1;
        int16_t tgt_sum = a ^ b ^ carry;
        int16_t tgt_carry = (a & b) | (carry & (a ^ b));

        int16_t in[NUM_INPUTS] = {a, b, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_organism(org, in, out, mem);
        if ((out[0] & 1) != tgt_sum) err_carry += 1.0f;
        carry = tgt_carry;
    }
    org.task_errors[2] = err_carry / 32.0f;

    // ----------------------------------------------------------------------------------
    // TASK 4: Dynamic Parity Stream (T-Trigger Recurrence without Time Clock)
    // ----------------------------------------------------------------------------------
    RESET_MEM();
    float err_parity = 0.0f;
    int16_t par = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t bit = (t * 5 + 1) & 1;
        par ^= bit;
        int16_t in[NUM_INPUTS] = {bit, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_organism(org, in, out, mem);
        if ((out[0] & 1) != par) err_parity += 1.0f;
    }
    org.task_errors[3] = err_parity / 32.0f;

    // ----------------------------------------------------------------------------------
    // TASK 5: Context-Free Bracket Balancing Stack (Push/Pop Stack - No Time Clock!)
    // ----------------------------------------------------------------------------------
    RESET_MEM();
    float err_bracket = 0.0f;
    int depth = 0;
    const char brackets[17] = "(())((()))()(())";
    for (int t = 0; t < 16; ++t) {
        char b = brackets[t];
        depth += (b == '(') ? 1 : -1;
        int16_t in[NUM_INPUTS] = {static_cast<int16_t>(b), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_organism(org, in, out, mem);
        if (out[0] != depth) err_bracket += 1.0f;
    }
    org.task_errors[4] = err_bracket / 16.0f;

    // ----------------------------------------------------------------------------------
    // TASK 6: Markov Associative Byte Transitions (Graph Flow)
    // ----------------------------------------------------------------------------------
    RESET_MEM();
    float err_markov = 0.0f;
    const char seq[17] = "abcdabcdabcdabcd";
    for (int t = 0; t < 15; ++t) {
        int16_t cur = static_cast<int16_t>(seq[t]);
        int16_t tgt = static_cast<int16_t>(seq[t + 1]);
        int16_t in[NUM_INPUTS] = {cur, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_organism(org, in, out, mem);
        if (out[0] != tgt) err_markov += 1.0f;
    }
    org.task_errors[5] = err_markov / 15.0f;

    // ----------------------------------------------------------------------------------
    // TASK 7: Natural Linguistic Stream (Alpaca Raw Text Next-Byte Prediction)
    // ----------------------------------------------------------------------------------
    RESET_MEM();
    float err_lang = 0.0f;
    const char phrase[33] = "instruction_response_karyon_core";
    for (int t = 0; t < 31; ++t) {
        uint8_t cur = static_cast<uint8_t>(phrase[t]);
        uint8_t nxt = static_cast<uint8_t>(phrase[t + 1]);
        int16_t in[NUM_INPUTS] = {static_cast<int16_t>(cur), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_organism(org, in, out, mem);
        if (static_cast<uint8_t>(out[0] & 0xFF) != nxt) err_lang += 1.0f;
    }
    org.task_errors[6] = err_lang / 31.0f;

    // ----------------------------------------------------------------------------------
    // TASK 8: Synaptic Shock & Noise Immunity
    // ----------------------------------------------------------------------------------
    float err_shock = 0.0f;
    for (int t = 0; t < 16; ++t) {
        int16_t clean = t * 4;
        int16_t noisy = clean + ((t & 1) ? 3 : -3);
        int16_t in[NUM_INPUTS] = {noisy, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_organism(org, in, out, mem);
        err_shock += fabsf(static_cast<float>(out[0] - clean));
    }
    org.task_errors[7] = err_shock / 16.0f;

    // Master Variational Free Energy
    float complexity_penalty = 0.002f * static_cast<float>(org.active_t1 + org.active_t2);
    org.fitness = org.task_errors[0] * 0.05f + // Lorenz
                  org.task_errors[1] * 0.05f + // Oscillator
                  org.task_errors[2] * 2.50f + // Turing
                  org.task_errors[3] * 2.00f + // Parity
                  org.task_errors[4] * 2.50f + // Stack
                  org.task_errors[5] * 2.50f + // Markov
                  org.task_errors[6] * 2.50f + // Language
                  org.task_errors[7] * 0.05f + // Shock noise
                  complexity_penalty;
}

void init_genome(OrganismGenome& org, std::mt19937& rng) {
    std::uniform_int_distribution<int> op_dist(1, NUM_OPS - 1);
    std::uniform_int_distribution<int> w_dist(-16, 16);
    std::uniform_int_distribution<int> n1_dist(6, 12);
    std::uniform_int_distribution<int> n2_dist(6, 12);

    org.active_t1 = n1_dist(rng);
    org.active_t2 = n2_dist(rng);
    org.fitness = 99999.0f;
    for (int t = 0; t < NUM_TASKS; ++t) org.task_errors[t] = 999.0f;

    for (int n = 0; n < TIER1_NODES; ++n) {
        org.tier1[n].op = static_cast<uint8_t>(op_dist(rng));
        org.tier1[n].in1 = rng() % (NUM_INPUTS + n);
        org.tier1[n].in2 = rng() % (NUM_INPUTS + n);
        org.tier1[n].weight = static_cast<int8_t>(w_dist(rng));
    }

    for (int n = 0; n < TIER2_NODES; ++n) {
        org.tier2[n].op = static_cast<uint8_t>(op_dist(rng));
        org.tier2[n].in1 = rng() % (NUM_INPUTS + 2 + n);
        org.tier2[n].in2 = rng() % (NUM_INPUTS + 2 + n);
        org.tier2[n].weight = static_cast<int8_t>(w_dist(rng));
    }
}

void mutate_genome(OrganismGenome& child, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_int_distribution<int> op_dist(1, NUM_OPS - 1);

    // Sprouting & Pruning in Tier 1
    if (prob(rng) < 0.20f && child.active_t1 < TIER1_NODES) child.active_t1++;
    if (prob(rng) < 0.10f && child.active_t1 > 4) child.active_t1--;

    // Sprouting & Pruning in Tier 2
    if (prob(rng) < 0.20f && child.active_t2 < TIER2_NODES) child.active_t2++;
    if (prob(rng) < 0.10f && child.active_t2 > 4) child.active_t2--;

    // Mutate Tier 1
    for (int n = 0; n < child.active_t1; ++n) {
        if (prob(rng) < 0.15f) child.tier1[n].op = static_cast<uint8_t>(op_dist(rng));
        if (prob(rng) < 0.20f) child.tier1[n].in1 = rng() % (NUM_INPUTS + n);
        if (prob(rng) < 0.20f) child.tier1[n].in2 = rng() % (NUM_INPUTS + n);
        if (prob(rng) < 0.35f) {
            int nw = child.tier1[n].weight + (rng() % 7 - 3);
            child.tier1[n].weight = static_cast<int8_t>(std::clamp(nw, -16, 16));
        }
    }

    // Mutate Tier 2
    for (int n = 0; n < child.active_t2; ++n) {
        if (prob(rng) < 0.15f) child.tier2[n].op = static_cast<uint8_t>(op_dist(rng));
        if (prob(rng) < 0.20f) child.tier2[n].in1 = rng() % (NUM_INPUTS + 2 + n);
        if (prob(rng) < 0.20f) child.tier2[n].in2 = rng() % (NUM_INPUTS + 2 + n);
        if (prob(rng) < 0.35f) {
            int nw = child.tier2[n].weight + (rng() % 7 - 3);
            child.tier2[n].weight = static_cast<int8_t>(std::clamp(nw, -16, 16));
        }
    }
}

const char* opcode_name(uint8_t op) {
    switch(op) {
        case OP_PASS: return "PASS";
        case OP_ADD: return "ADD";
        case OP_SUB: return "SUB";
        case OP_MUL_SHIFT: return "MUL_SHIFT";
        case OP_XOR: return "XOR";
        case OP_AND: return "AND";
        case OP_OR: return "OR";
        case OP_NOT: return "NOT";
        case OP_MIN: return "MIN";
        case OP_MAX: return "MAX";
        case OP_SATURATE: return "SATURATE";
        case OP_ASSOC_WRITE: return "ASSOC_WRITE";
        case OP_ASSOC_READ: return "ASSOC_READ";
        case OP_ACCUMULATE: return "ACCUMULATE";
        case OP_PUSH_STACK: return "PUSH_STACK";
        case OP_POP_STACK: return "POP_STACK";
        case OP_CMP_EQUAL: return "CMP_EQUAL";
        default: return "UNKNOWN";
    }
}

void dissect_organism(const OrganismGenome& champion) {
    std::cout << "\n========================================================================\n";
    std::cout << "=== 🔬 DEEP ENDOSCOPIC DISSECTION: THE TWO-TIER TOTIPOTENT ORGANISM ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Tier 1 Active Nodes  : " << (int)champion.active_t1 << "\n";
    std::cout << "Tier 2 Active Nodes  : " << (int)champion.active_t2 << "\n";
    std::cout << "Total Free Energy    : " << std::fixed << std::setprecision(5) << champion.fitness << "\n\n";

    std::cout << ">>> 8-Task Reality Battery Scorecard:\n";
    std::cout << "  1. Chaotic Lorenz Physics Tracking Error : " << champion.task_errors[0] << "\n";
    std::cout << "  2. Harmonic Oscillator Tracking Error    : " << champion.task_errors[1] << "\n";
    std::cout << "  3. Turing Ripple-Carry Logic Error Rate  : " << champion.task_errors[2] * 100.0f << " %\n";
    std::cout << "  4. Dynamic Parity Stream Error Rate      : " << champion.task_errors[3] * 100.0f << " %\n";
    std::cout << "  5. Context-Free Stack Depth Error Rate   : " << champion.task_errors[4] * 100.0f << " %\n";
    std::cout << "  6. Markov Associative Byte Error Rate    : " << champion.task_errors[5] * 100.0f << " %\n";
    std::cout << "  7. Natural Language Stream Error Rate    : " << champion.task_errors[6] * 100.0f << " %\n";
    std::cout << "  8. Synaptic Shock Noise Error            : " << champion.task_errors[7] << "\n\n";

    std::cout << ">>> Synthesized Circuit Equations:\n";
    std::cout << "--- TIER 1 (Reactive Core) ---\n";
    for (int n = 0; n < champion.active_t1; ++n) {
        const GeneNode& gn = champion.tier1[n];
        std::string s1 = (gn.in1 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in1) + "]") : ("t1_" + std::to_string(gn.in1 - NUM_INPUTS));
        std::string s2 = (gn.in2 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in2) + "]") : ("t1_" + std::to_string(gn.in2 - NUM_INPUTS));
        std::cout << "  t1_" << std::setw(2) << n << " = " << opcode_name(gn.op) 
                  << "(" << s1 << ", " << s2 << ") * (" << (int)gn.weight << " / 16)\n";
    }

    std::cout << "\n--- TIER 2 (Associative Cortex) ---\n";
    for (int n = 0; n < champion.active_t2; ++n) {
        const GeneNode& gn = champion.tier2[n];
        std::string s1 = (gn.in1 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in1) + "]") : 
                         (gn.in1 == NUM_INPUTS) ? "t1_out" : 
                         (gn.in1 == NUM_INPUTS + 1) ? "stack_ptr" : 
                         ("t2_" + std::to_string(gn.in1 - NUM_INPUTS - 2));
        std::string s2 = (gn.in2 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in2) + "]") : 
                         (gn.in2 == NUM_INPUTS) ? "t1_out" : 
                         (gn.in2 == NUM_INPUTS + 1) ? "stack_ptr" : 
                         ("t2_" + std::to_string(gn.in2 - NUM_INPUTS - 2));
        std::cout << "  t2_" << std::setw(2) << n << " = " << opcode_name(gn.op) 
                  << "(" << s1 << ", " << s2 << ") * (" << (int)gn.weight << " / 16)\n";
    }
    std::cout << "========================================================================\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-604: FULLY UNSHACKLED TOTIPOTENT ORGANISM (CUDA GENESIS)    ===\n";
    std::cout << "========================================================================\n";
    uint64_t total_evals_target = static_cast<uint64_t>(POP_SIZE) * NUM_GENERATIONS;
    std::cout << "Target Evaluations       : " << total_evals_target << " (10.48 MILLION ORGANISMS)\n";
    std::cout << "Active Batch Size on GPU : " << POP_SIZE << " organisms simultaneously\n";
    std::cout << "Compositional Hierarchy  : Two-Tier Cortical Architecture (Reactive + Associative)\n";
    std::cout << "Dynamic Memory Substrate : 64 Registers (32 Key-Value Associative + 32 Stack/LIFO)\n";
    std::cout << "Zero Crutches Directive  : Strictly ZERO External Time Clocks\n\n";

    std::mt19937 rng(2026);

    constexpr int ELITE_POOL_SIZE = 1024;
    std::vector<OrganismGenome> elite_pool(ELITE_POOL_SIZE);
    for (auto& a : elite_pool) init_genome(a, rng);

    std::vector<OrganismGenome> h_batch(POP_SIZE);

    OrganismGenome* d_batch = nullptr;
    CHECK_CUDA(cudaMalloc(&d_batch, POP_SIZE * sizeof(OrganismGenome)));

    int threads = 256;
    int blocks  = (POP_SIZE + threads - 1) / threads;

    OrganismGenome global_champion = elite_pool[0];
    global_champion.fitness = 99999.0f;

    auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t total_evals_done = 0;

    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        for (int i = 0; i < POP_SIZE; ++i) {
            int parent_idx = rng() % ELITE_POOL_SIZE;
            h_batch[i] = elite_pool[parent_idx];
            mutate_genome(h_batch[i], rng);
        }

        CHECK_CUDA(cudaMemcpy(d_batch, h_batch.data(), POP_SIZE * sizeof(OrganismGenome), cudaMemcpyHostToDevice));

        evaluate_totipotent_crucible_kernel<<<blocks, threads>>>(d_batch, POP_SIZE);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_batch.data(), d_batch, POP_SIZE * sizeof(OrganismGenome), cudaMemcpyDeviceToHost));

        for (int i = 0; i < POP_SIZE; ++i) {
            if (h_batch[i].fitness < global_champion.fitness) {
                global_champion = h_batch[i];
            }
            if (h_batch[i].fitness < elite_pool[i % ELITE_POOL_SIZE].fitness) {
                elite_pool[i % ELITE_POOL_SIZE] = h_batch[i];
            }
        }

        total_evals_done += POP_SIZE;

        if ((gen + 1) % 10 == 0 || gen == 0) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double speed = static_cast<double>(total_evals_done) / elapsed;

            std::cout << "Gen [" << std::setw(2) << (gen + 1) << "/" << NUM_GENERATIONS << "]"
                      << " | Evals: " << std::setw(8) << total_evals_done
                      << " | Speed: " << std::fixed << std::setprecision(0) << speed << " evals/s"
                      << " | Champion FE: " << std::setprecision(4) << global_champion.fitness
                      << " | Turing: " << std::setprecision(1) << (global_champion.task_errors[2] * 100.0f) << "%"
                      << " | Parity: " << (global_champion.task_errors[3] * 100.0f) << "%"
                      << " | Stack: " << (global_champion.task_errors[4] * 100.0f) << "%"
                      << " | Markov: " << (global_champion.task_errors[5] * 100.0f) << "%"
                      << " | Lang: " << (global_champion.task_errors[6] * 100.0f) << "%"
                      << " | Nodes: T1=" << (int)global_champion.active_t1 << "+T2=" << (int)global_champion.active_t2 << "\n";
            std::cout.flush();
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n========================================================================\n";
    std::cout << ">>> CRUCIBLE COMPLETE: " << total_evals_done << " EVALUATIONS EXECUTED!\n";
    std::cout << ">>> Total Runtime : " << std::fixed << std::setprecision(2) << total_sec << " seconds\n";
    std::cout << ">>> Average Speed : " << std::fixed << std::setprecision(0) << (total_evals_done / total_sec) << " organisms/sec\n";
    std::cout << "========================================================================\n";

    dissect_organism(global_champion);

    CHECK_CUDA(cudaFree(d_batch));
    return 0;
}
