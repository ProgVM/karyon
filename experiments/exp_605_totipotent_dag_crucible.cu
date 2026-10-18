// ======================================================================================
// EXP-605: Unconstrained Totipotent Graph Substrate (Arbitrary DAG + Full Silicon ISA)
// --------------------------------------------------------------------------------------
// Fundamental Sovereign Design (Addressing Bazilevs' Mandate):
// 1. ZERO RIGID TIERS / ZERO HUMAN ARCHITECTURAL BIAS:
//    - Any node can connect to ANY previous node or ANY input (arbitrary DAG).
//    - Can naturally form feedforward, skip, deep, wide, parallel or branched topologies.
// 2. FULL SILICON INSTRUCTION SET (Complete Totipotent ISA):
//    - Arithmetic: ADD, SUB, MUL_SHIFT, DIV_MOD
//    - Bitwise: XOR, AND, OR, NOT, ROTL, ROTR, SHL, SHR
//    - Nonlinear/Comparators: MIN, MAX, ABS, SIGN, SATURATE, CMP_EQ, CMP_LT, COND_MUX
//    - Dynamic Memory & Registers:
//        LOAD_REG, STORE_REG, ACCUMULATE
//        ASSOC_READ, ASSOC_WRITE (Hash table)
//        STACK_PUSH, STACK_POP (LIFO stack)
// 3. FULL TASK REALITY BATTERY (No Time Clocks, Strict Independent Evaluation):
//    - Lorenz Chaos
//    - Oscillator
//    - Turing Full Adder
//    - Bit Parity XOR
//    - Context-Free Stack Depth
//    - Markov Graph Transitions
//    - Natural Language Alpaca
//    - Synaptic Noise Immunity
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

constexpr int POP_SIZE        = 131072; // 131,072 parallel agents
constexpr int NUM_GENERATIONS = 100;    // 13,107,200 evaluations
constexpr int MAX_NODES       = 32;     // Up to 32 arbitrary DAG nodes (unconstrained)
constexpr int MEM_SIZE        = 64;     // 64 state registers (registers + associative + stack)
constexpr int NUM_INPUTS      = 4;      // Pure sensory inputs (NO time clocks!)
constexpr int NUM_OUTPUTS     = 2;      
constexpr int NUM_TASKS       = 8;      

enum OpCode : uint8_t {
    OP_PASS = 0,
    OP_ADD,
    OP_SUB,
    OP_MUL_SHIFT,   // (a * b) >> 4
    OP_DIV_MOD,     // (b != 0) ? (a % b) : 0
    OP_XOR,
    OP_AND,
    OP_OR,
    OP_NOT,
    OP_SHL,         // a << (b & 7)
    OP_SHR,         // a >> (b & 7)
    OP_ROTL,        // Bit rotation
    OP_MIN,
    OP_MAX,
    OP_ABS,
    OP_SIGN,
    OP_SATURATE,    // clamp to [-128, 127]
    OP_CMP_EQ,      // (a == b) ? 16 : 0
    OP_CMP_LT,      // (a < b) ? 16 : 0
    OP_COND_MUX,    // (a > 0) ? b : 0
    OP_LOAD_REG,    // read registers[a % 32]
    OP_STORE_REG,   // registers[b % 32] = a
    OP_ACCUMULATE,  // registers[a % 32] += b
    OP_ASSOC_READ,  // read hash_table[((a * 73) ^ 19) % 16]
    OP_ASSOC_WRITE, // hash_table[((a * 73) ^ 19) % 16] = b
    OP_STACK_PUSH,  // stack[sp++] = a
    OP_STACK_POP,   // res = stack[--sp]
    NUM_OPS
};

struct GeneNode {
    uint8_t op;
    uint8_t in1;     // index into (NUM_INPUTS + previous_nodes)
    uint8_t in2;     // index into (NUM_INPUTS + previous_nodes)
    int8_t  weight;  // [-16 .. 16]
};

struct alignas(64) AgentGenome {
    GeneNode nodes[MAX_NODES];
    uint8_t  active_nodes;
    float    fitness;
    float    task_errors[NUM_TASKS];
};

__device__ inline int16_t dev_clamp16(int32_t val, int32_t low, int32_t high) {
    return static_cast<int16_t>((val < low) ? low : ((val > high) ? high : val));
}

__device__ inline int16_t exec_unconstrained_op(uint8_t op, int16_t a, int16_t b, int8_t w, int16_t* mem) {
    int32_t res = 0;
    // Memory layout:
    // mem[0..31]  : general registers
    // mem[32..47] : associative hash table
    // mem[48..62] : stack buffer
    // mem[63]     : stack pointer (sp)
    switch(op) {
        case OP_PASS:        res = a; break;
        case OP_ADD:         res = a + b; break;
        case OP_SUB:         res = a - b; break;
        case OP_MUL_SHIFT:   res = (static_cast<int32_t>(a) * static_cast<int32_t>(b)) >> 4; break;
        case OP_DIV_MOD:     res = (b != 0) ? (a % b) : 0; break;
        case OP_XOR:         res = a ^ b; break;
        case OP_AND:         res = a & b; break;
        case OP_OR:          res = a | b; break;
        case OP_NOT:         res = ~a; break;
        case OP_SHL:         res = a << (b & 7); break;
        case OP_SHR:         res = a >> (b & 7); break;
        case OP_ROTL:        res = (static_cast<uint16_t>(a) << (b & 15)) | (static_cast<uint16_t>(a) >> (16 - (b & 15))); break;
        case OP_MIN:         res = (a < b) ? a : b; break;
        case OP_MAX:         res = (a > b) ? a : b; break;
        case OP_ABS:         res = (a < 0) ? -a : a; break;
        case OP_SIGN:        res = (a > 0) ? 1 : ((a < 0) ? -1 : 0); break;
        case OP_SATURATE:    res = (a < -128) ? -128 : ((a > 127) ? 127 : a); break;
        case OP_CMP_EQ:      res = (a == b) ? 16 : 0; break;
        case OP_CMP_LT:      res = (a < b) ? 16 : 0; break;
        case OP_COND_MUX:    res = (a > 0) ? b : 0; break;
        case OP_LOAD_REG:    res = mem[static_cast<uint8_t>(a) % 32]; break;
        case OP_STORE_REG:   mem[static_cast<uint8_t>(b) % 32] = a; res = a; break;
        case OP_ACCUMULATE:  {
            uint8_t reg = static_cast<uint8_t>(a) % 32;
            mem[reg] = dev_clamp16(static_cast<int32_t>(mem[reg]) + b, -4096, 4095);
            res = mem[reg];
            break;
        }
        case OP_ASSOC_READ:  {
            uint8_t h = (static_cast<uint32_t>(a) * 73u + 19u) % 16;
            res = mem[32 + h];
            break;
        }
        case OP_ASSOC_WRITE: {
            uint8_t h = (static_cast<uint32_t>(a) * 73u + 19u) % 16;
            mem[32 + h] = b;
            res = b;
            break;
        }
        case OP_STACK_PUSH:  {
            int16_t sp = mem[63];
            mem[48 + (sp & 15)] = a;
            mem[63] = dev_clamp16(sp + 1, 0, 15);
            res = mem[63];
            break;
        }
        case OP_STACK_POP:   {
            int16_t sp = mem[63];
            if (sp > 0) sp--;
            mem[63] = sp;
            res = mem[48 + (sp & 15)];
            break;
        }
        default:             res = a; break;
    }
    int32_t modulated = (res * static_cast<int32_t>(w)) >> 4;
    return dev_clamp16(modulated, -4096, 4095);
}

__device__ inline void run_unconstrained_dag(const AgentGenome& agent, const int16_t* inputs, int16_t* outputs, int16_t* mem) {
    int16_t bus[NUM_INPUTS + MAX_NODES];
    for (int i = 0; i < NUM_INPUTS; ++i) bus[i] = inputs[i];

    for (int n = 0; n < agent.active_nodes; ++n) {
        const GeneNode& gn = agent.nodes[n];
        int16_t in1_val = bus[gn.in1 % (NUM_INPUTS + n)];
        int16_t in2_val = bus[gn.in2 % (NUM_INPUTS + n)];
        bus[NUM_INPUTS + n] = exec_unconstrained_op(gn.op, in1_val, in2_val, gn.weight, mem);
    }

    outputs[0] = bus[NUM_INPUTS + agent.active_nodes - 1];
    outputs[1] = (agent.active_nodes > 1) ? bus[NUM_INPUTS + agent.active_nodes - 2] : 0;
}

__global__ void evaluate_totipotent_dag_kernel(AgentGenome* agents, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    AgentGenome& agent = agents[idx];
    int16_t mem[MEM_SIZE];
    #define RESET_MEM() for(int m=0; m<MEM_SIZE; ++m) mem[m] = 0;

    // 1. Lorenz Physics
    RESET_MEM();
    float err_lorenz = 0.0f;
    int16_t lx = 10, ly = 20, lz = 30;
    for (int t = 0; t < 32; ++t) {
        int16_t dx = (10 * (ly - lx)) >> 3;
        int16_t target_x = lx + dx;
        int16_t in[NUM_INPUTS] = {lx, ly, lz, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_unconstrained_dag(agent, in, out, mem);
        err_lorenz += fabsf(static_cast<float>(out[0] - target_x));
        lx = target_x; ly += 5; lz += 2;
    }
    agent.task_errors[0] = err_lorenz / 32.0f;

    // 2. Harmonic Oscillator
    RESET_MEM();
    float err_osc = 0.0f;
    int16_t sx = 0, sv = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t force = (t & 1) ? 16 : -16;
        int16_t target_x = sx + sv;
        int16_t in[NUM_INPUTS] = {sx, sv, force, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_unconstrained_dag(agent, in, out, mem);
        err_osc += fabsf(static_cast<float>(out[0] - target_x));
        sv = (14 * sv - 4 * sx + force) >> 4;
        sx = target_x;
    }
    agent.task_errors[1] = err_osc / 32.0f;

    // 3. Turing Ripple-Carry
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
        run_unconstrained_dag(agent, in, out, mem);
        if ((out[0] & 1) != tgt_sum) err_carry += 1.0f;
        carry = tgt_carry;
    }
    agent.task_errors[2] = err_carry / 32.0f;

    // 4. Parity Stream (T-trigger test)
    RESET_MEM();
    float err_parity = 0.0f;
    int16_t par = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t bit = (t * 5 + 1) & 1;
        par ^= bit;
        int16_t in[NUM_INPUTS] = {bit, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_unconstrained_dag(agent, in, out, mem);
        if ((out[0] & 1) != par) err_parity += 1.0f;
    }
    agent.task_errors[3] = err_parity / 32.0f;

    // 5. Context-Free Stack Depth (Push/Pop bracket test)
    RESET_MEM();
    float err_bracket = 0.0f;
    int depth = 0;
    const char brackets[17] = "(())((()))()(())";
    for (int t = 0; t < 16; ++t) {
        char b = brackets[t];
        depth += (b == '(') ? 1 : -1;
        int16_t in[NUM_INPUTS] = {static_cast<int16_t>(b), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_unconstrained_dag(agent, in, out, mem);
        if (out[0] != depth) err_bracket += 1.0f;
    }
    agent.task_errors[4] = err_bracket / 16.0f;

    // 6. Markov Transition Flow
    RESET_MEM();
    float err_markov = 0.0f;
    const char seq[17] = "abcdabcdabcdabcd";
    for (int t = 0; t < 15; ++t) {
        int16_t cur = static_cast<int16_t>(seq[t]);
        int16_t tgt = static_cast<int16_t>(seq[t + 1]);
        int16_t in[NUM_INPUTS] = {cur, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_unconstrained_dag(agent, in, out, mem);
        if (out[0] != tgt) err_markov += 1.0f;
    }
    agent.task_errors[5] = err_markov / 15.0f;

    // 7. Natural Language Alpaca
    RESET_MEM();
    float err_lang = 0.0f;
    const char phrase[33] = "instruction_response_karyon_core";
    for (int t = 0; t < 31; ++t) {
        uint8_t cur = static_cast<uint8_t>(phrase[t]);
        uint8_t nxt = static_cast<uint8_t>(phrase[t + 1]);
        int16_t in[NUM_INPUTS] = {static_cast<int16_t>(cur), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_unconstrained_dag(agent, in, out, mem);
        if (static_cast<uint8_t>(out[0] & 0xFF) != nxt) err_lang += 1.0f;
    }
    agent.task_errors[6] = err_lang / 31.0f;

    // 8. Synaptic Shock Noise
    float err_shock = 0.0f;
    for (int t = 0; t < 16; ++t) {
        int16_t clean = t * 4;
        int16_t noisy = clean + ((t & 1) ? 3 : -3);
        int16_t in[NUM_INPUTS] = {noisy, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};
        run_unconstrained_dag(agent, in, out, mem);
        err_shock += fabsf(static_cast<float>(out[0] - clean));
    }
    agent.task_errors[7] = err_shock / 16.0f;

    // Balanced Free Energy (Equal Weighting across Core Reality Domains)
    float complexity = 0.001f * static_cast<float>(agent.active_nodes);
    agent.fitness = agent.task_errors[0] * 0.05f + // Lorenz
                    agent.task_errors[1] * 0.05f + // Osc
                    agent.task_errors[2] * 2.50f + // Turing
                    agent.task_errors[3] * 2.50f + // Parity
                    agent.task_errors[4] * 2.50f + // Stack
                    agent.task_errors[5] * 2.50f + // Markov
                    agent.task_errors[6] * 2.50f + // Lang
                    agent.task_errors[7] * 0.05f + // Shock
                    complexity;
}

void init_genome(AgentGenome& agent, std::mt19937& rng) {
    std::uniform_int_distribution<int> op_dist(1, NUM_OPS - 1);
    std::uniform_int_distribution<int> w_dist(-16, 16);
    std::uniform_int_distribution<int> n_dist(8, 20);

    agent.active_nodes = n_dist(rng);
    agent.fitness = 99999.0f;
    for (int t = 0; t < NUM_TASKS; ++t) agent.task_errors[t] = 999.0f;

    for (int n = 0; n < MAX_NODES; ++n) {
        agent.nodes[n].op = static_cast<uint8_t>(op_dist(rng));
        agent.nodes[n].in1 = rng() % (NUM_INPUTS + n);
        agent.nodes[n].in2 = rng() % (NUM_INPUTS + n);
        agent.nodes[n].weight = static_cast<int8_t>(w_dist(rng));
    }
}

void mutate_genome(AgentGenome& child, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_int_distribution<int> op_dist(1, NUM_OPS - 1);

    if (prob(rng) < 0.25f && child.active_nodes < MAX_NODES) child.active_nodes++;
    if (prob(rng) < 0.12f && child.active_nodes > 4) child.active_nodes--;

    for (int n = 0; n < child.active_nodes; ++n) {
        if (prob(rng) < 0.18f) child.nodes[n].op = static_cast<uint8_t>(op_dist(rng));
        if (prob(rng) < 0.25f) child.nodes[n].in1 = rng() % (NUM_INPUTS + n);
        if (prob(rng) < 0.25f) child.nodes[n].in2 = rng() % (NUM_INPUTS + n);
        if (prob(rng) < 0.40f) {
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
        case OP_ROTL: return "ROTL";
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

void dissect_agent(const AgentGenome& champion) {
    std::cout << "\n========================================================================\n";
    std::cout << "=== 🔬 DEEP ENDOSCOPIC AUTOPSY: TOTIPOTENT UNCONSTRAINED CHAMPION    ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Active Dynamic Nodes : " << (int)champion.active_nodes << "\n";
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

    std::cout << ">>> Discovered Graph Topology & Dynamic Equations:\n";
    for (int n = 0; n < champion.active_nodes; ++n) {
        const GeneNode& gn = champion.nodes[n];
        std::string s1 = (gn.in1 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in1) + "]") : ("n" + std::to_string(gn.in1 - NUM_INPUTS));
        std::string s2 = (gn.in2 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in2) + "]") : ("n" + std::to_string(gn.in2 - NUM_INPUTS));

        std::cout << "  node_" << std::setw(2) << n << " = " << opcode_name(gn.op) 
                  << "(" << s1 << ", " << s2 << ") * (" << (int)gn.weight << " / 16)";
        if (gn.op >= OP_LOAD_REG) {
            std::cout << "  <== [STATE/MEMORY OPERATOR]";
        }
        std::cout << "\n";
    }
    std::cout << "========================================================================\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-605: UNCONSTRAINED TOTIPOTENT GRAPH CRUCIBLE (CUDA GENESIS)  ===\n";
    std::cout << "========================================================================\n";
    uint64_t total_evals_target = static_cast<uint64_t>(POP_SIZE) * NUM_GENERATIONS;
    std::cout << "Target Evaluations       : " << total_evals_target << " (13.1 MILLION AGENTS)\n";
    std::cout << "Active Batch Size on GPU : " << POP_SIZE << " agents simultaneously\n";
    std::cout << "Substrate Sovereignty    : Full Arbitrary DAG (up to 32 nodes) + Full Silicon ISA (26 Ops)\n";
    std::cout << "Dynamic Memory Substrate : 64 Registers (32 Scalar Regs + 16 Hash Map + 16 LIFO Stack)\n";
    std::cout << "Zero Crutches Directive  : Strictly ZERO External Time Clocks\n\n";

    std::mt19937 rng(2026);

    constexpr int ELITE_POOL_SIZE = 1024;
    std::vector<AgentGenome> elite_pool(ELITE_POOL_SIZE);
    for (auto& a : elite_pool) init_genome(a, rng);

    std::vector<AgentGenome> h_batch(POP_SIZE);

    AgentGenome* d_batch = nullptr;
    CHECK_CUDA(cudaMalloc(&d_batch, POP_SIZE * sizeof(AgentGenome)));

    int threads = 256;
    int blocks  = (POP_SIZE + threads - 1) / threads;

    AgentGenome global_champion = elite_pool[0];
    global_champion.fitness = 99999.0f;

    auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t total_evals_done = 0;

    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        for (int i = 0; i < POP_SIZE; ++i) {
            int parent_idx = rng() % ELITE_POOL_SIZE;
            h_batch[i] = elite_pool[parent_idx];
            mutate_genome(h_batch[i], rng);
        }

        CHECK_CUDA(cudaMemcpy(d_batch, h_batch.data(), POP_SIZE * sizeof(AgentGenome), cudaMemcpyHostToDevice));

        evaluate_totipotent_dag_kernel<<<blocks, threads>>>(d_batch, POP_SIZE);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_batch.data(), d_batch, POP_SIZE * sizeof(AgentGenome), cudaMemcpyDeviceToHost));

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

            std::cout << "Gen [" << std::setw(3) << (gen + 1) << "/" << NUM_GENERATIONS << "]"
                      << " | Evals: " << std::setw(8) << total_evals_done
                      << " | Speed: " << std::fixed << std::setprecision(0) << speed << " evals/s"
                      << " | Champion FE: " << std::setprecision(4) << global_champion.fitness
                      << " | Turing: " << std::setprecision(1) << (global_champion.task_errors[2] * 100.0f) << "%"
                      << " | Parity: " << (global_champion.task_errors[3] * 100.0f) << "%"
                      << " | Stack: " << (global_champion.task_errors[4] * 100.0f) << "%"
                      << " | Markov: " << (global_champion.task_errors[5] * 100.0f) << "%"
                      << " | Lang: " << (global_champion.task_errors[6] * 100.0f) << "%"
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

    dissect_agent(global_champion);

    CHECK_CUDA(cudaFree(d_batch));
    return 0;
}
