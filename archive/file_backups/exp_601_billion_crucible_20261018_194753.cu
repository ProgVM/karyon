// ======================================================================================
// EXP-601 v2: The Billion-Scale Sovereign Crucible (1,000,000,000 Autonomous Evaluations)
// --------------------------------------------------------------------------------------
// Fix string literal lengths (+1 for null terminator).
// Scale: 524,288 agents/batch x 4 batches x 480 generations = 1,006,632,960 evaluations!
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

// Scale parameters
constexpr int BATCH_POP_SIZE   = 524288; // 524,288 agents per GPU batch
constexpr int NUM_BATCHES      = 4;      // 524,288 * 4 = 2,097,152 active agents per generation
constexpr int NUM_GENERATIONS  = 480;    // 2,097,152 * 480 = 1,006,632,960 (OVER 1 BILLION AGENTS EVALUATED!)

constexpr int MAX_NODES        = 32;     // Unconstrained DAG size up to 32 nodes
constexpr int NUM_REGISTERS    = 16;     // 16 dynamic state registers per agent
constexpr int NUM_INPUTS       = 6;      // Signal, aux_signal, context, carry, clock, time
constexpr int NUM_OUTPUTS      = 4;      // Main action, auxiliary, state write, memory address
constexpr int NUM_TASKS        = 10;     // 10 rigorous reality tasks

// Full Silicon Operator Palette
enum OpCode : uint8_t {
    OP_PASS = 0,
    OP_ADD,
    OP_SUB,
    OP_MUL_SHIFT,   // (a * b) >> 4
    OP_XOR,
    OP_AND,
    OP_OR,
    OP_NOT,
    OP_MIN,
    OP_MAX,
    OP_ABS,
    OP_SATURATE,    // clamp to [-128, 127]
    OP_SIGN,        // -1, 0, +1
    OP_ROTL,        // Bitwise rotation
    OP_STORE_REG,   // Write to registers[in2 % 16] = in1
    OP_LOAD_REG,    // Read from registers[in1 % 16]
    OP_ACCUMULATE,  // registers[in2 % 16] += in1
    NUM_OPS
};

struct GeneNode {
    uint8_t op;       // OpCode
    uint8_t in1;      // First operand source
    uint8_t in2;      // Second operand source
    int8_t  weight;   // Coupling parameter [-16 .. 16]
};

struct alignas(64) AgentGenome {
    GeneNode nodes[MAX_NODES];
    uint8_t  active_nodes;
    float    fitness;
    float    task_errors[NUM_TASKS];
};

// ======================================================================================
// CUDA DEVICE FUNCTIONS & EXECUTION ENGINE
// ======================================================================================
__device__ inline int16_t dev_clamp16(int32_t val, int32_t low, int32_t high) {
    return static_cast<int16_t>((val < low) ? low : ((val > high) ? high : val));
}

__device__ inline int16_t exec_sovereign_op(uint8_t op, int16_t a, int16_t b, int8_t w, int16_t* registers) {
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
        case OP_ABS:         res = (a < 0) ? -a : a; break;
        case OP_SATURATE:    res = (a < -128) ? -128 : ((a > 127) ? 127 : a); break;
        case OP_SIGN:        res = (a > 0) ? 1 : ((a < 0) ? -1 : 0); break;
        case OP_ROTL:        res = (static_cast<uint16_t>(a) << (b & 15)) | (static_cast<uint16_t>(a) >> (16 - (b & 15))); break;
        case OP_STORE_REG:   registers[static_cast<uint8_t>(b) % NUM_REGISTERS] = a; res = a; break;
        case OP_LOAD_REG:    res = registers[static_cast<uint8_t>(a) % NUM_REGISTERS]; break;
        case OP_ACCUMULATE:  {
            int reg_idx = static_cast<uint8_t>(b) % NUM_REGISTERS;
            registers[reg_idx] = dev_clamp16(static_cast<int32_t>(registers[reg_idx]) + a, -4096, 4095);
            res = registers[reg_idx];
            break;
        }
        default:             res = a; break;
    }
    int32_t modulated = (res * static_cast<int32_t>(w)) >> 4;
    return dev_clamp16(modulated, -4096, 4095);
}

__device__ inline void run_agent_cycle(const AgentGenome& agent, const int16_t* inputs, int16_t* outputs, int16_t* registers) {
    int16_t bus[NUM_INPUTS + MAX_NODES];

    for (int i = 0; i < NUM_INPUTS; ++i) {
        bus[i] = inputs[i];
    }

    for (int n = 0; n < agent.active_nodes; ++n) {
        const GeneNode& gn = agent.nodes[n];
        int16_t in1_val = bus[gn.in1 % (NUM_INPUTS + n)];
        int16_t in2_val = bus[gn.in2 % (NUM_INPUTS + n)];

        int16_t out_val = exec_sovereign_op(gn.op, in1_val, in2_val, gn.weight, registers);
        bus[NUM_INPUTS + n] = out_val;
    }

    for (int o = 0; o < NUM_OUTPUTS; ++o) {
        int src_node = agent.active_nodes - 1 - o;
        outputs[o] = (src_node >= 0) ? bus[NUM_INPUTS + src_node] : 0;
    }
}

// ======================================================================================
// 10-DOMAIN COMPREHENSIVE BATTERY EVALUATION KERNEL
// ======================================================================================
__global__ void evaluate_billion_crucible_kernel(AgentGenome* agents, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    AgentGenome& agent = agents[idx];
    int16_t registers[NUM_REGISTERS];

    #define CLEAR_REGS() for(int r=0; r<NUM_REGISTERS; ++r) registers[r] = 0;

    // TASK 1: Chaotic Lorenz Physics
    CLEAR_REGS();
    float err_lorenz = 0.0f;
    int16_t lx = 10, ly = 20, lz = 30;
    for (int t = 0; t < 32; ++t) {
        int16_t dx = (10 * (ly - lx)) >> 3;
        int16_t target_x = lx + dx;
        int16_t in[NUM_INPUTS] = {lx, ly, lz, static_cast<int16_t>(t), 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_cycle(agent, in, out, registers);
        err_lorenz += fabsf(static_cast<float>(out[0] - target_x));
        lx = target_x; ly += 5; lz += 2;
    }
    agent.task_errors[0] = err_lorenz / 32.0f;

    // TASK 2: Damped Forced Harmonic Oscillator
    CLEAR_REGS();
    float err_osc = 0.0f;
    int16_t state_x = 0, state_v = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t force = static_cast<int16_t>(16.0f * sinf(t * 0.25f));
        int16_t target_x = state_x + state_v;
        int16_t in[NUM_INPUTS] = {state_x, state_v, force, static_cast<int16_t>(t), 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_cycle(agent, in, out, registers);
        err_osc += fabsf(static_cast<float>(out[0] - target_x));
        state_v = (14 * state_v - 4 * state_x + force) >> 4;
        state_x = target_x;
    }
    agent.task_errors[1] = err_osc / 32.0f;

    // TASK 3: Turing Multi-Bit Ripple Carry Adder
    CLEAR_REGS();
    float err_carry = 0.0f;
    int16_t c_bit = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t a = (t * 11) & 1;
        int16_t b = (t * 17) & 1;
        int16_t tgt_sum = a ^ b ^ c_bit;
        int16_t tgt_c = (a & b) | (c_bit & (a ^ b));

        int16_t in[NUM_INPUTS] = {a, b, c_bit, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_cycle(agent, in, out, registers);
        if ((out[0] & 1) != tgt_sum) err_carry += 1.0f;
        if ((out[1] & 1) != tgt_c) err_carry += 1.0f;
        c_bit = tgt_c;
    }
    agent.task_errors[2] = err_carry / 64.0f;

    // TASK 4: Bitwise Dynamic Parity Accumulator
    CLEAR_REGS();
    float err_parity = 0.0f;
    int16_t par = 0;
    for (int t = 0; t < 32; ++t) {
        int16_t bit = (t * 7 + 3) & 1;
        par ^= bit;
        int16_t in[NUM_INPUTS] = {bit, static_cast<int16_t>(t), 0, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_cycle(agent, in, out, registers);
        if ((out[0] & 1) != par) err_parity += 1.0f;
    }
    agent.task_errors[3] = err_parity / 32.0f;

    // TASK 5: Fibonacci Multi-Step Recurrence Dynamics
    CLEAR_REGS();
    float err_fib = 0.0f;
    int16_t f0 = 1, f1 = 1;
    for (int t = 0; t < 16; ++t) {
        int16_t tgt_f = f1 + f0;
        int16_t in[NUM_INPUTS] = {f1, f0, static_cast<int16_t>(t), 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_cycle(agent, in, out, registers);
        err_fib += fabsf(static_cast<float>(out[0] - tgt_f));
        f0 = f1; f1 = tgt_f;
    }
    agent.task_errors[4] = err_fib / 16.0f;

    // TASK 6: Natural Language Word-Boundary Transitions
    CLEAR_REGS();
    float err_text = 0.0f;
    const char phrase[33] = "instruction_response_karyon_core";
    for (int t = 0; t < 31; ++t) {
        uint8_t cur = static_cast<uint8_t>(phrase[t]);
        uint8_t nxt = static_cast<uint8_t>(phrase[t+1]);

        int16_t in[NUM_INPUTS] = {static_cast<int16_t>(cur), static_cast<int16_t>(t), 0, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_cycle(agent, in, out, registers);
        if (static_cast<uint8_t>(out[0] & 0xFF) != nxt) err_text += 1.0f;
    }
    agent.task_errors[5] = err_text / 31.0f;

    // TASK 7: Linguistic Syllable Vowel-Consonant Rhythm
    CLEAR_REGS();
    float err_phono = 0.0f;
    const char rhythm[16] = "ba_da_ka_ma_ta_";
    for (int t = 0; t < 14; ++t) {
        int16_t in[NUM_INPUTS] = {static_cast<int16_t>(rhythm[t]), static_cast<int16_t>(t), 0, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_cycle(agent, in, out, registers);
        if (static_cast<uint8_t>(out[0] & 0xFF) != static_cast<uint8_t>(rhythm[t+1])) err_phono += 1.0f;
    }
    agent.task_errors[6] = err_phono / 14.0f;

    // TASK 8: Context-Free Grammar Bracket Balancing
    CLEAR_REGS();
    float err_bracket = 0.0f;
    int depth = 0;
    const char brackets[17] = "(())((()))()(())";
    for (int t = 0; t < 16; ++t) {
        char b = brackets[t];
        depth += (b == '(') ? 1 : -1;
        int16_t in[NUM_INPUTS] = {static_cast<int16_t>(b), static_cast<int16_t>(t), 0, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_cycle(agent, in, out, registers);
        if (out[0] != depth) err_bracket += 1.0f;
    }
    agent.task_errors[7] = err_bracket / 16.0f;

    // TASK 9: Periodic Sine-Wave Tracking
    CLEAR_REGS();
    float err_sine = 0.0f;
    for (int t = 0; t < 32; ++t) {
        int16_t tgt_sine = static_cast<int16_t>(32.0f * sinf(t * 0.3926f));
        int16_t in[NUM_INPUTS] = {static_cast<int16_t>(t), 0, 0, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_cycle(agent, in, out, registers);
        err_sine += fabsf(static_cast<float>(out[0] - tgt_sine));
    }
    agent.task_errors[8] = err_sine / 32.0f;

    // TASK 10: Synaptic Shock Resistance
    float err_shock = 0.0f;
    for (int t = 0; t < 16; ++t) {
        int16_t clean_sig = (t * 5);
        int16_t noise = (t & 1) ? 2 : -2;
        int16_t in[NUM_INPUTS] = {static_cast<int16_t>(clean_sig + noise), static_cast<int16_t>(t), 0, 0, 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_cycle(agent, in, out, registers);
        err_shock += fabsf(static_cast<float>(out[0] - clean_sig));
    }
    agent.task_errors[9] = err_shock / 16.0f;

    // Master Variational Free Energy
    float complexity_penalty = 0.003f * static_cast<float>(agent.active_nodes);
    float total_fe = agent.task_errors[0] * 0.05f + // Lorenz
                     agent.task_errors[1] * 0.05f + // Oscillator
                     agent.task_errors[2] * 2.00f + // Carry logic
                     agent.task_errors[3] * 1.50f + // Parity
                     agent.task_errors[4] * 0.02f + // Fibonacci
                     agent.task_errors[5] * 2.50f + // Language text
                     agent.task_errors[6] * 1.50f + // Phonotactics
                     agent.task_errors[7] * 1.50f + // Brackets / Stack
                     agent.task_errors[8] * 0.05f + // Sine
                     agent.task_errors[9] * 0.05f + // Shock resistance
                     complexity_penalty;

    agent.fitness = total_fe;
}

// ======================================================================================
// HOST SELECTION & MORPHOGENESIS ENGINE
// ======================================================================================
void init_genome(AgentGenome& agent, std::mt19937& rng) {
    std::uniform_int_distribution<int> op_dist(1, NUM_OPS - 1);
    std::uniform_int_distribution<int> w_dist(-16, 16);
    std::uniform_int_distribution<int> node_dist(6, 14);

    agent.active_nodes = node_dist(rng);
    agent.fitness = 99999.0f;
    for (int t = 0; t < NUM_TASKS; ++t) agent.task_errors[t] = 999.0f;

    for (int n = 0; n < MAX_NODES; ++n) {
        agent.nodes[n].op = static_cast<uint8_t>(op_dist(rng));
        agent.nodes[n].in1 = (NUM_INPUTS + n > 0) ? (rng() % (NUM_INPUTS + n)) : 0;
        agent.nodes[n].in2 = (NUM_INPUTS + n > 0) ? (rng() % (NUM_INPUTS + n)) : 0;
        agent.nodes[n].weight = static_cast<int8_t>(w_dist(rng));
    }
}

void mutate_genome(AgentGenome& child, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_int_distribution<int> op_dist(1, NUM_OPS - 1);

    if (prob(rng) < 0.22f && child.active_nodes < MAX_NODES) {
        int n = child.active_nodes;
        child.nodes[n].op = static_cast<uint8_t>(op_dist(rng));
        child.nodes[n].in1 = rng() % (NUM_INPUTS + n);
        child.nodes[n].in2 = rng() % (NUM_INPUTS + n);
        child.nodes[n].weight = static_cast<int8_t>(rng() % 33 - 16);
        child.active_nodes++;
    }

    if (prob(rng) < 0.12f && child.active_nodes > 4) {
        child.active_nodes--;
    }

    for (int n = 0; n < child.active_nodes; ++n) {
        if (prob(rng) < 0.18f) child.nodes[n].op = static_cast<uint8_t>(op_dist(rng));
        if (prob(rng) < 0.25f && NUM_INPUTS + n > 0) child.nodes[n].in1 = rng() % (NUM_INPUTS + n);
        if (prob(rng) < 0.25f && NUM_INPUTS + n > 0) child.nodes[n].in2 = rng() % (NUM_INPUTS + n);
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
        case OP_XOR: return "XOR";
        case OP_AND: return "AND";
        case OP_OR: return "OR";
        case OP_NOT: return "NOT";
        case OP_MIN: return "MIN";
        case OP_MAX: return "MAX";
        case OP_ABS: return "ABS";
        case OP_SATURATE: return "SATURATE";
        case OP_SIGN: return "SIGN";
        case OP_ROTL: return "ROTL";
        case OP_STORE_REG: return "STORE_REG";
        case OP_LOAD_REG: return "LOAD_REG";
        case OP_ACCUMULATE: return "ACCUMULATE";
        default: return "UNKNOWN";
    }
}

void dissect_billion_champion(const AgentGenome& champion) {
    std::cout << "\n========================================================================\n";
    std::cout << "=== 🔬 DEEP ENDOSCOPIC DISSECTION: THE BILLION-SCALE CHAMPION ATOM  ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Active Dynamic Nodes : " << (int)champion.active_nodes << " (Pruned from max 32)\n";
    std::cout << "Total Free Energy    : " << std::fixed << std::setprecision(5) << champion.fitness << "\n\n";

    std::cout << ">>> Comprehensive 10-Task Reality Battery Scorecard:\n";
    std::cout << "  1. Chaotic Lorenz Physics Tracking Error : " << champion.task_errors[0] << "\n";
    std::cout << "  2. Harmonic Oscillator Tracking Error    : " << champion.task_errors[1] << "\n";
    std::cout << "  3. Turing Ripple-Carry Logic Error Rate  : " << champion.task_errors[2] * 100.0f << " %\n";
    std::cout << "  4. Dynamic Parity Accumulator Error Rate : " << champion.task_errors[3] * 100.0f << " %\n";
    std::cout << "  5. Fibonacci Recurrent Tracking Error    : " << champion.task_errors[4] << "\n";
    std::cout << "  6. Natural Language Stream Error Rate    : " << champion.task_errors[5] * 100.0f << " %\n";
    std::cout << "  7. Phonotactic Syllable Error Rate       : " << champion.task_errors[6] * 100.0f << " %\n";
    std::cout << "  8. Stack Bracket Balancing Error Rate    : " << champion.task_errors[7] * 100.0f << " %\n";
    std::cout << "  9. Sine-Wave Periodic Tracking Error     : " << champion.task_errors[8] << "\n";
    std::cout << " 10. Synaptic Shock Noise Error            : " << champion.task_errors[9] << "\n\n";

    std::cout << ">>> Complete Synthesized Topological Circuit & Dynamic Equations:\n";
    for (int n = 0; n < champion.active_nodes; ++n) {
        const GeneNode& gn = champion.nodes[n];
        std::string s1 = (gn.in1 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in1) + "]") : ("node_" + std::to_string(gn.in1 - NUM_INPUTS));
        std::string s2 = (gn.in2 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in2) + "]") : ("node_" + std::to_string(gn.in2 - NUM_INPUTS));

        std::cout << "  node_" << std::setw(2) << n << " = " << opcode_name(gn.op) 
                  << "(" << s1 << ", " << s2 << ") * (" << (int)gn.weight << " / 16)";
        if (gn.op == OP_STORE_REG || gn.op == OP_LOAD_REG || gn.op == OP_ACCUMULATE) {
            std::cout << "  <== [VECTOR MEMORY ACCESS]";
        }
        std::cout << "\n";
    }

    std::cout << "\n>>> High-Speed Native C++20 Execution Cell for Karyon Production:\n";
    std::cout << "struct KaryonUniversalCell {\n";
    std::cout << "    int16_t registers[16] = {0};\n\n";
    std::cout << "    int16_t step(int16_t in0, int16_t in1, int16_t in2, int16_t in3, int16_t in4, int16_t in5) {\n";
    std::cout << "        int16_t in[6] = {in0, in1, in2, in3, in4, in5};\n";
    for (int n = 0; n < champion.active_nodes; ++n) {
        const GeneNode& gn = champion.nodes[n];
        std::string s1 = (gn.in1 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in1) + "]") : ("n" + std::to_string(gn.in1 - NUM_INPUTS));
        std::string s2 = (gn.in2 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in2) + "]") : ("n" + std::to_string(gn.in2 - NUM_INPUTS));
        std::cout << "        int16_t n" << n << " = (" << opcode_name(gn.op) << "(" << s1 << ", " << s2 << ") * " << (int)gn.weight << ") >> 4;\n";
    }
    std::cout << "        return n" << (champion.active_nodes - 1) << ";\n";
    std::cout << "    }\n";
    std::cout << "};\n";
    std::cout << "========================================================================\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-601: THE BILLION-SCALE SOVEREIGN CRUCIBLE (CUDA GENESIS)     ===\n";
    std::cout << "========================================================================\n";
    uint64_t total_evals_target = static_cast<uint64_t>(BATCH_POP_SIZE) * NUM_BATCHES * NUM_GENERATIONS;
    std::cout << "Target Evaluations       : " << total_evals_target << " (~1.006 BILLION AGENTS)\n";
    std::cout << "Active Batch Size on GPU : " << BATCH_POP_SIZE << " agents simultaneously\n";
    std::cout << "Reality Battery          : 10 Cross-Domain Tasks (Physics, Logic, Grammar, Speech)\n";
    std::cout << "Substrate Sovereignty    : Full Topological DAG + 16 Vector Registers\n\n";

    std::mt19937 rng(2026);

    constexpr int ELITE_POOL_SIZE = 1024;
    std::vector<AgentGenome> elite_pool(ELITE_POOL_SIZE);
    for (auto& a : elite_pool) init_genome(a, rng);

    std::vector<AgentGenome> h_batch(BATCH_POP_SIZE);

    AgentGenome* d_batch = nullptr;
    CHECK_CUDA(cudaMalloc(&d_batch, BATCH_POP_SIZE * sizeof(AgentGenome)));

    int threads = 256;
    int blocks  = (BATCH_POP_SIZE + threads - 1) / threads;

    AgentGenome global_champion = elite_pool[0];
    global_champion.fitness = 99999.0f;

    auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t total_evals_done = 0;

    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        for (int b = 0; b < NUM_BATCHES; ++b) {
            for (int i = 0; i < BATCH_POP_SIZE; ++i) {
                int parent_idx = rng() % ELITE_POOL_SIZE;
                h_batch[i] = elite_pool[parent_idx];
                mutate_genome(h_batch[i], rng);
            }

            CHECK_CUDA(cudaMemcpy(d_batch, h_batch.data(), BATCH_POP_SIZE * sizeof(AgentGenome), cudaMemcpyHostToDevice));

            evaluate_billion_crucible_kernel<<<blocks, threads>>>(d_batch, BATCH_POP_SIZE);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(h_batch.data(), d_batch, BATCH_POP_SIZE * sizeof(AgentGenome), cudaMemcpyDeviceToHost));

            for (int i = 0; i < BATCH_POP_SIZE; ++i) {
                if (h_batch[i].fitness < global_champion.fitness) {
                    global_champion = h_batch[i];
                }
                if (h_batch[i].fitness < elite_pool[i % ELITE_POOL_SIZE].fitness) {
                    elite_pool[i % ELITE_POOL_SIZE] = h_batch[i];
                }
            }

            total_evals_done += BATCH_POP_SIZE;
        }

        if ((gen + 1) % 40 == 0 || gen == 0) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double speed = static_cast<double>(total_evals_done) / elapsed;

            std::cout << "Gen [" << std::setw(3) << (gen + 1) << "/" << NUM_GENERATIONS << "]"
                      << " | Evals: " << std::setw(10) << total_evals_done
                      << " | Speed: " << std::fixed << std::setprecision(0) << speed << " evals/s"
                      << " | Champion FE: " << std::setprecision(4) << global_champion.fitness
                      << " | Turing: " << std::setprecision(1) << (global_champion.task_errors[2] * 100.0f) << "%"
                      << " | Stack: " << (global_champion.task_errors[7] * 100.0f) << "%"
                      << " | Lang: " << (global_champion.task_errors[5] * 100.0f) << "%"
                      << " | Nodes: " << (int)global_champion.active_nodes << "\n";
            std::cout.flush();
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n========================================================================\n";
    std::cout << ">>> CRUCIBLE COMPLETE: " << total_evals_done << " EVALUATIONS EXECUTED!\n";
    std::cout << ">>> Total Runtime : " << std::fixed << std::setprecision(2) << total_sec << " seconds\n";
    std::cout << ">>> Average Speed : " << std::fixed << std::setprecision(0) << (total_evals_done / total_sec) << " agent-tasks/sec\n";
    std::cout << "========================================================================\n";

    dissect_billion_champion(global_champion);

    CHECK_CUDA(cudaFree(d_batch));
    return 0;
}
