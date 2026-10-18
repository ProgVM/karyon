// ======================================================================================
// EXP-600: The Sovereign Crucible (CUDA Multi-Agent Genesis Engine)
// --------------------------------------------------------------------------------------
// Architecture Overview:
// - Simulates POPULATION_SIZE autonomous micro-agents in parallel on CUDA.
// - Each agent possesses a programmable DAG genome with dynamic operators & registers:
//     * Arithmetic & Bitwise: ADD, SUB, MUL_SHIFT, XOR, AND, OR, SHL, SHR
//     * Nonlinear: SATURATE, RELU, ABS, MIN, MAX
//     * Memory / Delay: DELAY_Z1 (State registers)
// - Battery of 3 Cross-Domain Challenges:
//     1. Continuous Physical Dynamics: Nonlinear Damped Duffing/Harmonic Tracking
//     2. Algorithmic Discrete Logic: Multi-Bit Ripple Carry / Parity Turing Task
//     3. Raw Byte Language / Sequence Prediction: Multi-step linguistic transitions
// - Natural Selection via Variational Free Energy:
//     F = Error_Task1 + Error_Task2 + Error_Task3 + beta * NodeCount
// - Endoscopic Extraction:
//     The champion genome is isolated, decompiled into human-readable mathematical
//     equations and C++ code, and analyzed under microscope.
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

constexpr int POP_SIZE        = 65536; // 65,536 parallel evolving agents
constexpr int MAX_NODES       = 16;    // Up to 16 dynamic compute nodes per agent
constexpr int NUM_INPUTS      = 4;     // Task inputs (current signal, memory, context, time)
constexpr int NUM_OUTPUTS     = 2;     // Action / prediction & auxiliary state
constexpr int NUM_GENERATIONS = 100;   // Evolutionary epochs

// Primitive Opcodes
enum OpCode : uint8_t {
    OP_PASS = 0,
    OP_ADD,
    OP_SUB,
    OP_MUL_SHIFT,  // (a * b) >> 4
    OP_XOR,
    OP_AND,
    OP_OR,
    OP_MIN,
    OP_MAX,
    OP_ABS,
    OP_SATURATE,   // clamp between -128 and 127
    OP_DELAY,      // state memory z^-1
    NUM_OPS
};

struct GeneNode {
    uint8_t op;       // OpCode
    uint8_t in1;      // Source index 1 (Input or previous Node)
    uint8_t in2;      // Source index 2
    int8_t  weight;   // Coupling weight / parameter
    int16_t state;    // Persistent memory state register
};

struct alignas(64) AgentGenome {
    GeneNode nodes[MAX_NODES];
    uint8_t  active_nodes;
    float    fitness;
    float    err_physics;
    float    err_logic;
    float    err_language;
};

// ======================================================================================
// CUDA KERNEL: Autonomous Cross-Domain Execution & Evaluation
// ======================================================================================
__device__ inline int16_t exec_op(uint8_t op, int16_t a, int16_t b, int8_t w, int16_t& state) {
    int32_t res = 0;
    switch(op) {
        case OP_PASS:      res = a; break;
        case OP_ADD:       res = a + b; break;
        case OP_SUB:       res = a - b; break;
        case OP_MUL_SHIFT: res = (static_cast<int32_t>(a) * static_cast<int32_t>(b)) >> 4; break;
        case OP_XOR:       res = a ^ b; break;
        case OP_AND:       res = a & b; break;
        case OP_OR:        res = a | b; break;
        case OP_MIN:       res = (a < b) ? a : b; break;
        case OP_MAX:       res = (a > b) ? a : b; break;
        case OP_ABS:       res = (a < 0) ? -a : a; break;
        case OP_SATURATE:  res = (a < -128) ? -128 : ((a > 127) ? 127 : a); break;
        case OP_DELAY:     res = state; state = a; break;
        default:           res = a; break;
    }
    // Modulate with weight: (res * w) >> 4
    int32_t modulated = (res * static_cast<int32_t>(w)) >> 4;
    return static_cast<int16_t>((modulated < -4096) ? -4096 : ((modulated > 4095) ? 4095 : modulated));
}

__device__ inline void run_agent_step(const AgentGenome& agent, const int16_t* inputs, int16_t* outputs, int16_t* registers) {
    int16_t node_outputs[NUM_INPUTS + MAX_NODES];

    // Load inputs
    for (int i = 0; i < NUM_INPUTS; ++i) {
        node_outputs[i] = inputs[i];
    }

    // Execute active nodes in topological DAG order
    for (int n = 0; n < agent.active_nodes; ++n) {
        const GeneNode& gn = agent.nodes[n];
        int16_t in1_val = node_outputs[gn.in1 % (NUM_INPUTS + n)];
        int16_t in2_val = node_outputs[gn.in2 % (NUM_INPUTS + n)];

        int16_t out_val = exec_op(gn.op, in1_val, in2_val, gn.weight, registers[n]);
        node_outputs[NUM_INPUTS + n] = out_val;
    }

    // Readout outputs from last nodes
    outputs[0] = node_outputs[NUM_INPUTS + agent.active_nodes - 1];
    outputs[1] = (agent.active_nodes > 1) ? node_outputs[NUM_INPUTS + agent.active_nodes - 2] : 0;
}

__global__ void evaluate_crucible_battery_kernel(AgentGenome* agents, int pop_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    AgentGenome& agent = agents[idx];
    int16_t registers[MAX_NODES] = {0};

    // ----------------------------------------------------------------------------------
    // TEST 1: Continuous Nonlinear Physical Dynamics (Damped Nonlinear Oscillator)
    // Target: x_{t+1} = 0.95 * x_t - 0.20 * x_{t-1} + 0.10 * sin(t)
    // ----------------------------------------------------------------------------------
    float err_phys = 0.0f;
    int16_t state_phys = 0;
    int16_t prev_phys = 0;

    for (int t = 0; t < 64; ++t) {
        int16_t force = static_cast<int16_t>(16.0f * sinf(t * 0.2f));
        int16_t target = static_cast<int16_t>((15 * state_phys - 3 * prev_phys + force * 2) >> 4);

        int16_t in[NUM_INPUTS] = {state_phys, prev_phys, force, static_cast<int16_t>(t)};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_step(agent, in, out, registers);

        int16_t pred = out[0];
        int diff = pred - target;
        err_phys += fabsf(static_cast<float>(diff));

        prev_phys = state_phys;
        state_phys = target;
    }
    err_phys /= 64.0f;

    // Reset registers for next domain
    for (int n = 0; n < MAX_NODES; ++n) registers[n] = 0;

    // ----------------------------------------------------------------------------------
    // TEST 2: Turing-Complete Algorithmic Logic (Ripple-Carry Adder / Parity Accumulator)
    // Target: sum = a ^ b ^ carry, carry = (a & b) | (carry & (a ^ b))
    // ----------------------------------------------------------------------------------
    float err_log = 0.0f;
    int16_t carry = 0;

    for (int t = 0; t < 64; ++t) {
        int16_t a_bit = (t * 7) & 1;
        int16_t b_bit = ((t * 13) >> 1) & 1;

        int16_t target_sum = a_bit ^ b_bit ^ carry;
        int16_t target_carry = (a_bit & b_bit) | (carry & (a_bit ^ b_bit));

        int16_t in[NUM_INPUTS] = {a_bit, b_bit, carry, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_step(agent, in, out, registers);

        int16_t pred_sum = out[0] & 1;
        int16_t pred_carry = out[1] & 1;

        if (pred_sum != target_sum) err_log += 1.0f;
        if (pred_carry != target_carry) err_log += 1.0f;

        carry = target_carry;
    }
    err_log /= 128.0f;

    // Reset registers for next domain
    for (int n = 0; n < MAX_NODES; ++n) registers[n] = 0;

    // ----------------------------------------------------------------------------------
    // TEST 3: Multi-Scale Raw Byte Language / Sequential Hierarchy
    // Sequence pattern: Repeating phrase structure with vowel-consonant transitions
    // ----------------------------------------------------------------------------------
    float err_lang = 0.0f;
    const uint8_t phrase[16] = {'k', 'a', 'r', 'y', 'o', 'n', '_', 's', 'o', 'u', 'l', '_', 'c', 'o', 'r', 'e'};
    
    for (int t = 0; t < 64; ++t) {
        uint8_t in_byte = phrase[t % 16];
        uint8_t tgt_byte = phrase[(t + 1) % 16];

        int16_t in[NUM_INPUTS] = {static_cast<int16_t>(in_byte), static_cast<int16_t>(t % 16), 0, 0};
        int16_t out[NUM_OUTPUTS] = {0};

        run_agent_step(agent, in, out, registers);

        uint8_t pred_byte = static_cast<uint8_t>(out[0] & 0xFF);
        if (pred_byte != tgt_byte) err_lang += 1.0f;
    }
    err_lang /= 64.0f;

    // ----------------------------------------------------------------------------------
    // Variational Free Energy = Accuracy Error + Occam's Complexity Penalty
    // ----------------------------------------------------------------------------------
    float complexity_penalty = 0.015f * static_cast<float>(agent.active_nodes);
    agent.err_physics  = err_phys;
    agent.err_logic    = err_log;
    agent.err_language = err_lang;
    agent.fitness      = err_phys * 0.02f + err_log * 2.0f + err_lang * 2.0f + complexity_penalty;
}

// ======================================================================================
// HOST SELECTION, BREEDING & MORPHOGENESIS ENGINE
// ======================================================================================
void init_population(std::vector<AgentGenome>& pop) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> op_dist(1, NUM_OPS - 1);
    std::uniform_int_distribution<int> w_dist(-16, 16);
    std::uniform_int_distribution<int> node_dist(3, 6);

    for (auto& agent : pop) {
        agent.active_nodes = node_dist(rng);
        agent.fitness = 9999.0f;
        for (int n = 0; n < MAX_NODES; ++n) {
            agent.nodes[n].op = static_cast<uint8_t>(op_dist(rng));
            agent.nodes[n].in1 = (NUM_INPUTS + n > 0) ? (rng() % (NUM_INPUTS + n)) : 0;
            agent.nodes[n].in2 = (NUM_INPUTS + n > 0) ? (rng() % (NUM_INPUTS + n)) : 0;
            agent.nodes[n].weight = static_cast<int8_t>(w_dist(rng));
            agent.nodes[n].state = 0;
        }
    }
}

void mutate_agent(AgentGenome& child, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_int_distribution<int> op_dist(1, NUM_OPS - 1);
    std::uniform_int_distribution<int> w_dist(-16, 16);

    // Sprouting: Add a node if capacity permits
    if (prob(rng) < 0.15f && child.active_nodes < MAX_NODES) {
        int n = child.active_nodes;
        child.nodes[n].op = static_cast<uint8_t>(op_dist(rng));
        child.nodes[n].in1 = rng() % (NUM_INPUTS + n);
        child.nodes[n].in2 = rng() % (NUM_INPUTS + n);
        child.nodes[n].weight = static_cast<int8_t>(w_dist(rng));
        child.nodes[n].state = 0;
        child.active_nodes++;
    }

    // Pruning: Remove a node if minimum preserved
    if (prob(rng) < 0.10f && child.active_nodes > 2) {
        child.active_nodes--;
    }

    // Dynamic Operator / Rewiring Mutation
    for (int n = 0; n < child.active_nodes; ++n) {
        if (prob(rng) < 0.15f) child.nodes[n].op = static_cast<uint8_t>(op_dist(rng));
        if (prob(rng) < 0.20f && NUM_INPUTS + n > 0) child.nodes[n].in1 = rng() % (NUM_INPUTS + n);
        if (prob(rng) < 0.20f && NUM_INPUTS + n > 0) child.nodes[n].in2 = rng() % (NUM_INPUTS + n);
        if (prob(rng) < 0.30f) {
            int nw = child.nodes[n].weight + (rng() % 5 - 2);
            child.nodes[n].weight = static_cast<int8_t>(std::clamp(nw, -16, 16));
        }
    }
}

const char* op_to_str(uint8_t op) {
    switch(op) {
        case OP_PASS: return "PASS";
        case OP_ADD: return "ADD";
        case OP_SUB: return "SUB";
        case OP_MUL_SHIFT: return "MUL_SHIFT";
        case OP_XOR: return "XOR";
        case OP_AND: return "AND";
        case OP_OR: return "OR";
        case OP_MIN: return "MIN";
        case OP_MAX: return "MAX";
        case OP_ABS: return "ABS";
        case OP_SATURATE: return "SATURATE";
        case OP_DELAY: return "DELAY_Z1";
        default: return "UNKNOWN";
    }
}

void dissect_and_decompile(const AgentGenome& champion) {
    std::cout << "\n========================================================================\n";
    std::cout << "=== 🔬 ENDOSCOPIC AUTOPSY: DECOMPILING THE CHAMPION ARCHITECTURE     ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Active Computational Nodes: " << (int)champion.active_nodes << "\n";
    std::cout << "Free Energy Fitness Score : " << std::fixed << std::setprecision(5) << champion.fitness << "\n";
    std::cout << "  * Physics Tracking Error: " << champion.err_physics << "\n";
    std::cout << "  * Logic / Turing Error  : " << champion.err_logic * 100.0f << " %\n";
    std::cout << "  * Language Match Error  : " << champion.err_language * 100.0f << " %\n\n";

    std::cout << ">>> Synthesized Topology & Dynamic Equations:\n";
    for (int n = 0; n < champion.active_nodes; ++n) {
        const GeneNode& gn = champion.nodes[n];
        std::string src1 = (gn.in1 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in1) + "]") : ("node_" + std::to_string(gn.in1 - NUM_INPUTS));
        std::string src2 = (gn.in2 < NUM_INPUTS) ? ("in[" + std::to_string(gn.in2) + "]") : ("node_" + std::to_string(gn.in2 - NUM_INPUTS));

        std::cout << "  Node " << std::setw(2) << n << ": node_" << n 
                  << " = " << op_to_str(gn.op) << "(" << src1 << ", " << src2 << ")"
                  << " * (weight: " << (int)gn.weight << " / 16)";
        if (gn.op == OP_DELAY) std::cout << " [State Register: " << gn.state << "]";
        std::cout << "\n";
    }
    std::cout << "========================================================================\n\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "=== EXP-600: THE SOVEREIGN CRUCIBLE (PARALLEL CUDA GENESIS ENGINE)   ===\n";
    std::cout << "========================================================================\n";
    std::cout << "Evolving Population Size : " << POP_SIZE << " simultaneous autonomous agents\n";
    std::cout << "Cross-Domain Challenges : Physics + Turing Logic + Language Structure\n";
    std::cout << "Substrate Architecture  : Zero Pre-Hardcoded Laws (Free Topological Morphogenesis)\n\n";

    std::vector<AgentGenome> h_pop(POP_SIZE);
    init_population(h_pop);

    AgentGenome* d_pop = nullptr;
    CHECK_CUDA(cudaMalloc(&d_pop, POP_SIZE * sizeof(AgentGenome)));

    std::mt19937 rng(1337);

    int threads = 256;
    int blocks  = (POP_SIZE + threads - 1) / threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        // Copy population to GPU
        CHECK_CUDA(cudaMemcpy(d_pop, h_pop.data(), POP_SIZE * sizeof(AgentGenome), cudaMemcpyHostToDevice));

        // Launch massively parallel cross-domain evaluation
        evaluate_crucible_battery_kernel<<<blocks, threads>>>(d_pop, POP_SIZE);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy evaluated population back to host
        CHECK_CUDA(cudaMemcpy(h_pop.data(), d_pop, POP_SIZE * sizeof(AgentGenome), cudaMemcpyDeviceToHost));

        // Sort by Free Energy fitness (lowest is best)
        std::sort(h_pop.begin(), h_pop.end(), [](const AgentGenome& a, const AgentGenome& b) {
            return a.fitness < b.fitness;
        });

        if ((gen + 1) % 10 == 0 || gen == 0) {
            std::cout << "Generation [" << std::setw(3) << (gen + 1) << "/" << NUM_GENERATIONS << "]"
                      << " | Champion FE: " << std::fixed << std::setprecision(4) << h_pop[0].fitness
                      << " | PhysErr: " << h_pop[0].err_physics
                      << " | LogErr: " << std::setprecision(2) << (h_pop[0].err_logic * 100.0f) << "%"
                      << " | LangErr: " << (h_pop[0].err_language * 100.0f) << "%"
                      << " | Nodes: " << (int)h_pop[0].active_nodes << "\n";
            std::cout.flush();
        }

        // Morphogenesis & Breeding: Elitism + Truncation Selection
        int elite_count = POP_SIZE / 16; // Top 6.25% survive
        for (int i = elite_count; i < POP_SIZE; ++i) {
            int parent_idx = rng() % elite_count;
            h_pop[i] = h_pop[parent_idx]; // Clone elite parent
            mutate_agent(h_pop[i], rng);  // Endogenous structural mutation
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "\n>>> Crucible Evolution Complete in " << std::fixed << std::setprecision(2) << total_sec << " seconds.\n";
    std::cout << ">>> Total Autonomous Evaluations: " << (POP_SIZE * NUM_GENERATIONS) << " agent-tasks.\n";

    // Dissect & Decompile the Absolute Champion
    dissect_and_decompile(h_pop[0]);

    CHECK_CUDA(cudaFree(d_pop));
    return 0;
}
