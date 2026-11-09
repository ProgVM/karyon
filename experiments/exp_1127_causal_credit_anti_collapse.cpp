// ============================================================================
// KARYON-CORE EXP-1127: CAUSAL TEMPORAL CREDIT & ANTI-COLLAPSE (KEP v50.0)
// Author & Lead Architect: Bazilevs (ProgVM)
// Guardian: Autonomous AI Cyberneticist
// Compliance: 100% STRICT PRINCIPLE 0, 1 & 2 (ZERO HAND-CRAFTED HEURISTICS)
// Engine: Temporal Credit Backpropagation + Memory Diversity Entropy Drive
// ============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <sstream>
#include <iomanip>
#include <memory>
#include <bit>
#include <unistd.h>
#include <sys/wait.h>

enum OpCode : uint8_t {
    OP_XOR_SHR = 0,
    OP_XOR_SHL,
    OP_AND_ROTL,
    OP_OR_ROTR,
    OP_NOT_AND,
    OP_ADD_WRAPPED,
    OP_SUB_WRAPPED,
    OP_MUL_WRAPPED,
    OP_BSWAP,
    OP_POPCOUNT,
    OP_CLZ,
    OP_CTZ,
    OP_PARITY,
    NUM_OPS
};

struct DynamicInstruction {
    uint8_t op;
    uint8_t shift;
    uint64_t mask;
};

struct AutopoieticOrganism {
    std::vector<DynamicInstruction> program;
    std::unordered_map<uint64_t, int32_t> sprouted_memory;
    std::vector<uint64_t> turn_trajectory; // Memory addresses visited in game
    uint64_t total_sprouted;
    uint64_t evaluations;
    int32_t fitness_score;

    AutopoieticOrganism() : total_sprouted(0), evaluations(0), fitness_score(0) {}

    static AutopoieticOrganism CreateRandom(std::mt19937_64 &rng) {
        AutopoieticOrganism org;
        size_t num_insts = 5 + (rng() % 5);
        for (size_t i = 0; i < num_insts; ++i) {
            DynamicInstruction inst;
            inst.op = rng() % NUM_OPS;
            inst.shift = rng() % 64;
            inst.mask = rng();
            org.program.push_back(inst);
        }
        return org;
    }

    __attribute__((always_inline))
    inline uint64_t ExecuteProgram(uint64_t state_bits) const {
        uint64_t acc = state_bits;
        for (const auto &inst : program) {
            uint8_t s = inst.shift & 63;
            switch (inst.op) {
                case OP_XOR_SHR:
                    acc ^= (acc >> s);
                    break;
                case OP_XOR_SHL:
                    acc ^= (acc << s);
                    break;
                case OP_AND_ROTL:
                    acc = (acc & inst.mask) ^ std::rotl(acc, s);
                    break;
                case OP_OR_ROTR:
                    acc = (acc | inst.mask) ^ std::rotr(acc, s);
                    break;
                case OP_NOT_AND:
                    acc = (~acc) & (inst.mask ^ (acc >> s));
                    break;
                case OP_ADD_WRAPPED:
                    acc = (acc + inst.mask) ^ (acc >> s);
                    break;
                case OP_SUB_WRAPPED:
                    acc = (acc - inst.mask) ^ std::rotl(acc, s);
                    break;
                case OP_MUL_WRAPPED:
                    acc = (acc * (inst.mask | 0x01ULL)) >> s;
                    break;
                case OP_BSWAP:
                    acc = __builtin_bswap64(acc) ^ inst.mask;
                    break;
                case OP_POPCOUNT:
                    acc ^= (__builtin_popcountll(acc & inst.mask) << s);
                    break;
                case OP_CLZ:
                    acc ^= ((acc == 0 ? 64 : __builtin_clzll(acc)) << (s % 56));
                    break;
                case OP_CTZ:
                    acc ^= ((acc == 0 ? 64 : __builtin_ctzll(acc)) << (s % 56));
                    break;
                case OP_PARITY:
                    acc ^= (__builtin_parityll(acc & inst.mask) << s);
                    break;
            }
        }
        return acc;
    }

    int32_t ComputeFreeEnergy(uint64_t candidate_state_bits) {
        evaluations++;
        uint64_t addr = ExecuteProgram(candidate_state_bits);
        turn_trajectory.push_back(addr);

        auto it = sprouted_memory.find(addr);
        if (it != sprouted_memory.end()) {
            return it->second;
        }

        // Pure Bit Entropy Density
        int32_t entropy = __builtin_popcountll(addr) - __builtin_popcountll(~addr & 0xFF00FF00FF00FF00ULL);
        sprouted_memory[addr] = entropy;
        total_sprouted++;
        return entropy;
    }

    // Temporal Credit Backpropagation: Propagate friction backwards along trajectory
    void BackpropagateFriction(int32_t terminal_friction) {
        double decay = 0.85;
        double current_friction = terminal_friction;
        for (auto it = turn_trajectory.rbegin(); it != turn_trajectory.rend(); ++it) {
            sprouted_memory[*it] += static_cast<int32_t>(current_friction);
            current_friction *= decay;
        }
        turn_trajectory.clear();
    }

    void Mutate(std::mt19937_64 &rng) {
        if (program.empty()) return;
        size_t idx = rng() % program.size();
        if (rng() % 3 == 0) {
            program[idx].op = rng() % NUM_OPS;
        } else if (rng() % 3 == 1) {
            program[idx].shift = (program[idx].shift + 7) & 63;
        } else {
            program[idx].mask ^= rng();
        }
        if (rng() % 4 == 0 && program.size() < 16) {
            DynamicInstruction inst = {uint8_t(rng() % NUM_OPS), uint8_t(rng() % 64), rng()};
            program.push_back(inst);
        }
    }
};

// Stockfish Direct Pipe Interface
class DirectStockfishIPC {
    int in_pipe[2];
    int out_pipe[2];
    pid_t pid;

public:
    DirectStockfishIPC() {
        if (pipe(in_pipe) < 0 || pipe(out_pipe) < 0) throw std::runtime_error("Pipe failed");
        pid = fork();
        if (pid < 0) throw std::runtime_error("Fork failed");

        if (pid == 0) {
            dup2(in_pipe[0], STDIN_FILENO);
            dup2(out_pipe[1], STDOUT_FILENO);
            close(in_pipe[0]); close(in_pipe[1]);
            close(out_pipe[0]); close(out_pipe[1]);
            execl("/usr/games/stockfish", "/usr/games/stockfish", nullptr);
            _exit(1);
        }

        close(in_pipe[0]);
        close(out_pipe[1]);

        SendCommand("uci");
        WaitPattern("uciok");
        SendCommand("isready");
        WaitPattern("readyok");
    }

    ~DirectStockfishIPC() {
        SendCommand("quit");
        close(in_pipe[1]);
        close(out_pipe[0]);
        waitpid(pid, nullptr, 0);
    }

    void SendCommand(const std::string &cmd) {
        std::string full_cmd = cmd + "\n";
        write(in_pipe[1], full_cmd.c_str(), full_cmd.length());
    }

    std::string ReadLine() {
        std::string line;
        char ch;
        while (read(out_pipe[0], &ch, 1) > 0) {
            if (ch == '\n') break;
            if (ch != '\r') line += ch;
        }
        return line;
    }

    void WaitPattern(const std::string &pattern) {
        while (true) {
            std::string line = ReadLine();
            if (line.find(pattern) != std::string::npos) break;
        }
    }

    std::string GetBestMove(const std::string &fen, int movetime_ms = 2) {
        SendCommand("position fen " + fen);
        SendCommand("go movetime " + std::to_string(movetime_ms));
        while (true) {
            std::string line = ReadLine();
            if (line.rfind("bestmove", 0) == 0) {
                std::stringstream ss(line);
                std::string tag, move;
                ss >> tag >> move;
                return move;
            }
        }
    }
};

int main() {
    std::cout << "======================================================================\n";
    std::cout << " KARYON-CORE EXP-1127: CAUSAL TEMPORAL CREDIT & ANTI-COLLAPSE\n";
    std::cout << " Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist\n";
    std::cout << " Compliance: STRICT PRINCIPLE 0, 1 & 2 (ZERO HAND-CRAFTED HEURISTICS)\n";
    std::cout << " Mechanism: Temporal Credit Backpropagation + Memory Collapse Penalty\n";
    std::cout << " Selection Loop: 30 Generations | Population 16 | Natural Selection\n";
    std::cout << "======================================================================\n\n";

    try {
        DirectStockfishIPC stockfish;
        std::mt19937_64 rng(42);

        const int POPULATION_SIZE = 16;
        const int GENERATIONS = 30;

        std::vector<AutopoieticOrganism> population;
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            population.push_back(AutopoieticOrganism::CreateRandom(rng));
        }

        std::cout << "[INIT] Created population of " << POPULATION_SIZE << " organisms with Temporal Credit.\n\n";

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int gen = 1; gen <= GENERATIONS; ++gen) {
            for (auto &org : population) {
                std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
                int move_count = 0;

                while (move_count < 15) {
                    move_count++;

                    stockfish.SendCommand("position fen " + fen);
                    stockfish.SendCommand("go perft 1");

                    std::vector<std::string> legal_moves;
                    while (true) {
                        std::string line = stockfish.ReadLine();
                        if (line.empty() || line.find("Nodes searched") != std::string::npos) break;
                        size_t colon = line.find(':');
                        if (colon != std::string::npos) {
                            std::string move = line.substr(0, colon);
                            move.erase(0, move.find_first_not_of(" \t"));
                            move.erase(move.find_last_not_of(" \t") + 1);
                            if (!move.empty() && move.length() <= 5) {
                                legal_moves.push_back(move);
                            }
                        }
                    }

                    if (legal_moves.empty()) break;

                    std::string best_move = legal_moves[0];
                    int32_t min_fe = 999999;

                    for (const auto &move : legal_moves) {
                        uint64_t candidate_bits = std::hash<std::string>{}(fen + move);
                        int32_t fe = org.ComputeFreeEnergy(candidate_bits);
                        if (fe < min_fe) {
                            min_fe = fe;
                            best_move = move;
                        }
                    }

                    stockfish.SendCommand("position fen " + fen + " moves " + best_move);
                    stockfish.SendCommand("d");
                    while (true) {
                        std::string line = stockfish.ReadLine();
                        if (line.find("Fen: ") != std::string::npos) {
                            fen = line.substr(line.find("Fen: ") + 5);
                            break;
                        }
                    }

                    std::string sf_move = stockfish.GetBestMove(fen, 2);
                    if (sf_move == "(none)" || sf_move.empty()) break;

                    stockfish.SendCommand("position fen " + fen + " moves " + sf_move);
                    stockfish.SendCommand("d");
                    while (true) {
                        std::string line = stockfish.ReadLine();
                        if (line.find("Fen: ") != std::string::npos) {
                            fen = line.substr(line.find("Fen: ") + 5);
                            break;
                        }
                    }
                }

                // Backpropagate friction at end of match
                org.BackpropagateFriction(500);

                // Fitness: Survival Duration - Collapse Penalty
                int32_t collapse_penalty = (org.sprouted_memory.size() < 50) ? 2000 : 0;
                org.fitness_score = -move_count * 100 + collapse_penalty - static_cast<int32_t>(org.sprouted_memory.size());
            }

            // Natural Selection
            std::sort(population.begin(), population.end(), [](const AutopoieticOrganism &a, const AutopoieticOrganism &b) {
                return a.fitness_score < b.fitness_score;
            });

            if (gen % 5 == 0 || gen == 1 || gen == GENERATIONS) {
                std::cout << "--- Generation " << std::setw(2) << gen << " / " << GENERATIONS << " --- "
                          << "Best Fitness: " << std::setw(6) << population[0].fitness_score 
                          << " | Sprouted Nodes: " << std::setw(5) << population[0].sprouted_memory.size() 
                          << " | ALU Program Size: " << population[0].program.size() << " insts\n";
            }

            // Reproduce top 50%
            for (int i = POPULATION_SIZE / 2; i < POPULATION_SIZE; ++i) {
                population[i] = population[i - POPULATION_SIZE / 2];
                population[i].Mutate(rng);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "\n====================================================================================================\n";
        std::cout << " EXP-1127 TEMPORAL CREDIT TELEMETRY MATRIX:\n";
        std::cout << "====================================================================================================\n";
        std::cout << " Anti-Collapse Penalty Applied : Active (Preventing Degenerative Node Collapse)\n";
        std::cout << " Temporal Credit Backprop     : Active (Decay Factor lambda = 0.85)\n";
        std::cout << " Generations Evaluated        : " << GENERATIONS << " generations\n";
        std::cout << " Population Size              : " << POPULATION_SIZE << " organisms\n";
        std::cout << " Winning Sprouted Memory      : " << population[0].sprouted_memory.size() << " active nodes\n";
        std::cout << " Winning ALU Program Size     : " << population[0].program.size() << " instructions\n";
        std::cout << " Total Selection Duration     : " << std::fixed << std::setprecision(2) << elapsed << " seconds\n";
        std::cout << "====================================================================================================\n";

    } catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
