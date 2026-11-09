// ============================================================================
// KARYON-CORE EXP-1125: EVOLUTIONARY AUTOPOIETIC SUBSTRATE SELECTION (KEP v50.0)
// Author & Lead Architect: Bazilevs (ProgVM)
// Guardian: Autonomous AI Cyberneticist
// Compliance: STRICT PRINCIPLE 0, 1 & 2 (ZERO HAND-CRAFTED HEURISTICS)
// Engine: Multi-Generational Natural Selection of Hardware ALU Genomes
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
#include <unistd.h>
#include <sys/wait.h>

enum OpCode : uint8_t {
    OP_XOR_SHR = 0,
    OP_AND_ROT,
    OP_ADD_WRAPPED,
    OP_BSWAP,
    OP_POPCOUNT_MASK,
    NUM_OPS
};

struct DynamicInstruction {
    uint8_t op;
    uint8_t shift;
    uint64_t mask;
};

// Autopoietic Organism Candidate
struct AutopoieticOrganism {
    std::vector<DynamicInstruction> program;
    std::unordered_map<uint64_t, int32_t> sprouted_memory;
    uint64_t total_sprouted;
    uint64_t evaluations;
    int32_t fitness_score; // Lower is better (Free Energy F)

    AutopoieticOrganism() : total_sprouted(0), evaluations(0), fitness_score(0) {}

    static AutopoieticOrganism CreateRandom(std::mt19937_64 &rng) {
        AutopoieticOrganism org;
        size_t num_insts = 4 + (rng() % 5);
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
            switch (inst.op) {
                case OP_XOR_SHR:
                    acc ^= (acc >> (inst.shift & 63));
                    break;
                case OP_AND_ROT:
                    acc = (acc & inst.mask) ^ std::rotl(acc, inst.shift & 63);
                    break;
                case OP_ADD_WRAPPED:
                    acc = (acc + inst.mask) ^ (acc >> (inst.shift & 63));
                    break;
                case OP_BSWAP:
                    acc = __builtin_bswap64(acc) ^ inst.mask;
                    break;
                case OP_POPCOUNT_MASK:
                    acc ^= (__builtin_popcountll(acc & inst.mask) << (inst.shift & 63));
                    break;
            }
        }
        return acc;
    }

    int32_t ComputeFreeEnergy(uint64_t candidate_state_bits) {
        evaluations++;
        uint64_t addr = ExecuteProgram(candidate_state_bits);
        auto it = sprouted_memory.find(addr);
        if (it != sprouted_memory.end()) {
            return it->second;
        }

        // Pure Bit Entropy Density (Zero Human Heuristics)
        int32_t entropy = __builtin_popcountll(addr) - __builtin_popcountll(~addr & 0xFF00FF00FF00FF00ULL);
        sprouted_memory[addr] = entropy;
        total_sprouted++;
        return entropy;
    }

    void Mutate(std::mt19937_64 &rng) {
        if (program.empty()) return;
        size_t idx = rng() % program.size();
        if (rng() % 2 == 0) {
            program[idx].shift = (program[idx].shift + 7) & 63;
        } else {
            program[idx].mask ^= rng();
        }
        if (rng() % 5 == 0 && program.size() < 12) {
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

    std::string GetBestMove(const std::string &fen, int movetime_ms = 5) {
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
    std::cout << " KARYON-CORE EXP-1125: EVOLUTIONARY AUTOPOIETIC SELECTION\n";
    std::cout << " Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist\n";
    std::cout << " Compliance: STRICT PRINCIPLE 0, 1 & 2 (ZERO HAND-CRAFTED HEURISTICS)\n";
    std::cout << " Selection Loop: 10 Generations | Population 8 | Natural Selection\n";
    std::cout << "======================================================================\n\n";

    try {
        DirectStockfishIPC stockfish;
        std::mt19937_64 rng(42);

        const int POPULATION_SIZE = 8;
        const int GENERATIONS = 10;

        std::vector<AutopoieticOrganism> population;
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            population.push_back(AutopoieticOrganism::CreateRandom(rng));
        }

        std::cout << "[INIT] Created initial population of " << POPULATION_SIZE << " autopoietic organisms.\n\n";

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int gen = 1; gen <= GENERATIONS; ++gen) {
            std::cout << "--- Generation " << std::setw(2) << gen << " / " << GENERATIONS << " --- " << std::flush;

            // Evaluate population against Stockfish
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

                    // Candidate selection by Free Energy
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

                    std::string sf_move = stockfish.GetBestMove(fen, 5);
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

                // Fitness: Survival duration + sprouted node entropy efficiency
                org.fitness_score = -move_count * 100 + org.total_sprouted % 50;
            }

            // Natural Selection: Sort by fitness (lower/more negative score is better)
            std::sort(population.begin(), population.end(), [](const AutopoieticOrganism &a, const AutopoieticOrganism &b) {
                return a.fitness_score < b.fitness_score;
            });

            std::cout << "Best Organism Fitness: " << population[0].fitness_score 
                      << " | Sprouted Nodes: " << population[0].total_sprouted 
                      << " | Program Size: " << population[0].program.size() << " instructions\n";

            // Reproduce & Mutate top 50%
            for (int i = POPULATION_SIZE / 2; i < POPULATION_SIZE; ++i) {
                population[i] = population[i - POPULATION_SIZE / 2];
                population[i].Mutate(rng);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "\n====================================================================================================\n";
        std::cout << " EXP-1125 NATURAL SELECTION TELEMETRY MATRIX:\n";
        std::cout << "====================================================================================================\n";
        std::cout << " Generations Evaluated     : " << GENERATIONS << " generations\n";
        std::cout << " Population Size           : " << POPULATION_SIZE << " organisms\n";
        std::cout << " Winning ALU Program Size  : " << population[0].program.size() << " instructions\n";
        std::cout << " Winning Sprouted Nodes    : " << population[0].total_sprouted << " nodes\n";
        std::cout << " Total Selection Duration  : " << std::fixed << std::setprecision(2) << elapsed << " seconds\n";
        std::cout << "====================================================================================================\n";

    } catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
