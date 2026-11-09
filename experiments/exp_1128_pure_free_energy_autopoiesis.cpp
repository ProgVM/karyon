// ============================================================================
// KARYON-CORE EXP-1128: PURE FREE ENERGY AUTOPOIESIS (KEP v50.0 MASTER)
// Author & Lead Architect: Bazilevs (ProgVM)
// Guardian: Autonomous AI Cyberneticist
// Compliance: 100% PRINCIPLE 0, 1 & 2 (ZERO ARTIFICIAL PENALTIES / NO CRUTCHES)
// Mechanics: Free Energy F = Environmental Bit Discrepancy popcount(S_pred ^ S_actual)
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
    std::unordered_map<uint64_t, uint64_t> sprouted_memory; // Map: state_bits -> predicted_next_bits
    uint64_t total_sprouted;
    uint64_t total_free_energy_bits; // Sum of environmental bit discrepancies
    uint64_t total_turns;

    AutopoieticOrganism() : total_sprouted(0), total_free_energy_bits(0), total_turns(0) {}

    static AutopoieticOrganism CreateRandom(std::mt19937_64 &rng) {
        AutopoieticOrganism org;
        size_t num_insts = 5 + (rng() % 6);
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

    // Predict next environment state bits
    uint64_t PredictNextState(uint64_t state_bits) {
        uint64_t predicted_bits = ExecuteProgram(state_bits);
        if (sprouted_memory.find(state_bits) == sprouted_memory.end()) {
            sprouted_memory[state_bits] = predicted_bits;
            total_sprouted++;
        }
        return predicted_bits;
    }

    // Measure Environmental Bit Discrepancy (Pure Free Energy F)
    void RecordEnvironmentalDiscrepancy(uint64_t predicted_bits, uint64_t actual_bits) {
        uint64_t discrepancy_bits = __builtin_popcountll(predicted_bits ^ actual_bits);
        total_free_energy_bits += discrepancy_bits;
        total_turns++;
    }

    double GetMeanFreeEnergy() const {
        if (total_turns == 0) return 64.0;
        return static_cast<double>(total_free_energy_bits) / static_cast<double>(total_turns);
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
    std::cout << " KARYON-CORE EXP-1128: PURE FREE ENERGY AUTOPOIESIS (ZERO PENALTIES)\n";
    std::cout << " Lead Architect: Bazilevs (ProgVM) | Guardian: Autonomous AI Cyberneticist\n";
    std::cout << " Compliance: 100% STRICT PRINCIPLE 0, 1 & 2 (ZERO ARTIFICIAL HEURISTICS)\n";
    std::cout << " Objective: Natural Selection Driven SOLELY by Free Energy F = Bit Error\n";
    std::cout << " Selection Loop: 30 Generations | Population 16 | Pure Active Inference\n";
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

        std::cout << "[INIT] Created population of " << POPULATION_SIZE << " organisms. Zero artificial penalties.\n\n";

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int gen = 1; gen <= GENERATIONS; ++gen) {
            for (auto &org : population) {
                org.total_free_energy_bits = 0;
                org.total_turns = 0;

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

                    // Current environmental bit state
                    uint64_t current_env_bits = std::hash<std::string>{}(fen);

                    // Organism predicts next state using its autopoietic program
                    uint64_t predicted_bits = org.PredictNextState(current_env_bits);

                    // Candidate selection: choose move that minimizes internal bit friction
                    std::string best_move = legal_moves[0];
                    uint64_t min_discrepancy = 64;

                    for (const auto &move : legal_moves) {
                        uint64_t move_bits = std::hash<std::string>{}(fen + move);
                        uint64_t candidate_pred = org.ExecuteProgram(move_bits);
                        uint64_t candidate_discrepancy = __builtin_popcountll(candidate_pred ^ predicted_bits);
                        if (candidate_discrepancy < min_discrepancy) {
                            min_discrepancy = candidate_discrepancy;
                            best_move = move;
                        }
                    }

                    // Execute chosen move in Stockfish environment
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

                    // Measure actual resulting state bits from environment
                    uint64_t actual_next_env_bits = std::hash<std::string>{}(fen);

                    // Calculate Free Energy F = Environmental Bit Discrepancy
                    org.RecordEnvironmentalDiscrepancy(predicted_bits, actual_next_env_bits);
                }
            }

            // Natural Selection: Organisms sorted STRICTLY by Mean Free Energy F (Lower F = Better)
            std::sort(population.begin(), population.end(), [](const AutopoieticOrganism &a, const AutopoieticOrganism &b) {
                return a.GetMeanFreeEnergy() < b.GetMeanFreeEnergy();
            });

            if (gen % 5 == 0 || gen == 1 || gen == GENERATIONS) {
                std::cout << "--- Generation " << std::setw(2) << gen << " / " << GENERATIONS << " --- "
                          << "Best Free Energy F: " << std::fixed << std::setprecision(4) << population[0].GetMeanFreeEnergy() << " bits/turn "
                          << " | Sprouted Memory: " << std::setw(5) << population[0].sprouted_memory.size() << " nodes"
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
        std::cout << " EXP-1128 PURE FREE ENERGY AUTOPOIESIS TELEMETRY MATRIX:\n";
        std::cout << "====================================================================================================\n";
        std::cout << " Imposed Penalties / Rules   : ZERO (0.00% Imposition - Pure FE Drive)\n";
        std::cout << " Free Energy Formulation F   : Environmental Bit Discrepancy popcount(pred ^ actual)\n";
        std::cout << " Generations Evaluated       : " << GENERATIONS << " generations\n";
        std::cout << " Population Size             : " << POPULATION_SIZE << " organisms\n";
        std::cout << " Winning Free Energy F       : " << std::fixed << std::setprecision(4) << population[0].GetMeanFreeEnergy() << " bits / turn\n";
        std::cout << " Winning Sprouted Memory     : " << population[0].sprouted_memory.size() << " active nodes\n";
        std::cout << " Winning ALU Program Size    : " << population[0].program.size() << " instructions\n";
        std::cout << " Total Selection Duration    : " << std::setprecision(2) << elapsed << " seconds\n";
        std::cout << "====================================================================================================\n";

    } catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
