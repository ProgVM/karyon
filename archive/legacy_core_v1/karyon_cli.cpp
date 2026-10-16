#include "karyon_native_core.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

void print_banner() {
    std::cout << "==================================================================\n";
    std::cout << "   KARYON-CoRE v19.0 — SOVEREIGN NATIVE C++20 AUTONOMOUS ENGINE    \n";
    std::cout << "   Zero-Python / Zero-Torch / Pure Substrate Physics (Bazilevs)   \n";
    std::cout << "==================================================================\n";
}

void print_help(const char* prog) {
    std::cout << "Usage: " << prog << " [command] [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  train <file> [--kcore out.kcore]  Online stream learning on a raw file\n";
    std::cout << "  interact [--kcore soul.kcore]     Direct conversational stream via stdin\n";
    std::cout << "  benchmark                         Run built-in multi-head physical benchmark\n";
    std::cout << "  info <soul.kcore>                 Inspect binary container metadata\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --heads <N>       Number of associative heads (default: 8)\n";
    std::cout << "  --lr <float>      Learning rate (default: 0.015)\n";
    std::cout << "  --kcore <path>    Path to .kcore binary checkpoint\n";
    std::cout << "==================================================================\n";
}

int run_benchmark() {
    std::cout << "[INFO] Initializing Karyon Sovereign Native Engine (C++20)...\n";
    karyon::NativeConfig cfg;
    karyon::SovereignNativeEngine engine(cfg);

    std::string test_data = 
        "def karyon_sovereign_autopoiesis(stream):\n"
        "    super_operator = Liouvillian(phase_space)\n"
        "    return super_operator.synthesize()\n\n"
        "#include <iostream>\n"
        "void dynamic_kernel() { /* zero-torch */ }\n\n"
        "Mind is substrate-independent continuous field dynamics.\n"
        "Ashby homeostatic ultrastability balances energy and surprise.\n";

    std::vector<uint8_t> stream;
    for (int cycle = 0; cycle < 10; ++cycle) {
        stream.insert(stream.end(), test_data.begin(), test_data.end());
    }

    std::cout << "[INFO] Stream length: " << stream.size() << " bytes.\n";
    std::cout << "[INFO] Starting single-pass online learning loop...\n";

    auto t_start = std::chrono::high_resolution_clock::now();
    float total_loss = 0.0f;
    size_t correct = 0;
    size_t total_turns = 0;

    for (size_t i = 0; i < stream.size() - 1; ++i) {
        auto m = engine.step(stream[i], stream[i + 1]);
        total_loss += m.loss;
        if (m.is_correct) correct++;
        total_turns += m.turns_taken;

        if ((i + 1) % 500 == 0 || i == stream.size() - 2) {
            float cur_loss = total_loss / static_cast<float>(i + 1);
            float cur_acc = (static_cast<float>(correct) / static_cast<float>(i + 1)) * 100.0f;
            float cur_turns = static_cast<float>(total_turns) / static_cast<float>(i + 1);
            std::cout << "  Step " << std::setw(4) << (i + 1) << "/" << (stream.size() - 1)
                      << " | Loss: " << std::fixed << std::setprecision(4) << cur_loss
                      << " | Acc: " << std::setw(5) << std::setprecision(2) << cur_acc << "%"
                      << " | Avg Turns: " << std::setprecision(2) << cur_turns
                      << " | Beta: " << std::setprecision(2) << m.beta
                      << " | S_norm: " << std::setprecision(2) << m.s_norm << "\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
    double throughput = static_cast<double>(stream.size() - 1) / elapsed_s;

    std::cout << "==================================================================\n";
    std::cout << "BENCHMARK RESULTS:\n";
    std::cout << "  Final Loss   : " << std::fixed << std::setprecision(4) << (total_loss / (stream.size() - 1)) << " nats\n";
    std::cout << "  Accuracy     : " << std::fixed << std::setprecision(2) << ((double)correct / (stream.size() - 1) * 100.0) << " %\n";
    std::cout << "  Throughput   : " << std::fixed << std::setprecision(2) << throughput << " steps/second\n";
    std::cout << "  Elapsed Time : " << std::fixed << std::setprecision(3) << elapsed_s << " s\n";
    std::cout << "==================================================================\n";

    // Test saving .kcore container
    std::vector<uint8_t> kcore_data = engine.serialize_kcore();
    std::ofstream out("standalone_soul.kcore", std::ios::binary);
    out.write(reinterpret_cast<const char*>(kcore_data.data()), kcore_data.size());
    out.close();
    std::cout << "[INFO] Saved standalone .kcore container (" << kcore_data.size() << " bytes) to standalone_soul.kcore\n";

    return 0;
}

int run_train_file(const std::string& filepath, const std::string& kcore_out) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filepath << "\n";
        return 1;
    }

    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
    in.close();

    if (data.size() < 2) {
        std::cerr << "[ERROR] File is too small for stream learning: " << filepath << "\n";
        return 1;
    }

    std::cout << "[INFO] Training on " << filepath << " (" << data.size() << " bytes)...\n";
    karyon::NativeConfig cfg;
    karyon::SovereignNativeEngine engine(cfg);

    auto t_start = std::chrono::high_resolution_clock::now();
    float total_loss = 0.0f;
    size_t correct = 0;

    for (size_t i = 0; i < data.size() - 1; ++i) {
        auto m = engine.step(data[i], data[i + 1]);
        total_loss += m.loss;
        if (m.is_correct) correct++;

        if ((i + 1) % 1000 == 0 || i == data.size() - 2) {
            float cur_loss = total_loss / static_cast<float>(i + 1);
            float cur_acc = (static_cast<float>(correct) / static_cast<float>(i + 1)) * 100.0f;
            std::cout << "  Processed " << (i + 1) << "/" << (data.size() - 1)
                      << " bytes | Loss: " << std::fixed << std::setprecision(4) << cur_loss
                      << " | Acc: " << std::setprecision(2) << cur_acc << "%\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "[INFO] Completed in " << elapsed_s << " s (" << (data.size() / elapsed_s) << " bytes/sec)\n";

    if (!kcore_out.empty()) {
        auto kcore_data = engine.serialize_kcore();
        std::ofstream out(kcore_out, std::ios::binary);
        out.write(reinterpret_cast<const char*>(kcore_data.data()), kcore_data.size());
        out.close();
        std::cout << "[INFO] Saved soul to " << kcore_out << " (" << kcore_data.size() << " bytes)\n";
    }

    return 0;
}

int main(int argc, char* argv[]) {
    print_banner();

    if (argc < 2) {
        print_help(argv[0]);
        return 0;
    }

    std::string cmd = argv[1];
    if (cmd == "benchmark") {
        return run_benchmark();
    } else if (cmd == "train") {
        if (argc < 3) {
            std::cerr << "Error: missing input file for train.\n";
            return 1;
        }
        std::string filepath = argv[2];
        std::string kcore_out = "karyon_soul.kcore";
        for (int i = 3; i < argc - 1; ++i) {
            if (std::string(argv[i]) == "--kcore") {
                kcore_out = argv[i + 1];
            }
        }
        return run_train_file(filepath, kcore_out);
    } else {
        print_help(argv[0]);
    }

    return 0;
}
