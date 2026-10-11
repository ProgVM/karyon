// experiments/exp_463_swarm_udp_stress_test.cpp — EXP-463: Swarm Collective Intelligence Stress-Test
// Standard: KEP v16.0 Sovereign Master & Principle 12 (Universal Modality-Agnostic Substrate)
// Zero-Python / Zero-Torch / Pure C++20 Sovereign Architecture
//
// HYPOTHESIS:
// Deploying a 4-node distributed swarm of Karyon-CoRE instances operating on asynchronous UDP socket mesh
// where each node specializes in a distinct domain (Code, Math, Logic, Prose) will enable continuous background
// phase distillation across the swarm, demonstrating non-blocking collective convergence without packet loss.

#include "../karyon_native_core.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>

namespace karyon {

class SwarmParticipantNode : public SovereignNativeEngine {
public:
    SwarmParticipantNode(const NativeConfig& cfg, uint32_t node_id, int my_port, const std::vector<int>& peer_ports)
        : SovereignNativeEngine(cfg), node_id_(node_id), my_port_(my_port), peer_ports_(peer_ports) {}

    // Broadcast phase spike to all active peers in swarm
    size_t broadcast_to_swarm(float free_energy, const std::vector<float>& q, const std::vector<float>& v) {
        if (p2p_sock_fd_ < 0) return 0;

        SynapticSpikePacket pkt;
        pkt.node_id = node_id_;
        pkt.free_energy = free_energy;

        size_t k_len = std::min(q.size(), static_cast<size_t>(16));
        size_t v_len = std::min(v.size(), static_cast<size_t>(32));

        for (size_t i = 0; i < k_len; ++i) pkt.phase_q[i] = q[i];
        for (size_t i = 0; i < v_len; ++i) pkt.phase_v[i] = v[i];

        size_t sent_count = 0;
        for (int p_port : peer_ports_) {
            struct sockaddr_in dest_addr{};
            dest_addr.sin_family = AF_INET;
            dest_addr.sin_port = htons(p_port);
            inet_pton(AF_INET, "127.0.0.1", &dest_addr.sin_addr);

            ssize_t sent = sendto(p2p_sock_fd_, &pkt, sizeof(pkt), 0, (struct sockaddr*)&dest_addr, sizeof(dest_addr));
            if (sent == static_cast<ssize_t>(sizeof(pkt))) sent_count++;
        }
        return sent_count;
    }

private:
    uint32_t node_id_;
    int my_port_;
    std::vector<int> peer_ports_;
};

} // namespace karyon

int main() {
    std::cout << "==================================================================\n";
    std::cout << "   KARYON-CoRE EXP-463: 4-NODE SWARM UDP STRESS-TEST BENCHMARK\n";
    std::cout << "   Multi-Node Asynchronous Phase Vector Distillation over UDP\n";
    std::cout << "==================================================================\n";

    // Define 4 Node configs with distinct UDP ports
    int ports[4] = {19101, 19102, 19103, 19104};

    std::vector<int> p0 = {ports[1], ports[2], ports[3]};
    std::vector<int> p1 = {ports[0], ports[2], ports[3]};
    std::vector<int> p2 = {ports[0], ports[1], ports[3]};
    std::vector<int> p3 = {ports[0], ports[1], ports[2]};

    karyon::NativeConfig cfg;
    cfg.dim = 258;
    cfg.num_heads = 4;
    cfg.dim_k = 16;
    cfg.dim_v = 32;

    cfg.p2p_port = ports[0];
    karyon::SwarmParticipantNode node_code(cfg, 201, ports[0], p0);

    cfg.p2p_port = ports[1];
    karyon::SwarmParticipantNode node_math(cfg, 202, ports[1], p1);

    cfg.p2p_port = ports[2];
    karyon::SwarmParticipantNode node_logic(cfg, 203, ports[2], p2);

    cfg.p2p_port = ports[3];
    karyon::SwarmParticipantNode node_prose(cfg, 204, ports[3], p3);

    std::string streams[4] = {
        "template <typename T> void process_stream(T data) { return; }\n",
        "E = mc^2; f(x) = int_0^inf exp(-x^2) dx = sqrt(pi) / 2;\n",
        "if (A && B) { C = true; } else { C = false; }\n",
        "The continuous mind operates beyond artificial tokenization boundaries.\n"
    };

    std::cout << "[SWARM CONCURRENCY] Running 4 nodes in parallel threads...\n";

    size_t total_sent_spikes = 0;
    size_t total_received_spikes = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Worker lambda for each swarm node
    auto run_worker = [](karyon::SwarmParticipantNode* node, const std::string& text, size_t* sent_out, size_t* recv_out) {
        size_t sent_sum = 0;
        size_t recv_sum = 0;

        for (size_t i = 0; i < text.size() - 1; ++i) {
            uint8_t in_b = static_cast<uint8_t>(text[i]);
            uint8_t target_b = static_cast<uint8_t>(text[i + 1]);

            auto m = node->step(in_b, target_b);

            std::vector<float> q_dummy(16, 0.1f * (i % 5));
            std::vector<float> v_dummy(32, 0.2f * (i % 7));

            sent_sum += node->broadcast_to_swarm(m.loss, q_dummy, v_dummy);

            while (node->receive_mesh_spike()) {
                recv_sum++;
            }
        }
        *sent_out = sent_sum;
        *recv_out = recv_sum;
    };

    size_t sent_arr[4] = {0}, recv_arr[4] = {0};

    std::thread t1(run_worker, &node_code, std::ref(streams[0]), &sent_arr[0], &recv_arr[0]);
    std::thread t2(run_worker, &node_math, std::ref(streams[1]), &sent_arr[1], &recv_arr[1]);
    std::thread t3(run_worker, &node_logic, std::ref(streams[2]), &sent_arr[2], &recv_arr[2]);
    std::thread t4(run_worker, &node_prose, std::ref(streams[3]), &sent_arr[3], &recv_arr[3]);

    t1.join(); t2.join(); t3.join(); t4.join();

    // Small drain window for socket buffers
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    for (int i = 0; i < 4; ++i) {
        total_sent_spikes += sent_arr[i];
        total_received_spikes += recv_arr[i];
    }

    auto t1_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1_time - t0).count();

    size_t safe_denom = std::max(total_sent_spikes, static_cast<size_t>(1));
    float delivery_rate = 100.0f * total_received_spikes / safe_denom;

    std::cout << "\n=============================================================\n";
    std::cout << "   EXP-463 BENCHMARK TELEMETRY & DECISION VERDICT\n";
    std::cout << "=============================================================\n";
    std::cout << "Active Swarm Nodes         : 4 (Code, Math, Logic, Prose)\n";
    std::cout << "Total Swarm Sent Spikes    : " << total_sent_spikes << "\n";
    std::cout << "Total Swarm Recv Spikes    : " << total_received_spikes << "\n";
    std::cout << "P2P Socket Delivery Rate   : " << delivery_rate << " %\n";
    std::cout << "Swarm Processing Time      : " << (elapsed * 1000.0) << " ms\n";
    std::cout << "=============================================================\n";

    std::ofstream out_json("experiments/exp_463_telemetry.json");
    out_json << "{\n"
             << "  \"exp_id\": \"EXP-463\",\n"
             << "  \"swarm_nodes\": 4,\n"
             << "  \"total_sent_spikes\": " << total_sent_spikes << ",\n"
             << "  \"total_recv_spikes\": " << total_received_spikes << ",\n"
             << "  \"delivery_rate_percent\": " << delivery_rate << ",\n"
             << "  \"elapsed_ms\": " << (elapsed * 1000.0) << "\n"
             << "}\n";
    out_json.close();

    return 0;
}
