#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

struct SAMState {
    int len = 0;
    int link = -1;
    int next[2] = {-1, -1};
    uint32_t count[2] = {0, 0};
};

int main() {
    std::ifstream file("data/ultrahd_1024.bin", std::ios::binary);
    if (!file.is_open()) return 1;
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    size_t total_bits = raw.size() * 8;
    int stride_bits = 1024 * 3 * 8; // 24576 bits (1 row)

    std::vector<SAMState> st;
    st.push_back({0, -1, {-1, -1}, {0, 0}});
    int last = 0;
    std::vector<int> state_history;

    uint64_t valid_evals = 0;
    uint64_t correct_preds = 0;
    double fe_sum = 0.0;

    for (size_t t = 0; t < total_bits; ++t) {
        size_t byte_idx = t / 8;
        int bit_idx = 7 - (t % 8);
        uint8_t bit = (raw[byte_idx] >> bit_idx) & 1;

        // Predict bit at t using state at (t - stride_bits) ONLY
        if (t >= static_cast<size_t>(stride_bits)) {
            size_t query_t = t - stride_bits;
            int s = state_history[query_t];
            
            float p1 = 0.5f;
            while (s != -1) {
                uint32_t c0 = st[s].count[0];
                uint32_t c1 = st[s].count[1];
                uint32_t tot = c0 + c1;
                if (tot > 0) {
                    if (c0 > 0 && c1 == 0) p1 = 0.000001f;
                    else if (c1 > 0 && c0 == 0) p1 = 0.999999f;
                    else p1 = static_cast<float>(c1) / tot;
                    break;
                }
                s = st[s].link;
            }

            p1 = std::clamp(p1, 1e-6f, 1.0f - 1e-6f);
            float p_act = (bit == 1) ? p1 : (1.0f - p1);
            fe_sum += -std::log(p_act);
            if ((p1 >= 0.5f && bit == 1) || (p1 < 0.5f && bit == 0)) correct_preds++;
            valid_evals++;

            // UPDATE ONLY FOR SPATIAL STRIDE (NOT 1D)
            s = state_history[query_t];
            while (s != -1) {
                st[s].count[bit]++;
                s = st[s].link;
            }
        }

        // SAM 1D Topological State Extension (WITHOUT updating 1D count!)
        int cur = static_cast<int>(st.size());
        st.push_back({st[last].len + 1, -1, {-1, -1}, {0, 0}});

        int p = last;
        while (p != -1 && st[p].next[bit] == -1) {
            st[p].next[bit] = cur;
            p = st[p].link;
        }

        if (p == -1) {
            st[cur].link = 0;
        } else {
            int q = st[p].next[bit];
            if (st[p].len + 1 == st[q].len) {
                st[cur].link = q;
            } else {
                int clone = static_cast<int>(st.size());
                st.push_back({st[p].len + 1, st[q].link, {st[q].next[0], st[q].next[1]}, {st[q].count[0], st[q].count[1]}});
                while (p != -1 && st[p].next[bit] == q) {
                    st[p].next[bit] = clone;
                    p = st[p].link;
                }
                st[q].link = st[cur].link = clone;
            }
        }
        last = cur;
        state_history.push_back(cur);
    }

    std::cout << "ISOLATED Spatial SAM for s = " << stride_bits << " bits (1 row):\n"
              << "  - Valid Evals: " << valid_evals << "\n"
              << "  - Correct Predictions: " << correct_preds << "\n"
              << "  - Accuracy: " << (double)correct_preds / valid_evals * 100.0 << "%\n"
              << "  - Free Energy: " << fe_sum / valid_evals << " nats/bit\n";

    return 0;
}
