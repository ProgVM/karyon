#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

struct SAMState {
    int len = 0;
    int link = -1;
    int next[2] = {-1, -1};
    uint32_t count[2] = {0, 0};
};

class StrideSAM {
public:
    int stride_bits;
    std::vector<SAMState> st;
    int last = 0;
    std::vector<int> state_history;

    StrideSAM(int stride) : stride_bits(stride) {
        st.push_back({0, -1, {-1, -1}, {0, 0}});
        last = 0;
    }

    float predict_bit(size_t current_t, int& out_len) const {
        out_len = 0;
        if (current_t < static_cast<size_t>(stride_bits)) {
            return 0.5f;
        }
        size_t query_t = current_t - stride_bits;
        if (query_t >= state_history.size()) return 0.5f;

        int s = state_history[query_t];
        while (s != -1) {
            uint32_t c0 = st[s].count[0];
            uint32_t c1 = st[s].count[1];
            uint32_t tot = c0 + c1;
            if (tot > 0) {
                out_len = st[s].len;
                if (c0 > 0 && c1 == 0) return 0.000001f;
                if (c1 > 0 && c0 == 0) return 0.999999f;
                return static_cast<float>(c1) / static_cast<float>(tot);
            }
            s = st[s].link;
        }
        return 0.5f;
    }

    void extend_bit(size_t current_t, uint8_t bit) {
        // Update prediction count of state at t - stride_bits
        if (current_t >= static_cast<size_t>(stride_bits)) {
            size_t query_t = current_t - stride_bits;
            if (query_t < state_history.size()) {
                int s = state_history[query_t];
                while (s != -1) {
                    st[s].count[bit]++;
                    if (st[s].count[0] > 0 && st[s].count[1] > 0) {
                        s = st[s].link;
                    } else {
                        break;
                    }
                }
            }
        }

        // Extend SAM with 1D sequential bit to build context state at t
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
};

int main() {
    std::cout << "Testing StrideSAM logic...\n";
    StrideSAM sam(8); // 8-bit stride
    std::vector<uint8_t> bits = {1,0,1,1,0,0,1,1,  1,0,1,1,0,0,1,1,  1,0,1,1,0,0,1,1};
    for (size_t t = 0; t < bits.size(); ++t) {
        int len;
        float p1 = sam.predict_bit(t, len);
        std::cout << "t=" << t << " bit=" << (int)bits[t] << " pred_p1=" << p1 << " len=" << len << "\n";
        sam.extend_bit(t, bits[t]);
    }
    return 0;
}
