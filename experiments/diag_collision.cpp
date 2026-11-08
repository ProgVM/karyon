// Diagnostic to isolate the 516 bit errors in EXP-1100
#include <iostream>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <algorithm>

int main() {
    const size_t TOTAL_STEPS = 50000;
    std::vector<uint64_t> stream(TOTAL_STEPS);
    uint64_t stack[1024] = {0};
    int sp = 0;
    uint64_t accum = 0xFEEDFACECAFEBABEULL;
    uint64_t state = 1;

    for (size_t t = 0; t < TOTAL_STEPS; ++t) {
        size_t complexity_factor = 1 + (t / 3125);
        size_t num_opcodes = std::min((size_t)16, 4 + complexity_factor);
        uint64_t opcode = (t * 7) % num_opcodes;

        switch (opcode) {
            case 0: if (sp < 1000) stack[sp++] = accum ^ t; accum = (accum << 5) | (accum >> 59); break;
            case 1: if (sp > 0) accum ^= stack[--sp]; break;
            case 2: accum ^= (state * 0x9E3779B97F4A7C15ULL); break;
            case 3: state = (accum ^ (accum >> 17)) & 0x7FULL; break;
            case 4: if (sp > 1) { uint64_t tmp = stack[sp-1]; stack[sp-1] = stack[sp-2]; stack[sp-2] = tmp; } break;
            case 5: accum = ~accum + t; break;
            case 6: accum = (accum << (t % 31)) | (accum >> (64 - (t % 31))); break;
            case 7: state = (state + sp) & 0xFFULL; break;
            case 8: if (sp > 0) accum += stack[sp - 1] * 3; break;
            case 9: accum ^= 0x55AA55AA55AA55AAULL; break;
            case 10: accum = accum * 6364136223846793005ULL + 1442695040888963407ULL; break;
            case 11: if (sp < 1000) stack[sp++] = state; break;
            case 12: accum ^= (accum >> 23); break;
            case 13: state ^= (accum & 0xFFFFULL); break;
            case 14: accum = (accum >> 11) | (accum << 53); break;
            case 15: accum ^= (t * 0xBF58476D1CE4E5B9ULL); break;
        }

        uint64_t word = ((state & 0xFFFFULL) << 48) | (((uint64_t)sp & 0xFFFFULL) << 32) | (accum & 0xFFFFFFFFULL);
        stream[t] = word;
    }

    std::unordered_map<uint64_t, uint64_t> sprouted;
    std::unordered_map<uint64_t, std::vector<size_t>> sig_to_timesteps;

    uint64_t w[8] = {0};
    uint64_t phase = 0;

    auto compute_sig = [](const uint64_t w[8], uint64_t p) -> uint64_t {
        uint64_t h = p * 0xCBF29CE484222325ULL;
        h ^= w[0] * 0x9E3779B97F4A7C15ULL; h = (h << 31) | (h >> 33);
        h ^= w[1] * 0xBF58476D1CE4E5B9ULL; h = (h << 27) | (h >> 37);
        h ^= w[2] * 0x94D049BB133111EBULL; h = (h << 41) | (h >> 23);
        h ^= w[3] * 0xA24BAED4963EE407ULL; h = (h << 19) | (h >> 45);
        h ^= w[4] * 0x9E3779B185EBCA87ULL; h = (h << 37) | (h >> 27);
        h ^= w[5] * 0xC2B2AE3D27D4EB4FULL; h = (h << 13) | (h >> 51);
        h ^= w[6] * 0x855DD4DB2F49D461ULL; h = (h << 47) | (h >> 17);
        h ^= w[7] * 0x1B87359328224531ULL;
        h ^= (h >> 33); h *= 0xFF51AFD7ED558CCDULL; h ^= (h >> 33);
        return h;
    };

    // Pass 1
    for (size_t t = 0; t < TOTAL_STEPS; ++t) {
        uint64_t sig = compute_sig(w, phase);
        sprouted[sig] = stream[t];
        sig_to_timesteps[sig].push_back(t);

        for (int i = 7; i > 0; --i) w[i] = w[i - 1];
        w[0] = stream[t];
        phase = ((phase << 7) | (phase >> 57)) ^ (stream[t] * 0x9E3779B97F4A7C15ULL);
    }

    // Inspect collisions
    size_t collisions = 0;
    for (const auto& kv : sig_to_timesteps) {
        if (kv.second.size() > 1) {
            // Check if targets were different
            uint64_t first_target = stream[kv.second[0]];
            bool diff_target = false;
            for (size_t idx : kv.second) {
                if (stream[idx] != first_target) {
                    diff_target = true;
                    break;
                }
            }
            if (diff_target) {
                collisions++;
                std::cout << "Collision at sig: 0x" << std::hex << kv.first << std::dec 
                          << " across " << kv.second.size() << " steps: ";
                for (size_t idx : kv.second) std::cout << idx << " ";
                std::cout << std::endl;
            }
        }
    }
    std::cout << "Total divergent collisions: " << collisions << std::endl;

    return 0;
}
