#include <iostream>
#include <x86intrin.h>

int main() {
    uint64_t tsc1 = __rdtsc();
    uint64_t tsc2 = __rdtsc();
    std::cout << "tsc1: " << tsc1 << " tsc2: " << tsc2 << " delta: " << (tsc2 - tsc1) << std::endl;

    // Check rdrand
    unsigned long long val = 0;
    int success = _rdrand64_step(&val);
    std::cout << "rdrand support: " << success << " val: 0x" << std::hex << val << std::dec << std::endl;

    return 0;
}
