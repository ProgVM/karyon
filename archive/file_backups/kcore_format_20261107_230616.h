// kcore_format.h
/*
===============================================================================
KARYON KCORE BINARY CONTAINER SPECIFICATION v6.0 SOVEREIGN
Weightless Native Hardware & Topological Substrate Specification (.kcore)
Standard: KEP v40.0 Sovereign Master | Principle 0, 1, 4, 5, 8, 9 & 13
Author: Bazilevs (ProgVM member) & Autonomous AI Cyberneticist (2026)

SUPREME AXIOM & PRINCIPLE 5 (WEIGHTLESS TOPOLOGICAL SUBSTRATE):
Karyon containers store NO static weight matrices ($O(D^2)$ weights are completely abolished).
The container stores strictly:
1. MANIFEST (JSON DNA Genome & Epistemic Records)
2. LOGIC_CODE_BUNDLE (Native C++20 / CUDA Hardware Kernels)
3. TOPOLOGY (Dynamic Hardware Scalar-Phase Delay Taps, Structural Wiring O(K))
4. DYNAMIC_MEMORY (Working Register States, Phase Traces & Endogenous Nodes)
===============================================================================
*/
#pragma once
#include <cstdint>

#pragma pack(push, 1)

// Magic Bytes: "KCORE" + Version 6.0 Sovereign (8 bytes total)
constexpr uint8_t KCORE_MAGIC_V6[8] = {'K', 'C', 'O', 'R', 'E', 0x06, 0x00, 0x00};

enum class KcoreSectionType : uint32_t {
    MANIFEST           = 0x01, // JSON DNA Genome, hardware parameters, epistemic telemetry
    LOGIC_CODE_BUNDLE  = 0x02, // Native C++20 / CUDA kernel source code & bytecode
    TOPOLOGY           = 0x03, // Weightless structural wiring, dynamic bit-taps O(K)
    DYNAMIC_MEMORY     = 0x04, // Dynamic register states, working pages & sprouted nodes
    EPIGENETIC_LOCK    = 0x05  // Epigenetic invariant masks protecting proven capabilities
};

enum class KcoreSectionFlags : uint32_t {
    NONE            = 0x00,
    ZLIB_COMPRESSED = 0x01, // Section payload compressed via zlib
    ALIGNED_64B     = 0x02  // Aligned for 64-byte AVX-512 / GPU warp cache lines
};

struct KcoreHeader {
    uint8_t  magic[8];        // "KCORE\x06\x00\x00"
    uint32_t header_size;     // Size of header struct (32 Bytes)
    uint32_t num_sections;    // Number of binary sections
    uint64_t total_file_size; // Total container size in bytes
    uint64_t flags;           // Execution flags (Bit 0: Weightless Substrate Active)
};

struct KcoreSectionHeader {
    uint32_t type;            // KcoreSectionType enum value
    uint32_t flags;           // KcoreSectionFlags bitmask
    uint64_t offset;          // Byte offset from start of file
    uint64_t size;            // Size of section payload in bytes
    uint64_t alignment;       // Memory alignment boundary (64 bytes)
    char     name[32];        // Human-readable section tag
};

#pragma pack(pop)
