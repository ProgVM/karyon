// kcore_format.h
/*
===============================================================================
KARYON KCORE BINARY CONTAINER SPECIFICATION v7.0 SOVEREIGN
Native Hardware Bit Substrate & Code Architecture Specification (.kcore)
Standard: KEP v40.0 Sovereign Master | Principle 0, 1, 3, 4, 5, 8, 9 & 13
Author: Bazilevs (ProgVM member) & Autonomous AI Cyberneticist (2026)

SOVEREIGN AXIOM (BAZILEVS' SPECIFICATION):
The `.kcore` container encapsulates strictly the essence of the entity:
1. MANIFEST           (0x01): JSON DNA Genome, Hardware Limits & Empirical Records
2. BIT_SUBSTRATE      (0x02): Pure Native Bit Substrate (V=2, Hardware Bit-Taps & Topological Registers)
3. LOGIC_CODE_BUNDLE  (0x03): Native C++20 / CUDA Hardware Execution Kernels & Bytecode
4. STORAGE            (0x04): Optional Auxiliary Artifact/Knowledge Storage (Decoupled from Substrate)

NOTE: Memory logic and dynamic node sprouting reside 100% INSIDE the substrate execution loop
itself (endogenous memory morphogenesis), decoupled from static container schemas.
===============================================================================
*/
#pragma once
#include <cstdint>

#pragma pack(push, 1)

// Magic Bytes: "KCORE" + Version 7.0 Sovereign (8 bytes total)
constexpr uint8_t KCORE_MAGIC_V7[8] = {'K', 'C', 'O', 'R', 'E', 0x07, 0x00, 0x00};

enum class KcoreSectionType : uint32_t {
    MANIFEST           = 0x01, // JSON DNA Genome, hardware parameters, epistemic telemetry
    BIT_SUBSTRATE      = 0x02, // Pure Native Bit Substrate (V=2), Bit-Tap Configurations
    LOGIC_CODE_BUNDLE  = 0x03, // Native C++20 / CUDA Execution Kernel Sources & Bytecode
    STORAGE            = 0x04  // Optional Auxiliary External Storage (Knowledge/Artifacts, Decoupled)
};

enum class KcoreSectionFlags : uint32_t {
    NONE            = 0x00,
    ZLIB_COMPRESSED = 0x01, // Section payload compressed via zlib
    ALIGNED_64B     = 0x02  // Aligned for 64-byte AVX-512 / GPU warp cache lines
};

struct KcoreHeader {
    uint8_t  magic[8];        // "KCORE\x07\x00\x00"
    uint32_t header_size;     // Size of header struct (32 Bytes)
    uint32_t num_sections;    // Number of binary sections
    uint64_t total_file_size; // Total container size in bytes
    uint64_t flags;           // Execution flags (Bit 0: Native Bit Substrate Active)
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
