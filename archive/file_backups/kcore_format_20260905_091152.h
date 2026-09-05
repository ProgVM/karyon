// kcore_format.h
#pragma once
#include <cstdint>

#pragma pack(push, 1)

// Magic Bytes: "KCORE" + Version 2.0 LLVM Enabled (8 bytes total)
constexpr uint8_t KCORE_MAGIC[8] = {'K', 'C', 'O', 'R', 'E', 0x02, 0x00, 0x00};

enum class KcoreSectionType : uint32_t {
    MANIFEST           = 0x01, // JSON DNA Genome, layer dimensions
    LOGIC_CPP_SOURCE   = 0x02, // Legacy C++ Source Code Text
    WEIGHTS            = 0x03, // Aligned model parameter weights
    STATE              = 0x04, // Dynamic persistent states (h_fast, h_slow, u_t, Memory)
    LOGIC_LLVM_BITCODE = 0x05  // LLVM IR Bitcode Binary (.bc)
};

struct KcoreHeader {
    uint8_t  magic[8];        // "KCORE\x02\x00\x00"
    uint32_t header_size;     // Size of header struct (32 Bytes)
    uint32_t num_sections;    // Number of binary sections (4)
    uint64_t total_file_size; // Total container size in bytes
    uint64_t flags;           // Execution flags
};

struct KcoreSectionHeader {
    uint32_t type;            // KcoreSectionType enum value
    uint32_t flags;           // Section flags (compression, encryption)
    uint64_t offset;          // Byte offset from start of file
    uint64_t size;            // Size of section payload in bytes
    uint64_t alignment;       // Memory alignment boundary (64 bytes)
    char     name[32];        // Human-readable section tag
};

#pragma pack(pop)
