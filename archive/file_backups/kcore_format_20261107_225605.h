// kcore_format.h
/*
===============================================================================
KARYON KCORE BINARY CONTAINER SPECIFICATION v5.0
Ground Specification for Autonomous Single-File Entity Soul (.kcore)
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
*/
#pragma once
#include <cstdint>

#pragma pack(push, 1)

// Magic Bytes: "KCORE" + Version 5.0 (8 bytes total)
constexpr uint8_t KCORE_MAGIC_V5[8] = {'K', 'C', 'O', 'R', 'E', 0x05, 0x00, 0x00};

enum class KcoreSectionType : uint32_t {
    MANIFEST           = 0x01, // JSON DNA Genome, layer dimensions, SHA-256 hashes
    LOGIC_CODE_BUNDLE  = 0x02, // Dynamically Ingested C++/Python Source Files
    WEIGHTS            = 0x03, // Aligned 64-byte zero-copy tensor parameters
    STATE              = 0x04, // Dynamic persistent states (h_fast, h_slow, u_t, Memory)
    LOGIC_LLVM_BITCODE = 0x05  // LLVM IR Bitcode Binary (.bc)
};

enum class KcoreSectionFlags : uint32_t {
    NONE            = 0x00,
    ZLIB_COMPRESSED = 0x01, // Section payload compressed via zlib
    ENCRYPTED       = 0x02  // Reserved for AES payload encryption
};

struct KcoreHeader {
    uint8_t  magic[8];        // "KCORE\x05\x00\x00"
    uint32_t header_size;     // Size of header struct (32 Bytes)
    uint32_t num_sections;    // Number of binary sections (4)
    uint64_t total_file_size; // Total container size in bytes
    uint64_t flags;           // Execution flags
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
