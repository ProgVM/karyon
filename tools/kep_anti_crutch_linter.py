#!/usr/bin/env python3
"""
KEP Anti-Crutch AST & Substrate Compliance Linter
Scans C++ and Python codebase for hardcoded crutches, magic limits,
artificial boxes, and KEP Protocol violations.
"""

import sys
import os
import re
import glob

# Anti-Crutch Violation Rules
CRUTCH_RULES = [
    {
        "id": "KEP-CRUTCH-01",
        "name": "Hardcoded Suffix/Tree Depth Limit",
        "pattern": r"(depth_limit\s*=\s*\d+|max_depth\s*=\s*\d+|depth\s*>\s*\d+)",
        "description": "Hardcoded depth limits restrict context length top-down. Must use endogenous entropy thresholds."
    },
    {
        "id": "KEP-CRUTCH-02",
        "name": "Fixed-Size Memory Ring Buffer / Array Clamping",
        "pattern": r"(RING_SIZE\s*=\s*\d+|constexpr\s+size_t\s+RING|#define\s+RING_SIZE)",
        "description": "Fixed ring buffer boundaries violate Axiom of Unshackled Flow (Principle 8)."
    },
    {
        "id": "KEP-CRUTCH-03",
        "name": "Hardcoded Magic Operator / Stride Lists",
        "pattern": r"operators\.push_back\(.*,\s*(1024|3072|40960|8192)\)",
        "description": "Hardcoding operator strides in main() violates Totipotent Sprouting. Must sprout bottom-up."
    },
    {
        "id": "KEP-CRUTCH-04",
        "name": "Hardcoded Exponential Weight Lookups (pow_lut)",
        "pattern": r"(pow_lut\[.*\]\s*=|\* 1\.25f|\* 1\.5f)",
        "description": "Hardcoded weight scaling tables violate Principle 4 (Self-Evolution of Equations)."
    },
    {
        "id": "KEP-CRUTCH-05",
        "name": "PCIe Sync Stall (.item() in loop)",
        "pattern": r"for\s+.*:\s*.*\.\s*item\(",
        "description": "Calling .item() inside a loop causes PCIe sync stalls (Principle 1 & KEP Rule #9)."
    }
]

def scan_file(filepath):
    violations = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line_idx, line in enumerate(lines, 1):
                # Ignore comments
                stripped = line.strip()
                if stripped.startswith("//") or stripped.startswith("#") or stripped.startswith("/*"):
                    continue
                for rule in CRUTCH_RULES:
                    if re.search(rule["pattern"], line):
                        violations.append({
                            "file": filepath,
                            "line": line_idx,
                            "rule_id": rule["id"],
                            "rule_name": rule["name"],
                            "code_snippet": line.strip(),
                            "description": rule["description"]
                        })
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return violations

def main():
    print("=================================================================")
    print("  KARYON KEP ANTI-CRUTCH & COMPLIANCE SCANNER")
    print("=================================================================\n")

    target_dirs = ["experiments", "src"]
    files_to_scan = []
    for d in target_dirs:
        if os.path.exists(d):
            files_to_scan.extend(glob.glob(f"{d}/*.cpp"))
            files_to_scan.extend(glob.glob(f"{d}/*.py"))

    # Also scan root python files
    files_to_scan.extend(glob.glob("*.py"))

    total_violations = 0
    file_reports = {}

    for filepath in files_to_scan:
        # Skip archive backups
        if "archive" in filepath or "backup" in filepath:
            continue
        violations = scan_file(filepath)
        if violations:
            file_reports[filepath] = violations
            total_violations += len(violations)

    if total_violations == 0:
        print("🟢 PASSED: Zero KEP crutches or protocol violations detected!")
        sys.exit(0)
    else:
        print(f"🔴 FAILED: Detected {total_violations} KEP crutches across {len(file_reports)} files:\n")
        for filepath, viols in file_reports.items():
            print(f"📄 {filepath}:")
            for v in viols:
                print(f"   - [Line {v['line']}] [{v['rule_id']}: {v['rule_name']}]")
                print(f"     Snippet: {v['code_snippet']}")
                print(f"     Reason : {v['description']}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
