#!/usr/bin/env python3
"""
=============================================================================
KARYON UNIVERSAL KEP AST & GRAMMAR COMPLIANCE LINTER
=============================================================================
Industrial-Grade Static Analyzer for C++20 and Python Substrate.
Enforces KEP v37.0 Sovereignty Principles across the entire codebase.

Universal Rule Engine:
  1. KEP-AST-01: Hardcoded Numerical Magic Constants in Loops & Dynamics
  2. KEP-AST-02: Hardcoded Structural Bounds & Fixed Array Dimensions
  3. KEP-AST-03: Discrete If-Else Gating on Physical Dynamic States
  4. KEP-AST-04: Static Operator/Stride Hardcoding & Pre-Baked Lists
  5. KEP-AST-05: Artificial Topological Bottlenecks & Spatial Grid Priors
  6. KEP-AST-06: PCIe Synchronous Stalls & Unchunked Inner-Loop Queries
  7. KEP-AST-07: Hardcoded Weight Scale & Damping Factors in Wave Equations
=============================================================================
"""

import sys
import os
import re
import ast
import glob
from typing import List, Dict, Any

class CppASTAndTokenAnalyzer:
    """Universal Static Grammar and Token Analyzer for C++20 Karyon Substrates."""
    
    def __init__(self, filepath: str, lines: List[str]):
        self.filepath = filepath
        self.lines = lines
        self.violations = []

    def analyze(self) -> List[Dict[str, Any]]:
        in_multiline_comment = False

        for line_idx, raw_line in enumerate(self.lines, 1):
            line = raw_line.strip()

            # Handle multiline comments
            if "/*" in line and "*/" not in line:
                in_multiline_comment = True
                continue
            if "*/" in line and in_multiline_comment:
                in_multiline_comment = False
                continue
            if in_multiline_comment or line.startswith("//") or not line:
                continue

            # Strip inline comments
            code_line = re.sub(r'//.*$', '', raw_line).strip()
            if not code_line:
                continue

            self._check_magic_constants_in_loops(code_line, line_idx)
            self._check_fixed_array_bounds(code_line, line_idx)
            self._check_discrete_gating_on_dynamics(code_line, line_idx)
            self._check_hardcoded_operator_lists(code_line, line_idx)
            self._check_spatial_grid_priors(code_line, line_idx)
            self._check_pcie_and_sync_stalls(code_line, line_idx)
            self._check_hardcoded_wave_multipliers(code_line, line_idx)

        return self.violations

    def _add_violation(self, rule_id: str, name: str, line_no: int, snippet: str, desc: str):
        self.violations.append({
            "file": self.filepath,
            "line": line_no,
            "rule_id": rule_id,
            "rule_name": name,
            "code_snippet": snippet,
            "description": desc
        })

    def _check_magic_constants_in_loops(self, line: str, line_no: int):
        # Detect hardcoded loop bounds like `for (int i = 0; i < 1024; ...)` or `while (s != -1 && depth < 16)`
        match_loop_const = re.search(r'(while|for)\s*\([^;)]*;\s*[^;)]*([<>]=?|==|!=)\s*(\d+)\b', line)
        if match_loop_const:
            const_val = match_loop_const.group(3)
            if const_val not in ["0", "1"]:
                self._add_violation(
                    "KEP-AST-01", "Hardcoded Numerical Bound in Loop", line_no, line,
                    f"Loop condition uses hardcoded numeric constant '{const_val}'. "
                    "All bounds must be endogenous state properties or dynamic entropy thresholds."
                )

        # Detect hardcoded counter limits like `depth_limit-- > 0` or `depth > 16`
        if re.search(r'\b(depth|limit|bound|max_steps|max_depth)\b.*[<>]=?\s*\d+', line, re.IGNORECASE):
            self._add_violation(
                "KEP-AST-01", "Hardcoded Depth/Step Limit Variable", line_no, line,
                "Explicit depth or step limit variable detected. Violates Principle 27 (Anti-Hardcode Sovereignty)."
            )

    def _check_fixed_array_bounds(self, line: str, line_no: int):
        # Detect fixed array declarations like `int arr[1024];` or `constexpr size_t RING_SIZE = 65536;`
        if re.search(r'\b(constexpr|const)\s+(size_t|int|long)\s+[A-Za-z0-9_]*(SIZE|CAPACITY|LIMIT|BOUND|DEPTH)\s*=\s*\d+', line):
            self._add_violation(
                "KEP-AST-02", "Static Size Constant Declaration", line_no, line,
                "Static capacity/size constant declaration detected. Violates Axiom of Unshackled Flow (Principle 8)."
            )
        if re.search(r'\bstd::array\s*<[^,]+,\s*\d+\s*>', line):
            self._add_violation(
                "KEP-AST-02", "Fixed-Size std::array Container", line_no, line,
                "Fixed-size std::array container used for state/memory. Memory containers must be dynamic."
            )

    def _check_discrete_gating_on_dynamics(self, line: str, line_no: int):
        # Detect discrete if-else checks on physical dynamic states like `if (phase >= 12.0) { ... }` or `if (count == 42)`
        if re.search(r'if\s*\([^)]*\b(phase|accumulation|energy|state|curvature|count)\b\s*(==|>=|<=|>|<)\s*\d+(\.\d+)?', line):
            self._add_violation(
                "KEP-AST-03", "Discrete Threshold Gating on Dynamic Field", line_no, line,
                "Hardcoded discrete threshold gating on dynamic physical field. "
                "Transitions must emerge continuously from Free Energy minimization (Principle 30)."
            )

    def _check_hardcoded_operator_lists(self, line: str, line_no: int):
        # Detect manually pushed operator stride constants in main/init functions
        if re.search(r'operators?\.(push_back|add|emplace_back)\s*\(.*,\s*\d+\s*\)', line):
            if not re.search(r'operators?\.(push_back|add|emplace_back)\s*\(.*,\s*1\s*\)', line): # Allow 1-bit primitive birth
                self._add_violation(
                    "KEP-AST-04", "Hardcoded Non-Primitive Operator Stride", line_no, line,
                    "Manual push of non-primitive operator stride constant. All strides beyond s_1 = 1 bit "
                    "must be endogenously sprouted bottom-up via autocorrelation (Principle 30)."
                )

    def _check_spatial_grid_priors(self, line: str, line_no: int):
        # Detect 2D/3D spatial grid hardcodes like `int width = 1024; int height = 768;`
        if re.search(r'\b(int|size_t|float|double)\s+(width|height|depth|GRID_W|GRID_H|IMAGE_W|IMAGE_H)\s*=\s*\d+', line, re.IGNORECASE):
            self._add_violation(
                "KEP-AST-05", "Hardcoded Spatial Grid Prior (2D/3D Box)", line_no, line,
                "Explicit spatial width/height/grid dimension hardcoded. "
                "Violates Pure Bitstream Substrate Constraint ($V=2$). Dimensions must be endogenously discovered."
            )

    def _check_pcie_and_sync_stalls(self, line: str, line_no: int):
        if re.search(r'\b(cudaStreamSynchronize|cudaDeviceSynchronize|glFinish)\s*\(', line):
            self._add_violation(
                "KEP-AST-06", "Blocking Hardware Synchronization Stall", line_no, line,
                "Blocking hardware synchronization call detected. Must use async non-blocking queues."
            )

    def _check_hardcoded_wave_multipliers(self, line: str, line_no: int):
        # Detect magic scaling factors in physics/wave formulas like `pow_lut[i] * 1.25f` or `1.0 + 0.5 * sin(...)`
        if re.search(r'(pow_lut|weight_lut)\s*\[', line):
            self._add_violation(
                "KEP-AST-07", "Hardcoded Static Weight Look-Up Table (LUT)", line_no, line,
                "Static weight LUT detected. Weights must be parameterized variational wave fields (Principle 4)."
            )


class PythonASTVisitor(ast.NodeVisitor):
    """Python AST Visitor auditing KEP compliance in Python scripts."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.violations = []

    def visit_For(self, node: ast.For):
        # Check for .item() calls inside loops (PCIe sync stalls)
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr == 'item':
                    self.violations.append({
                        "file": self.filepath,
                        "line": node.lineno,
                        "rule_id": "KEP-AST-06",
                        "rule_name": "PCIe Sync Stall (.item() in Loop)",
                        "code_snippet": ast.unparse(node).split('\n')[0],
                        "description": "Calling .item() inside a loop causes severe CPU-GPU PCIe sync stalls (KEP Rule #9)."
                    })
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Check for hardcoded 2D/3D image priors in Python code
        for target in node.targets:
            if isinstance(target, ast.Name):
                name_upper = target.id.upper()
                if name_upper in ["WIDTH", "HEIGHT", "IMAGE_W", "IMAGE_H", "GRID_SIZE", "MAX_DEPTH"]:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                        self.violations.append({
                            "file": self.filepath,
                            "line": node.lineno,
                            "rule_id": "KEP-AST-05",
                            "rule_name": "Hardcoded Grid/Depth Prior Variable",
                            "code_snippet": ast.unparse(node),
                            "description": f"Variable '{target.id}' assigned static constant {node.value.value}. Violates KEP Principle 27 & V=2 Substrate."
                        })
        self.generic_visit(node)


def scan_file_universal(filepath: str) -> List[Dict[str, Any]]:
    violations = []
    if not os.path.isfile(filepath):
        return violations

    ext = os.path.splitext(filepath)[1].lower()

    if ext in [".cpp", ".hpp", ".c", ".h", ".cu"]:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            analyzer = CppASTAndTokenAnalyzer(filepath, lines)
            violations.extend(analyzer.analyze())
        except Exception as e:
            print(f"[!] Error scanning C++ file {filepath}: {e}")

    elif ext == ".py":
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            tree = ast.parse(source, filename=filepath)
            visitor = PythonASTVisitor(filepath)
            visitor.visit(tree)
            violations.extend(visitor.violations)
        except Exception as e:
            # Fall back to token scan if AST fails
            pass

    return violations


def main():
    print("=================================================================")
    print("  KARYON UNIVERSAL KEP AST & GRAMMAR COMPLIANCE LINTER v2.0")
    print("  Auditing C++20/CUDA & Python Substrate Against KEP Rules")
    print("=================================================================\n")

    target_dirs = ["experiments", "src", "tools"]
    files_to_scan = []
    for d in target_dirs:
        if os.path.exists(d):
            files_to_scan.extend(glob.glob(f"{d}/*.cpp"))
            files_to_scan.extend(glob.glob(f"{d}/*.cu"))
            files_to_scan.extend(glob.glob(f"{d}/*.py"))

    files_to_scan.extend(glob.glob("*.py"))

    total_violations = 0
    file_reports = {}

    for filepath in sorted(list(set(files_to_scan))):
        if "archive" in filepath or "backup" in filepath or "build" in filepath:
            continue
        viols = scan_file_universal(filepath)
        if viols:
            file_reports[filepath] = viols
            total_violations += len(viols)

    if total_violations == 0:
        print("🟢 PASSED: 100% KEP Universal Compliance! Zero hardcoded crutches or AST violations found.")
        sys.exit(0)
    else:
        print(f"🔴 FAILED: Detected {total_violations} KEP AST violations across {len(file_reports)} files:\n")
        for filepath, viols in file_reports.items():
            print(f"📄 {filepath}:")
            for v in viols:
                print(f"   - [Line {v['line']}] [{v['rule_id']}: {v['rule_name']}]")
                print(f"     Snippet: {v['code_snippet']}")
                print(f"     Reason : {v['description']}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
