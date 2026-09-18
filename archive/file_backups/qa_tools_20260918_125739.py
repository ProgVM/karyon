# karyon_agent_runtime/tools/qa_tools.py
"""
===============================================================================
CODE QUALITY ASSURANCE, STATIC LINTING & PRE-FLIGHT VERIFICATION ENGINE (v29.0)
Features Programmatic AST Analysis, Byte-Code Compilation Checks, Flake8 Linter
Runner with Error Grouping, and Automated Test Suite Execution (pytest/unittest).
Author: Bazilevs (ProgVM) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import ast
import py_compile
import subprocess
import time
import karyon_agent_runtime.config as config


async def verify_code_syntax(filepath_or_dir: str = ".") -> str:
    """
    Performs deep AST parsing and byte-code compilation verification on Python files.
    Detects SyntaxError, IndentationError, and unclosed delimiters before execution or git commit.

    Args:
        filepath_or_dir: Specific file path or directory to recursively audit (default: '.').
    """
    root_path = (config.PROJECT_ROOT / filepath_or_dir).resolve()
    if not root_path.exists():
        return f"Error: Target '{filepath_or_dir}' does not exist on disk."

    targets = []
    if root_path.is_file():
        if root_path.suffix == ".py":
            targets.append(root_path)
    else:
        for p in root_path.rglob("*.py"):
            rel_str = str(p.relative_to(config.PROJECT_ROOT))
            if any(ign in rel_str for ign in [".git", "__pycache__", "build", "agent_data", "patch_backups"]):
                continue
            targets.append(p)

    if not targets:
        return f"Notice: No Python files found in '{filepath_or_dir}'."

    errors = []
    verified = 0

    for t in targets:
        rel = str(t.relative_to(config.PROJECT_ROOT))
        try:
            code = t.read_text(encoding="utf-8")
            ast.parse(code, filename=rel)
            py_compile.compile(str(t), doraise=True)
            verified += 1
        except SyntaxError as syn_err:
            errors.append(f"• **`{rel}:{syn_err.lineno}`**: SyntaxError: {syn_err.msg}")
        except Exception as ex:
            errors.append(f"• **`{rel}`**: Verification Error: {str(ex)}")

    if errors:
        header = f"❌ **Syntax & Compilation Verification Failed** ({len(errors)} issues found across {len(targets)} files):"
        return header + "\n\n" + "\n".join(errors[:30])

    return f"✅ **100% Verified**: {verified} Python file(s) in '{filepath_or_dir}' are syntactically valid and compilation-clean."


async def run_code_linter(path: str = ".", strict: bool = False, max_issues: int = 50) -> str:
    """
    Runs flake8 linter with project configuration, grouping critical bugs (F821, F841) and style issues.
    Falls back to AST static analysis if flake8 is not installed.

    Args:
        path: Relative path or directory to lint (default: '.').
        strict: If True, treats style warnings as errors.
        max_issues: Maximum issues to report in output (default: 50).
    """
    target_dir = (config.PROJECT_ROOT / path).resolve()
    if not target_dir.exists():
        return f"Error: Path '{path}' not found."

    max_len = "100" if strict else "160"
    cmd = ["flake8", f"--max-line-length={max_len}", "--ignore=E501,W503,W504,E226", str(target_dir)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0)
        output = proc.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return await verify_code_syntax(path)

    if not output:
        return f"🎉 **Flake8 Lint Passed**: Zero issues detected in `{path}`."

    lines = output.splitlines()
    critical_issues = []
    other_issues = []

    for line in lines:
        if any(code in line for code in ["F821", "F824", "F841", "E999", "SyntaxError"]):
            critical_issues.append(line)
        else:
            other_issues.append(line)

    report = [f"=== Flake8 Lint Report for `{path}` ({len(lines)} total notices) ==="]
    if critical_issues:
        report.append(f"\n⚠️ **Critical Undefined Names / Variables ({len(critical_issues)}):**")
        for ci in critical_issues[:max_issues]:
            report.append(f"  • {ci}")

    if other_issues and not critical_issues:
        report.append(f"\nℹ️ **Style & Formatting Notices ({len(other_issues)}):**")
        for oi in other_issues[:max_issues]:
            report.append(f"  • {oi}")

    return "\n".join(report)


async def run_unit_tests(test_path: str = "tests", verbose: bool = True, timeout_seconds: float = 120.0) -> str:
    """
    Executes automated test suites (pytest or unittest discovery) and reports pass/fail metrics.

    Args:
        test_path: Target directory or test file (default: 'tests').
        verbose: If True, adds detailed assertion output (default: True).
        timeout_seconds: Maximum test execution timeout (default: 120.0s).
    """
    target = (config.PROJECT_ROOT / test_path).resolve()
    t0 = time.perf_counter()

    cmd = ["pytest", "-q" if not verbose else "-v", str(target)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, cwd=str(config.PROJECT_ROOT))
        duration = time.perf_counter() - t0
        status_tag = "✅ PASSED" if proc.returncode == 0 else f"❌ FAILED (Exit Code: {proc.returncode})"
        output_txt = proc.stdout.strip() or proc.stderr.strip()
        return (
            f"=== Test Suite Result: {status_tag} ({duration:.2f}s) ===\n\n"
            f"```text\n{output_txt}\n```"
        )
    except FileNotFoundError:
        cmd_fallback = ["python", "-m", "unittest", "discover", "-s", str(test_path)]
        proc = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=timeout_seconds, cwd=str(config.PROJECT_ROOT))
        duration = time.perf_counter() - t0
        output_txt = proc.stdout.strip() or proc.stderr.strip()
        return (
            f"=== Unittest Fallback Result (Exit: {proc.returncode} | {duration:.2f}s) ===\n\n"
            f"```text\n{output_txt}\n```"
        )
    except Exception as ex:
        return f"Error executing test suite: {str(ex)}"
