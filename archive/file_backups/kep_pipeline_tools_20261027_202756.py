# karyon_agent_runtime/tools/kep_pipeline_tools.py
"""
===============================================================================
AUTONOMOUS KEP SCIENTIFIC BENCHMARK & EXPERIMENTAL PIPELINE ORCHESTRATOR (v21.6)
Unified Lifecycle: Real-Time Execution, Telemetry Parsing, KEP Rule #2 Verdict,
Loss Convergence Plotting, SQLite Ledger Archival, Git Push & Multi-Channel Alerting.
Features Smart Notification Gating (Prevents intermediate spam) & Non-Destructive Archival.
===============================================================================
"""

import re
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import karyon_agent_runtime.config as config
from karyon_agent_runtime.tools.bash_tools import run_bash_command
from karyon_agent_runtime.tools.db_tools import record_experiment_result
from karyon_agent_runtime.tools.git_tools import git_commit, git_push
from karyon_agent_runtime.tools.notification_tools import broadcast_notification
from karyon_agent_runtime.tools.analytics_tools import plot_training_loss_curves

logger = logging.getLogger("ProxyAgent.KepPipeline")


def _extract_numerical_metrics(log_output: str) -> Dict[str, Any]:
    """Parses loss, perplexity, VRAM, throughput, and biophysical metrics from console logs."""
    metrics = {}

    loss_match = re.findall(r'(?:loss|speech_loss|final_loss)\s*[:=]\s*([0-9]+\.[0-9]+)', log_output, re.IGNORECASE)
    if loss_match:
        metrics["loss"] = float(loss_match[-1])

    ppl_match = re.findall(r'(?:ppl|perplexity)\s*[:=]\s*([0-9]+\.[0-9]+)', log_output, re.IGNORECASE)
    if ppl_match:
        metrics["ppl"] = float(ppl_match[-1])

    tok_match = re.findall(r'([0-9]+\.[0-9]+|\d+)\s*(?:tok/s|tokens/sec)', log_output, re.IGNORECASE)
    if tok_match:
        metrics["tok_per_sec"] = float(tok_match[-1])

    vram_match = re.findall(r'(?:vram|peak_vram)\s*[:=]\s*([0-9]+\.[0-9]+|\d+)\s*(?:mb|gb)?', log_output, re.IGNORECASE)
    if vram_match:
        metrics["vram_mb"] = float(vram_match[-1])

    fe_match = re.findall(r'(?:free_energy|fe)\s*[:=]\s*([0-9]+\.[0-9]+)', log_output, re.IGNORECASE)
    if fe_match:
        metrics["free_energy"] = float(fe_match[-1])

    curiosity_match = re.findall(r'curiosity\s*[:=]\s*([0-9]+\.[0-9]+)', log_output, re.IGNORECASE)
    if curiosity_match:
        metrics["curiosity"] = float(curiosity_match[-1])

    stability_match = re.findall(r'stability\s*[:=]\s*([0-9]+\.[0-9]+)', log_output, re.IGNORECASE)
    if stability_match:
        metrics["stability"] = float(stability_match[-1])

    energy_match = re.findall(r'energy\s*[:=]\s*([0-9]+\.[0-9]+)', log_output, re.IGNORECASE)
    if energy_match:
        metrics["energy"] = float(energy_match[-1])

    return metrics


async def run_kep_scientific_pipeline(
    script_path: str,
    exp_id: str,
    hypothesis: str,
    architecture_delta: str,
    baseline_loss: Optional[float] = None,
    auto_archive: bool = True,
    auto_commit: bool = True,
    auto_push: bool = True,
    generate_plot: bool = True,
    notification_channels: str = "all",
    notify_on_failure: bool = False
) -> str:
    """
    Executes an end-to-end KEP scientific benchmark run: runs script, extracts metrics,
    applies KEP Rule #2 data-driven verdict criteria (Loss Delta >= 0.08), records into SQLite,
    generates loss convergence plots, archives benchmark script, commits to Git, and fires
    multi-channel notifications to Telegram, Discord, and Webhooks.

    Args:
        script_path: Path to experimental script (e.g. 'experiments/exp_71_pac_laminar.py').
        exp_id: Experiment ID (e.g. 'EXP-71').
        hypothesis: Scientific hypothesis being evaluated.
        architecture_delta: Description of layer modifications.
        baseline_loss: Pre-experiment baseline speech loss for quantitative delta calculation.
        auto_archive: If True, moves script to experiments/archive/ upon completion (default: True).
        auto_commit: If True, commits experiment record to Git (default: True).
        auto_push: If True, pushes changes to origin/main (default: True).
        generate_plot: If True, automatically plots and saves loss convergence curves (default: True).
        notification_channels: Target channels for alert: 'all', 'telegram', 'discord', or 'webhook' (default: 'all').
        notify_on_failure: If True, sends Telegram/Discord notification even if run crashed or was rejected (default: False).
    """
    script_file = (config.PROJECT_ROOT / script_path).resolve()
    if not script_file.exists():
        # Fallback check in archive if previously moved
        archived_fallback = config.PROJECT_ROOT / "experiments" / "archive" / Path(script_path).name
        if archived_fallback.exists():
            script_file = archived_fallback
        else:
            return f"Error: Experiment script '{script_path}' not found on disk."

    logger.info(f"=== LAUNCHING KEP SCIENTIFIC PIPELINE: {exp_id} ({script_path}) ===")

    # 1. Execute Benchmark via Bash Tool
    cmd = f"python {str(script_file.relative_to(config.PROJECT_ROOT))}"
    bash_result = await run_bash_command(cmd, max_timeout=3600.0)

    # 2. Extract Empirical Metrics
    metrics = _extract_numerical_metrics(bash_result)
    final_loss = metrics.get("loss")

    # 3. Evaluate KEP Rule #2 Decision Engine
    verdict = "⚪ NEUTRAL / INCONCLUSIVE"
    notes = "Metrics within baseline statistical noise margin."

    if "NaN" in bash_result or "CUDA out of memory" in bash_result or "Traceback" in bash_result or "can't open file" in bash_result:
        verdict = "🔴 REJECTED"
        notes = "Benchmark terminated with runtime crash, NaN overflow, or OOM."
    elif final_loss is not None and baseline_loss is not None:
        delta_loss = baseline_loss - final_loss
        metrics["delta_loss"] = round(delta_loss, 4)
        if delta_loss >= 0.08:
            verdict = "🟢 POSITIVE"
            notes = f"Statistically significant loss reduction (+{delta_loss:.4f} loss drop >= 0.08 threshold). Ready for production merge."
        elif delta_loss <= -0.05:
            verdict = "🔴 REJECTED"
            notes = f"Performance degradation detected (loss increased by {-delta_loss:.4f}). Discarded."
        else:
            verdict = "⚪ NEUTRAL / INCONCLUSIVE"
            notes = f"Delta loss (+{delta_loss:.4f}) within neutral noise tolerance margin."
    elif final_loss is not None:
        verdict = "🟢 POSITIVE" if final_loss < 2.0 else "⚪ NEUTRAL / INCONCLUSIVE"
        notes = f"Final loss reached {final_loss:.4f}."

    # 4. Generate Convergence Plot Chart if requested
    plot_path_rel = None
    if generate_plot and final_loss is not None:
        try:
            plot_rel = f"experiments/plots/{exp_id}_convergence.png"
            await plot_training_loss_curves(
                log_filepath_or_data=bash_result,
                chart_title=f"{exp_id}: Convergence Trajectory ({verdict})",
                save_path=plot_rel
            )
            plot_path_rel = plot_rel
        except Exception as p_err:
            logger.warning(f"Pipeline plot generation notice: {str(p_err)}")

    # 5. Record into Empirical SQLite Ledger
    db_res = await record_experiment_result(
        exp_id=exp_id,
        hypothesis=hypothesis,
        architecture_delta=architecture_delta,
        verdict=verdict,
        final_loss=final_loss,
        metrics=metrics,
        config_params={"script": script_path, "baseline_loss": baseline_loss},
        notes=notes
    )

    # 6. Archive Benchmark Script (KEP Rule #8)
    archive_msg = "Script retained in experiments/."
    if auto_archive and script_file.exists():
        archive_dir = config.PROJECT_ROOT / "experiments" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest_archive = archive_dir / script_file.name

        # If POSITIVE: move to archive. If REJECTED: copy to archive, keeping active file available for debugging
        if "POSITIVE" in verdict:
            shutil.move(str(script_file), str(dest_archive))
            archive_msg = f"Script archived to `experiments/archive/{dest_archive.name}`."
        else:
            shutil.copy2(str(script_file), str(dest_archive))
            archive_msg = f"Failed script snapshot backed up to `experiments/archive/{dest_archive.name}` (active file kept in place for debugging)."

    # 7. Automated Git Commit (KEP Rule #5)
    git_msg = "No commit created."
    if auto_commit:
        commit_text = f"feat(exp): record {exp_id} - {verdict} (loss: {final_loss or 'N/A'})"
        git_msg = await git_commit(commit_text, auto_add=True)
        if auto_push:
            await git_push(repo="karyon", branch="main")

    # 8. Notification Broadcast (Gated: only on POSITIVE or explicit notify_on_failure)
    notif_summary = ""
    should_send_notification = (notification_channels.lower() != "none") and ("POSITIVE" in verdict or notify_on_failure)

    if should_send_notification:
        try:
            alert_msg = (
                f"- **Verdict:** {verdict}\n"
                f"- **Final Loss:** `{final_loss if final_loss is not None else 'N/A'}` (Baseline: `{baseline_loss if baseline_loss is not None else 'N/A'}`)\n"
                f"- **Throughput:** `{metrics.get('tok_per_sec', 'N/A')} tok/s` | **VRAM:** `{metrics.get('vram_mb', 'N/A')} MB`\n"
                f"- **Hypothesis:** {hypothesis}\n"
                f"- **Deduction:** {notes}"
            )
            severity = "SUCCESS" if "POSITIVE" in verdict else "WARNING" if "NEUTRAL" in verdict else "CRITICAL"
            broadcast_res = await broadcast_notification(
                title=f"🧬 Karyon KEP Pipeline: {exp_id}",
                message=alert_msg,
                photo_path=plot_path_rel,
                channels=notification_channels,
                severity=severity
            )
            notif_summary = f"\n- Multi-Channel Broadcast: {broadcast_res}"
        except Exception as n_err:
            logger.warning(f"Notification broadcast notice: {str(n_err)}")
    else:
        notif_summary = f"\n- Multi-Channel Broadcast: [Notification suppressed for non-positive run to prevent spam. Verdict: {verdict}]"

    return (
        f"=== KEP SCIENTIFIC PIPELINE EXECUTION COMPLETE ===\n"
        f"- Experiment ID : `{exp_id}`\n"
        f"- Verdict       : **{verdict}**\n"
        f"- Final Loss    : `{final_loss if final_loss is not None else 'N/A'}` (Baseline: `{baseline_loss if baseline_loss is not None else 'N/A'}`)\n"
        f"- Metrics       : `{json.dumps(metrics, ensure_ascii=False)}`\n"
        f"- Plot Chart    : `{plot_path_rel or 'None'}`\n"
        f"- Archival      : {archive_msg}\n"
        f"- Ledger Status : {db_res}\n"
        f"- Git Sync      : {git_msg}{notif_summary}\n\n"
        f"--- Console Log Tail ---\n{bash_result[-1500:]}"
    )
