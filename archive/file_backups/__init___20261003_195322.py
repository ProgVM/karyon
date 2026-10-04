# karyon_agent_runtime/tools/__init__.py
"""
===============================================================================
KARYON CORE UNIFIED PRODUCTION TOOLKIT REGISTRY (183 PRODUCTION TOOLS MASTER)
Consolidates Gemini Engine, Native C/C++ & Python Execution, Shebang Scripts,
Expanded Dual-Repo Git Suite, Complete Hugging Face Suite, Hardware Telemetry,
Safe POSIX Process Engine, Multi-Channel Notifications, Configuration Suite,
QA Verification Suite, and Hierarchical Multi-Agent Cortical Orchestration.
===============================================================================
"""

from karyon_agent_runtime.tools.file_tools import (
    read_file,
    write_file,
    list_directory,
    search_codebase,
    find_files,
    delete_file_or_dir,
    make_directory,
    move_or_rename_file,
    get_codebase_tree,
    diff_files,
    batch_replace_text
)
from karyon_agent_runtime.tools.git_tools import (
    git_status,
    git_diff,
    git_log,
    git_add,
    git_commit,
    git_push,
    git_pull,
    git_fetch,
    git_branch,
    git_checkout,
    git_show,
    git_stash,
    git_rollback,
    git_blame,
    git_cherry_pick,
    git_tag,
    git_merge,
    git_clean,
    git_remote,
    git_sync_hard_reset,
    git_create_patch,
    git_apply_patch
)
from karyon_agent_runtime.tools.system_tools import (
    get_tpu_hardware_telemetry,
    benchmark_tpu_performance,
    clear_tpu_memory_cache,
    audit_tpu_workload,
    get_cpu_telemetry,
    benchmark_cpu_performance,
    get_ram_telemetry,
    audit_process_memory,
    optimize_ram_and_clean_caches,
    get_storage_telemetry,
    benchmark_storage_io,
    clean_disk_storage,
    get_gpu_hardware_telemetry,
    get_cuda_memory_summary,
    clear_cuda_cache,
    get_system_info,
    list_active_processes
)
from karyon_agent_runtime.tools.web_tools import (
    internet_search,
    internet_media_search,
    internet_deep_search,
    scrape_url,
    download_file_from_url,
    send_http_request
)
from karyon_agent_runtime.tools.agent_message_tools import (
    send_agent_message
)
from karyon_agent_runtime.tools.control_tools import (
    ignore_or_noop,
    sleep_delay
)
from karyon_agent_runtime.tools.media_tools import (
    embed_media
)
from karyon_agent_runtime.tools.config_tools import (
    update_runtime_config,
    reset_key_cooldowns,
    get_key_pool_telemetry,
    get_runtime_config,
    force_rotate_key,
    force_rotate_model
)
from karyon_agent_runtime.tools.code_execution_tools import (
    execute_python_code,
    list_python_sessions,
    reset_python_session,
    execute_cpp_code,
    execute_code_with_shebang,
    poll_code_job,
    tail_code_job,
    get_code_job_logs,
    list_code_jobs,
    kill_code_job,
    poll_python_job,
    tail_python_job,
    list_python_jobs,
    kill_python_job
)
from karyon_agent_runtime.tools.bash_tools import (
    run_bash_command,
    poll_bash_job,
    tail_bash_job,
    get_bash_job_logs,
    list_bash_jobs,
    kill_bash_job,
    kill_all_bash_jobs,
    clean_stale_bash_jobs,
    clear_background_jobs_history
)
from karyon_agent_runtime.tools.tool_runner_tools import (
    execute_tools_sequential,
    execute_tools_parallel,
    run_tool_in_background,
    poll_tool_job,
    tail_tool_job,
    get_tool_job_logs,
    list_tool_jobs,
    kill_tool_job,
    kill_all_tool_jobs,
    clean_stale_tool_jobs,
    clear_tool_jobs_history
)
from karyon_agent_runtime.tools.custom_tool_tools import (
    create_custom_tool,
    update_custom_tool,
    test_custom_tool,
    list_custom_tools,
    inspect_custom_tool,
    delete_custom_tool
)
from karyon_agent_runtime.tools.google_file_tools import (
    upload_file_to_google
)
from karyon_agent_runtime.tools.db_tools import (
    sync_agent_database,
    list_database_backups,
    create_database_snapshot,
    rename_database_snapshot,
    delete_database_snapshot,
    get_database_info,
    switch_database,
    restore_database_backup,
    record_experiment_result,
    get_experiment_history,
    get_latest_experiment,
    set_agent_memory,
    get_agent_memory,
    list_agent_memory,
    delete_agent_memory,
    search_persistent_memory,
    vacuum_and_optimize_database,
    prune_and_vacuum_database,
    export_empirical_database_json
)
from karyon_agent_runtime.tools.master_doc_tools import (
    list_master_documents,
    read_master_document,
    write_master_document,
    diff_master_documents,
    archive_master_document,
    validate_kep_compliance,
    create_new_master_version,
    export_master_bundle
)
from karyon_agent_runtime.tools.hf_tools import (
    hf_hub_status,
    hf_upload_model_file,
    hf_upload_folder,
    hf_download_model_file,
    hf_download_snapshot,
    hf_list_repo_files,
    hf_create_repo,
    hf_generate_karyon_model_card,
    hf_search_hub,
    hf_inspect_dataset,
    hf_download_dataset_sample,
    hf_delete_file,
    hf_delete_repo,
    hf_set_repo_visibility,
    hf_create_tag,
    hf_upload_files_atomic
)
from karyon_agent_runtime.tools.notification_tools import (
    send_telegram_notification,
    send_discord_notification,
    send_webhook_alert,
    broadcast_notification
)
from karyon_agent_runtime.tools.analytics_tools import (
    plot_experiment_comparison,
    plot_training_loss_curves,
    plot_phase_space_portrait,
    export_empirical_ledger,
    generate_empirical_report
)
from karyon_agent_runtime.tools.arxiv_tools import (
    arxiv_search_papers,
    arxiv_get_paper_details,
    arxiv_download_pdf
)
from karyon_agent_runtime.tools.cpp_build_tools import (
    build_and_verify_cpp_core,
    inspect_cuda_environment,
    benchmark_cpp_kernel,
    clean_cpp_build_cache
)
from karyon_agent_runtime.tools.kep_pipeline_tools import (
    run_kep_scientific_pipeline
)
from karyon_agent_runtime.tools.context_tools import (
    get_context_token_status,
    compress_context_now
)
from karyon_agent_runtime.tools.qa_tools import (
    verify_code_syntax,
    run_code_linter,
    run_unit_tests
)
from karyon_agent_runtime.tools.multi_agent_tools import (
    transfer_turn_to_agent,
    dispatch_parallel_agent_tasks,
    create_subagent,
    update_subagent,
    delete_subagent,
    list_subagents,
    get_subagent_info,
    dispatch_agent_task,
    send_agent_message_to,
    get_agent_inbox,
    get_agent_dialogue_history,
    get_agent_task_status,
    spawn_ephemeral_subagent,
    get_swarm_telemetry
)

AGENT_TOOLS = [
    # 1. Flow Control & Turn Messaging (3 tools)
    send_agent_message,
    ignore_or_noop,
    sleep_delay,

    # 2. Consolidated Code Execution & Process Management (14 tools)
    execute_python_code,
    list_python_sessions,
    reset_python_session,
    execute_cpp_code,
    execute_code_with_shebang,
    poll_code_job,
    tail_code_job,
    get_code_job_logs,
    list_code_jobs,
    kill_code_job,
    poll_python_job,
    tail_python_job,
    list_python_jobs,
    kill_python_job,

    # 3. Persistent Bash Execution Suite (9 tools)
    run_bash_command,
    poll_bash_job,
    tail_bash_job,
    get_bash_job_logs,
    list_bash_jobs,
    kill_bash_job,
    kill_all_bash_jobs,
    clean_stale_bash_jobs,
    clear_background_jobs_history,

    # 4. Runtime Configuration, 250k TPM Matrix & Key Management (6 tools)
    update_runtime_config,
    reset_key_cooldowns,
    get_key_pool_telemetry,
    get_runtime_config,
    force_rotate_key,
    force_rotate_model,

    # 5. Context Token Capacity & Compaction (2 tools)
    get_context_token_status,
    compress_context_now,

    # 6. Multimedia, Plots & Album Rendering (1 tool)
    embed_media,

    # 7. Master Architectural Docs & KEP Compliance (8 tools)
    list_master_documents,
    read_master_document,
    write_master_document,
    diff_master_documents,
    archive_master_document,
    validate_kep_compliance,
    create_new_master_version,
    export_master_bundle,

    # 8. Expanded Hugging Face Hub (16 tools)
    hf_hub_status,
    hf_upload_model_file,
    hf_upload_folder,
    hf_download_model_file,
    hf_download_snapshot,
    hf_list_repo_files,
    hf_create_repo,
    hf_generate_karyon_model_card,
    hf_search_hub,
    hf_inspect_dataset,
    hf_download_dataset_sample,
    hf_delete_file,
    hf_delete_repo,
    hf_set_repo_visibility,
    hf_create_tag,
    hf_upload_files_atomic,

    # 9. Multi-Channel Notifications (4 tools)
    send_telegram_notification,
    send_discord_notification,
    send_webhook_alert,
    broadcast_notification,

    # 10. Analytics & Biophysical Plotter (5 tools)
    plot_experiment_comparison,
    plot_training_loss_curves,
    plot_phase_space_portrait,
    export_empirical_ledger,
    generate_empirical_report,

    # 11. arXiv Deep Research & Literature Extraction (3 tools)
    arxiv_search_papers,
    arxiv_get_paper_details,
    arxiv_download_pdf,

    # 12. C++20 / CUDA JIT Compiler & Micro-Benchmarking (4 tools)
    build_and_verify_cpp_core,
    inspect_cuda_environment,
    benchmark_cpp_kernel,
    clean_cpp_build_cache,

    # 13. Autonomous KEP Scientific Pipeline (1 tool)
    run_kep_scientific_pipeline,

    # 14. Database Cloud Sync, Snapshots, Search & Optimization (18 tools)
    sync_agent_database,
    list_database_backups,
    create_database_snapshot,
    rename_database_snapshot,
    delete_database_snapshot,
    get_database_info,
    switch_database,
    restore_database_backup,
    record_experiment_result,
    get_experiment_history,
    get_latest_experiment,
    set_agent_memory,
    get_agent_memory,
    list_agent_memory,
    delete_agent_memory,
    search_persistent_memory,
    vacuum_and_optimize_database,
    prune_and_vacuum_database,
    export_empirical_database_json,

    # 15. Complete Git Control Suite (22 tools)
    git_status,
    git_diff,
    git_log,
    git_add,
    git_commit,
    git_push,
    git_pull,
    git_fetch,
    git_branch,
    git_checkout,
    git_show,
    git_stash,
    git_rollback,
    git_blame,
    git_cherry_pick,
    git_tag,
    git_merge,
    git_clean,
    git_remote,
    git_sync_hard_reset,
    git_create_patch,
    git_apply_patch,

    # 16. Web Research Suite (6 tools)
    internet_search,
    internet_media_search,
    internet_deep_search,
    scrape_url,
    download_file_from_url,
    send_http_request,

    # 17. Filesystem & Codebase Operations (11 tools)
    read_file,
    write_file,
    list_directory,
    search_codebase,
    find_files,
    delete_file_or_dir,
    make_directory,
    move_or_rename_file,
    get_codebase_tree,
    diff_files,
    batch_replace_text,

    # 18. System, CPU, RAM, Storage & TPU Accelerator Telemetry Engine (17 tools)
    get_tpu_hardware_telemetry,
    benchmark_tpu_performance,
    clear_tpu_memory_cache,
    audit_tpu_workload,
    get_cpu_telemetry,
    benchmark_cpu_performance,
    get_ram_telemetry,
    audit_process_memory,
    optimize_ram_and_clean_caches,
    get_storage_telemetry,
    benchmark_storage_io,
    clean_disk_storage,
    get_gpu_hardware_telemetry,
    get_cuda_memory_summary,
    clear_cuda_cache,
    get_system_info,
    list_active_processes,

    # 19. Meta-Tool Orchestration & Persistent Background Tool Suite (11 tools)
    execute_tools_sequential,
    execute_tools_parallel,
    run_tool_in_background,
    poll_tool_job,
    tail_tool_job,
    get_tool_job_logs,
    list_tool_jobs,
    kill_tool_job,
    kill_all_tool_jobs,
    clean_stale_tool_jobs,
    clear_tool_jobs_history,

    # 20. Dynamic Custom Tools Engine (6 tools)
    create_custom_tool,
    update_custom_tool,
    test_custom_tool,
    list_custom_tools,
    inspect_custom_tool,
    delete_custom_tool,

    # 21. Multimodal File Binding (1 tool)
    upload_file_to_google,

    # 22. Hierarchical Multi-Agent Cortical Hierarchy & Synaptic Network (14 tools)
    create_subagent,
    update_subagent,
    delete_subagent,
    list_subagents,
    get_subagent_info,
    dispatch_agent_task,
    send_agent_message_to,
    get_agent_inbox,
    get_agent_dialogue_history,
    get_agent_task_status,
    spawn_ephemeral_subagent,
    get_swarm_telemetry,
    transfer_turn_to_agent,
    dispatch_parallel_agent_tasks,

    # 23. Code Quality Assurance, Static Linting & Verification Suite (3 tools)
    verify_code_syntax,
    run_code_linter,
    run_unit_tests,
]

# Guaranteed unique function declaration schema filter for Gemini API
_seen_names = set()
_UNIQUE_TOOLS = []
for _tool in AGENT_TOOLS:
    _name = getattr(_tool, "__name__", str(_tool))
    if _name not in _seen_names:
        _seen_names.add(_name)
        _UNIQUE_TOOLS.append(_tool)

AGENT_TOOLS = _UNIQUE_TOOLS
