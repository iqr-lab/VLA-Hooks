from pathlib import Path
from typing import Any

from recorder.paths import resolve_path
from recorder.process import ManagedProcess


def build_mount_args(containers_cfg: dict[str, Any]) -> list[str]:
    mount_args: list[str] = []

    for mount in containers_cfg.get("mounts", []):
        mount_args.extend(["--bind", f"{mount}:{mount}"])

    return mount_args


def build_server_command(
    *,
    exp: dict[str, Any],
    models_cfg: dict[str, Any],
    containers_cfg: dict[str, Any],
    record_dir: Path,
) -> tuple[list[str], Path]:
    model_name = exp["model"]
    model_cfg = models_cfg["models"][model_name]

    repo_path = resolve_path(model_cfg["repo"])
    serve_policy_path = repo_path / model_cfg["serve_policy"]

    if not serve_policy_path.exists():
        raise FileNotFoundError(f"serve_policy.py not found: {serve_policy_path}")

    hook_config = resolve_path(exp["hook_config"])
    checkpoint = exp["checkpoint"]
    policy_config = exp.get("policy_config", model_cfg["default_policy_config"])

    port = str(exp["port"])
    env = exp.get("env", "LIBERO")

    use_apptainer = exp.get("use_apptainer", containers_cfg.get("use_apptainer", False))

    inner_command = [
        "python",
        "-u",
        str(serve_policy_path),
        "--env",
        env,
        "--port",
        port,
        "--record",
        "--hook-config",
        str(hook_config),
        "--record-dir",
        str(record_dir),
        "policy:checkpoint",
        "--policy.config",
        policy_config,
        "--policy.dir",
        checkpoint,
    ]

    if not use_apptainer:
        return inner_command, repo_path

    server_sif = containers_cfg["server_sif"]
    pythonpath = containers_cfg["pythonpath"]

    command = [
        "apptainer",
        "exec",
        "--nv",
        "--containall",
        "--bind",
        f"{repo_path}:/app",
        "--bind",
        f"{hook_config}:/app/hooks.yaml",
        *build_mount_args(containers_cfg),
        server_sif,
        "bash",
        "-c",
        " ".join(
            [
                "cd /app &&",
                f"export PYTHONPATH={pythonpath} &&",
                "export CUDA_VISIBLE_DEVICES=0 &&",
                "export XLA_PYTHON_CLIENT_PREALLOCATE=false &&",
                "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 &&",
                "export PYTHONUNBUFFERED=1 &&",
                "export PYTHONFAULTHANDLER=1 &&",
                "/.venv/bin/python -u scripts/serve_policy.py",
                f"--env {env}",
                f"--port {port}",
                "--record",
                "--hook-config /app/hooks.yaml",
                f"--record-dir {record_dir}",
                "policy:checkpoint",
                f"--policy.config {policy_config}",
                f"--policy.dir {checkpoint}",
            ]
        ),
    ]

    return command, repo_path


def start_server(
    *,
    exp: dict[str, Any],
    models_cfg: dict[str, Any],
    containers_cfg: dict[str, Any],
    record_dir: Path,
    log_dir: Path,
) -> ManagedProcess:
    command, cwd = build_server_command(
        exp=exp,
        models_cfg=models_cfg,
        containers_cfg=containers_cfg,
        record_dir=record_dir,
    )

    server_log = log_dir / "server.log"
    proc = ManagedProcess(command=command, log_path=server_log, cwd=cwd)
    proc.start()
    return proc