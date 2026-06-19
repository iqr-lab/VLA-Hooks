import subprocess
from pathlib import Path
from typing import Any

from recorder.paths import resolve_path
from recorder.server import build_mount_args


def build_libero_command(
    *,
    exp: dict[str, Any],
    models_cfg: dict[str, Any],
    containers_cfg: dict[str, Any],
    record_dir: Path,
) -> tuple[list[str], Path]:
    model_name = exp["model"]
    model_cfg = models_cfg["models"][model_name]
    repo_path = resolve_path(model_cfg["repo"])

    task_suite = exp["task_suite"]
    num_trials = str(exp["num_trials"])
    port = str(exp["port"])
    host = exp.get("host", "127.0.0.1")

    use_apptainer = exp.get("use_apptainer", containers_cfg.get("use_apptainer", False))

    if not use_apptainer:
        command = [
            "python",
            "-u",
            "examples/libero/main.py",
            "--args.task_suite_name",
            task_suite,
            "--args.num_trials_per_task",
            num_trials,
            "--args.port",
            port,
            "--args.host",
            host,
            "--args.record_dir",
            str(record_dir),
        ]
        return command, repo_path

    libero_sif = containers_cfg["libero_sif"]
    pythonpath = containers_cfg["pythonpath"]

    command = [
        "apptainer",
        "exec",
        "--nv",
        "--containall",
        "--bind",
        f"{repo_path}:/app",
        *build_mount_args(containers_cfg),
        libero_sif,
        "bash",
        "-c",
        " ".join(
            [
                "cd /app &&",
                f"export PYTHONPATH={pythonpath} &&",
                "export CUDA_VISIBLE_DEVICES=0 &&",
                "export PYTHONUNBUFFERED=1 &&",
                "export PYTHONFAULTHANDLER=1 &&",
                "mkdir -p /tmp/libero &&",
                "printf '%s\\n'",
                "'benchmark_root: /app/third_party/libero/libero/libero'",
                "'bddl_files: /app/third_party/libero/libero/libero/./bddl_files'",
                "'init_states: /app/third_party/libero/libero/libero/./init_files'",
                "'datasets: /app/third_party/libero/libero/libero/../datasets'",
                "'assets: /app/third_party/libero/libero/libero/./assets'",
                "> /tmp/libero/config.yaml &&",
                "export MUJOCO_GL=osmesa &&",
                "export PYOPENGL_PLATFORM=osmesa &&",
                "/.venv/bin/python -u examples/libero/main.py",
                f"--args.task_suite_name {task_suite}",
                f"--args.num_trials_per_task {num_trials}",
                f"--args.port {port}",
                f"--args.host {host}",
                f"--args.record_dir {record_dir}",
            ]
        ),
    ]

    return command, repo_path


def run_libero_eval(
    *,
    exp: dict[str, Any],
    models_cfg: dict[str, Any],
    containers_cfg: dict[str, Any],
    record_dir: Path,
    log_dir: Path,
) -> None:
    command, cwd = build_libero_command(
        exp=exp,
        models_cfg=models_cfg,
        containers_cfg=containers_cfg,
        record_dir=record_dir,
    )

    eval_log = log_dir / "eval.log"
    eval_log.parent.mkdir(parents=True, exist_ok=True)

    print("Running LIBERO command:")
    print(" ".join(command))
    print(f"Logging to: {eval_log}")

    with eval_log.open("w") as f:
        subprocess.run(
            command,
            cwd=cwd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )