from pathlib import Path

from recorder.config import load_yaml
from recorder.paths import make_run_dirs
from recorder.process import wait_for_log_pattern
from recorder.server import start_server
from recorder.libero import run_libero_eval


def run_experiment(
    *,
    experiment_path: str,
    models_path: str = "configs/models.yaml",
    containers_path: str = "configs/containers.yaml",
) -> None:
    exp = load_yaml(experiment_path)
    models_cfg = load_yaml(models_path)
    containers_cfg = load_yaml(containers_path)

    record_dir, log_dir = make_run_dirs(
        exp_name=exp["name"],
        record_root=exp.get("record_root", "records"),
        log_root=exp.get("log_root", "logs"),
    )

    print("========================================")
    print(f"Experiment: {exp['name']}")
    print(f"Model: {exp['model']}")
    print(f"Record dir: {record_dir}")
    print(f"Log dir: {log_dir}")
    print("========================================")

    server = None

    try:
        server = start_server(
            exp=exp,
            models_cfg=models_cfg,
            containers_cfg=containers_cfg,
            record_dir=record_dir,
            log_dir=log_dir,
        )

        wait_for_log_pattern(
            log_path=log_dir / "server.log",
            process=server,
            patterns=[
                "DEBUG: entering serve_forever",
                "server listening",
                "Server listening",
                "Listening",
            ],
            timeout_s=360,
        )

        run_libero_eval(
            exp=exp,
            models_cfg=models_cfg,
            containers_cfg=containers_cfg,
            record_dir=record_dir,
            log_dir=log_dir,
        )

    finally:
        if server is not None:
            print("Stopping server...")
            server.stop()

    print("========================================")
    print("DONE")
    print(f"Records: {record_dir}")
    print(f"Logs: {log_dir}")
    print("========================================")