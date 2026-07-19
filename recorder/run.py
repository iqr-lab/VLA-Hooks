import random
import socket
from typing import Any

from recorder.config import load_yaml
from recorder.paths import make_run_dirs
from recorder.process import wait_for_log_pattern
from recorder.server import start_server
from recorder.libero import run_libero_eval


def _is_auto_port(exp: dict[str, Any]) -> bool:
    return str(exp.get("port", "")).strip().lower() == "auto"


def _auto_port_range(exp: dict[str, Any], containers_cfg: dict[str, Any]) -> tuple[int, int]:
    range_cfg = exp.get("auto_port_range", containers_cfg.get("auto_port_range", [20000, 65000]))

    if isinstance(range_cfg, dict):
        start = int(range_cfg.get("start", range_cfg.get("min", 20000)))
        end = int(range_cfg.get("end", range_cfg.get("max", 65000)))
    else:
        start, end = [int(value) for value in range_cfg]

    if start <= 0 or end > 65535 or start > end:
        raise ValueError(f"Invalid auto_port_range: {range_cfg}")

    return start, end


def _can_bind_port(port: int, *, host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _find_free_port(exp: dict[str, Any], containers_cfg: dict[str, Any]) -> int:
    start, end = _auto_port_range(exp, containers_cfg)
    attempts = max(1, int(exp.get("auto_port_scan_attempts", containers_cfg.get("auto_port_scan_attempts", 200))))
    span = end - start + 1

    for port in random.SystemRandom().sample(range(start, end + 1), k=min(attempts, span)):
        if _can_bind_port(port):
            return port

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", 0))
            return int(sock.getsockname()[1])
    except OSError as exc:
        raise RuntimeError(
            "Could not auto-select a free policy server port. "
            "Set an explicit `port:` in the experiment YAML or adjust `auto_port_range`."
        ) from exc


def _looks_like_port_collision(exc: Exception) -> bool:
    text = str(exc).lower()
    return "address already in use" in text or "errno 98" in text or "errno 48" in text


def run_experiment(
    *,
    experiment_path: str,
    models_path: str = "configs/models.yaml",
    containers_path: str = "configs/containers.yaml",
) -> None:
    exp = dict(load_yaml(experiment_path))
    models_cfg = load_yaml(models_path)
    containers_cfg = load_yaml(containers_path)
    model_cfg = models_cfg["models"][exp["model"]]
    uses_server = model_cfg.get("run_mode") != "direct_libero"
    auto_port = uses_server and _is_auto_port(exp)

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
        max_port_attempts = max(1, int(exp.get("auto_port_retries", containers_cfg.get("auto_port_retries", 8))))

        for port_attempt in range(max_port_attempts if auto_port else 1):
            if auto_port:
                exp["port"] = _find_free_port(exp, containers_cfg)
                print(f"Auto-selected policy server port: {exp['port']}")

            try:
                if uses_server:
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
                break
            except Exception as exc:
                if server is not None:
                    server.stop()
                    server = None

                can_retry = auto_port and _looks_like_port_collision(exc) and port_attempt + 1 < max_port_attempts
                if not can_retry:
                    raise

                print(f"Auto-selected port {exp['port']} was busy by server startup; retrying...")

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
