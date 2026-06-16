from pathlib import Path
from datetime import datetime


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    return repo_root() / path


def make_run_dirs(exp_name: str, record_root: str, log_root: str) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    record_dir = resolve_path(record_root) / f"{exp_name}_{timestamp}"
    log_dir = resolve_path(log_root) / f"{exp_name}_{timestamp}"

    record_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return record_dir, log_dir