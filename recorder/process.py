import subprocess
import time
from pathlib import Path


class ManagedProcess:
    def __init__(self, command: list[str], log_path: Path, cwd: Path | None = None):
        self.command = command
        self.log_path = log_path
        self.cwd = cwd
        self.process: subprocess.Popen | None = None
        self.log_file = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w")

        print("Starting command:")
        print(" ".join(self.command))
        print(f"Logging to: {self.log_path}")

        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        if self.process is None:
            return

        if self.process.poll() is None:
            self.process.terminate()

            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

        if self.log_file is not None:
            self.log_file.close()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def wait(self) -> int:
        if self.process is None:
            raise RuntimeError("Process was never started")

        code = self.process.wait()

        if self.log_file is not None:
            self.log_file.close()

        return code


def wait_for_log_pattern(
    *,
    log_path: Path,
    process: ManagedProcess,
    patterns: list[str],
    timeout_s: int = 360,
    poll_s: float = 2.0,
) -> None:
    start = time.time()

    while time.time() - start < timeout_s:
        if not process.is_running():
            content = log_path.read_text(errors="replace") if log_path.exists() else ""
            raise RuntimeError(
                f"Process exited before becoming ready.\n\nLog:\n{content}"
            )

        if log_path.exists():
            text = log_path.read_text(errors="replace")
            if any(pattern in text for pattern in patterns):
                print("Server is ready")
                return

        time.sleep(poll_s)

    content = log_path.read_text(errors="replace") if log_path.exists() else ""
    raise TimeoutError(f"Timed out waiting for server.\n\nLog:\n{content}")