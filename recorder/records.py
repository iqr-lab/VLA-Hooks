"""Read policy record files written by the hooked openpi policy servers.

Records are written once per inference call as `step_N.pirec` (a compressed
container) or, for older runs, `step_N.npy`. Both are read through
`record_io.load_record`, which lives in the model repo rather than here, so this
module locates it in the configured submodule and puts it on `sys.path`.
"""

from __future__ import annotations

import dataclasses
import json
import struct
import sys
from pathlib import Path
from typing import Any

from recorder.config import load_yaml
from recorder.paths import resolve_path

# Container magic and header framing; see record_io.py in the model repo.
_MAGIC = b"PIREC001"
_HEADER_LEN = struct.Struct("<I")

RECORD_SUFFIXES = (".pirec", ".npy")


def find_record_io(models_path: str | Path = "configs/models.yaml"):
    """Import `record_io` from whichever model repo provides it.

    The model repos are git submodules under `external/`, so this fails with an
    actionable message when they have not been checked out.
    """
    if "openpi.policies.record_io" in sys.modules:
        return sys.modules["openpi.policies.record_io"]

    models_cfg = load_yaml(resolve_path(models_path))
    searched: list[Path] = []

    for model_cfg in models_cfg.get("models", {}).values():
        repo = model_cfg.get("repo")
        if repo is None:
            continue
        src = resolve_path(repo) / "src"
        candidate = src / "openpi" / "policies" / "record_io.py"
        searched.append(candidate)
        if not candidate.exists():
            continue
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from openpi.policies import record_io

        return record_io

    listing = "\n  ".join(str(p) for p in searched) or "(no repos listed in models.yaml)"
    raise FileNotFoundError(
        "Could not find record_io.py in any configured model repo. Looked for:\n  "
        f"{listing}\n"
        "If the submodules are not checked out, run:\n"
        "  git submodule update --init external/openpi-pi0fast-hooks"
    )


def read_container_header(path: Path) -> dict[str, Any] | None:
    """Return the `.pirec` JSON header, or None if this is not a container.

    Reading the header alone is cheap and gives per-field stored (compressed)
    sizes without decompressing anything.
    """
    with path.open("rb") as f:
        if f.read(len(_MAGIC)) != _MAGIC:
            return None
        (header_len,) = _HEADER_LEN.unpack(f.read(_HEADER_LEN.size))
        return json.loads(f.read(header_len).decode("utf-8"))


@dataclasses.dataclass
class Field:
    """One leaf of a record."""

    key: str
    dtype: str
    shape: tuple[int, ...] | None
    nbytes: int  # in memory, after loading
    stored: int | None  # compressed bytes in the file, when known
    value: Any = None  # kept only for scalars and short values

    @property
    def ratio(self) -> float | None:
        if not self.stored:
            return None
        return self.nbytes / self.stored


@dataclasses.dataclass
class Schema:
    path: Path
    file_size: int
    fields: list[Field]
    codec: str | None = None
    float_dtype: str | None = None

    @property
    def nbytes(self) -> int:
        return sum(f.nbytes for f in self.fields)

    @property
    def ratio(self) -> float | None:
        return self.nbytes / self.file_size if self.file_size else None


def _describe(key: str, value: Any, stored: int | None) -> Field:
    import numpy as np

    if isinstance(value, np.ndarray) and value.dtype != object:
        scalar = value.item() if value.size == 1 else None
        return Field(key, str(value.dtype), tuple(value.shape), int(value.nbytes), stored, scalar)

    # Scalars, None, strings, and anything else the hooks emitted.
    text = repr(value)
    return Field(
        key,
        type(value).__name__,
        None,
        sys.getsizeof(value),
        stored,
        value if len(text) <= 120 else None,
    )


def read_schema(path: str | Path, *, models_path: str | Path = "configs/models.yaml") -> Schema:
    """Load one record and describe every field in it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such record: {path}")

    record_io = find_record_io(models_path)
    record = record_io.load_record(path)

    stored: dict[str, int] = {}
    codec = float_dtype = None
    if path.suffix == ".pirec":
        header = read_container_header(path)
        if header is not None:
            codec = header.get("codec")
            float_dtype = header.get("float_dtype")
            stored = {e["key"]: e["nbytes"] for e in header.get("entries", [])}

    fields = [_describe(k, v, stored.get(k)) for k, v in record.items()]
    return Schema(
        path=path,
        file_size=path.stat().st_size,
        fields=fields,
        codec=codec,
        float_dtype=float_dtype,
    )
