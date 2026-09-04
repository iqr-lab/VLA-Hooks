"""Read policy record files written by the hooked openpi policy servers.

Records are written once per inference call as `step_N.pirec` (a compressed
container) or, for older runs, `step_N.npy`. Both are read through
`record_io.load_record`, which lives in the model repo rather than here, so this
module locates it in the configured submodule and puts it on `sys.path`.
"""

from __future__ import annotations

import dataclasses
import importlib
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

# Where record_io lives in each kind of model repo, as
# (path of the file inside the repo, sys.path root inside the repo, module name).
# The openpi forks ship it as a package module under src/; openvla-oft keeps its
# hook code under experiments/.
_RECORD_IO_LAYOUTS = (
    (Path("src/openpi/policies/record_io.py"), Path("src"), "openpi.policies.record_io"),
    (
        Path("experiments/robot/openvla_hooks/record_io.py"),
        Path("."),
        "experiments.robot.openvla_hooks.record_io",
    ),
)


def find_record_io(models_path: str | Path = "configs/models.yaml"):
    """Import `record_io` from whichever model repo provides it.

    All three model repos ship an identical record_io, just in different places,
    so the first one found wins. They are git submodules under `external/`, so
    this fails with an actionable message when none have been checked out.
    """
    for module_name in (name for _, _, name in _RECORD_IO_LAYOUTS):
        if module_name in sys.modules:
            return sys.modules[module_name]

    models_cfg = load_yaml(resolve_path(models_path))
    searched: list[Path] = []

    for model_cfg in models_cfg.get("models", {}).values():
        repo = model_cfg.get("repo")
        if repo is None:
            continue
        repo_dir = resolve_path(repo)
        for rel_file, rel_root, module_name in _RECORD_IO_LAYOUTS:
            candidate = repo_dir / rel_file
            searched.append(candidate)
            if not candidate.exists():
                continue
            root = str((repo_dir / rel_root).resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            return importlib.import_module(module_name)

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
    # True for a list/dict that was expanded into child fields. Its nbytes is the
    # recursive total of those children, so it must not be summed with them.
    is_container: bool = False

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
        return sum(f.nbytes for f in self.fields if not f.is_container)

    @property
    def ratio(self) -> float | None:
        return self.nbytes / self.file_size if self.file_size else None


def _leaf_field(key: str, value: Any, stored: int | None) -> Field:
    import numpy as np

    if isinstance(value, np.ndarray) and value.dtype != object:
        scalar = value.item() if value.size == 1 else None
        return Field(key, str(value.dtype), tuple(value.shape), int(value.nbytes), stored, scalar)

    if hasattr(value, "detach") and hasattr(value, "numel"):  # torch.Tensor
        nbytes = int(value.numel() * value.element_size())
        return Field(key, str(value.dtype), tuple(value.shape), nbytes, stored)

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


def _describe(key: str, value: Any, stored: int | None, *, depth: int = 0) -> list[Field]:
    """Describe one record entry, descending into lists and dicts.

    `hook_records` arrives as a list of dicts because the flatten step only
    recurses into dicts, so it reaches us as a single pickled leaf. Expanding it
    here is what makes the arrays inside it visible. The container row keeps the
    stored (compressed) size, since that is only known for the blob as a whole;
    its children report memory only.
    """
    import numpy as np

    is_array = isinstance(value, np.ndarray) and value.dtype != object
    if is_array or depth >= _MAX_EXPAND_DEPTH or not isinstance(value, (list, tuple, dict)):
        return [_leaf_field(key, value, stored)]

    if isinstance(value, dict):
        items = list(value.items())
        label = f"dict[{len(items)}]"
    else:
        items = list(enumerate(value))
        label = f"{type(value).__name__}[{len(items)}]"

    children: list[Field] = []
    for child_key, child_value in items:
        children.extend(_describe(f"{key}/{child_key}", child_value, None, depth=depth + 1))

    if not children:
        return [_leaf_field(key, value, stored)]

    total = sum(f.nbytes for f in children if not f.is_container)
    return [Field(key, label, None, total, stored, None, is_container=True), *children]


_MAX_EXPAND_DEPTH = 8


def load_record(
    path: str | Path, *, models_path: str | Path = "configs/models.yaml"
) -> dict[str, Any]:
    """Load one record as a flat dict, handling both `.pirec` and legacy `.npy`.

    This is the raw record; use :func:`read_schema` when you only want to
    describe it without materializing every array for inspection.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such record: {path}")

    return find_record_io(models_path).load_record(path)


def read_schema(path: str | Path, *, models_path: str | Path = "configs/models.yaml") -> Schema:
    """Load one record and describe every field in it."""
    path = Path(path)
    record = load_record(path, models_path=models_path)

    stored: dict[str, int] = {}
    codec = float_dtype = None
    if path.suffix == ".pirec":
        header = read_container_header(path)
        if header is not None:
            codec = header.get("codec")
            float_dtype = header.get("float_dtype")
            stored = {e["key"]: e["nbytes"] for e in header.get("entries", [])}

    fields: list[Field] = []
    for key, value in record.items():
        fields.extend(_describe(key, value, stored.get(key)))
    return Schema(
        path=path,
        file_size=path.stat().st_size,
        fields=fields,
        codec=codec,
        float_dtype=float_dtype,
    )
