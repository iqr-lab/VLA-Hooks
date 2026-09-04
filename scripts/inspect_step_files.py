#!/usr/bin/env python3
"""Print the schema of a policy record file (`step_N.pirec` or `step_N.npy`).

Examples:
    python scripts/inspect_step_files.py /path/to/run/step_0.pirec
    python scripts/inspect_step_files.py step_0.pirec --sort size --values
    python scripts/inspect_step_files.py step_0.npy --json
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# Runnable straight from a checkout, without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recorder.records import read_schema


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def print_table(schema, *, sort: str, show_values: bool) -> None:
    fields = list(schema.fields)
    if sort == "size":
        fields.sort(key=lambda f: f.nbytes, reverse=True)
    else:
        fields.sort(key=lambda f: f.key)

    has_stored = any(f.stored is not None for f in fields)

    print(f"File      {schema.path}")
    fmt = "pirec container" if schema.path.suffix == ".pirec" else "legacy npy (pickled dict)"
    if schema.codec:
        fmt += f"  [codec={schema.codec}, float_dtype={schema.float_dtype}]"
    print(f"Format    {fmt}")
    print(f"On disk   {human(schema.file_size)}")
    print(f"In memory {human(schema.nbytes)}" + (f"  ({schema.ratio:.2f}x)" if schema.ratio else ""))
    print(f"Fields    {len(fields)}")
    print()

    key_w = min(max(max((len(f.key) for f in fields), default=3), 3), 70)
    dtype_w = max((len(f.dtype) for f in fields), default=5)

    header = f"{'key':<{key_w}}  {'dtype':<{dtype_w}}  {'shape':<24}  {'memory':>10}"
    if has_stored:
        header += f"  {'stored':>10}  {'ratio':>6}"
    print(header)
    print("-" * len(header))

    for f in fields:
        shape = "-" if f.shape is None else str(tuple(f.shape))
        row = f"{f.key:<{key_w}}  {f.dtype:<{dtype_w}}  {shape:<24}  {human(f.nbytes):>10}"
        if has_stored:
            stored = human(f.stored) if f.stored is not None else "-"
            ratio = f"{f.ratio:.2f}x" if f.ratio else "-"
            row += f"  {stored:>10}  {ratio:>6}"
        print(row)
        if show_values and f.value is not None:
            print(f"{'':<{key_w}}  = {f.value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("record", help="Path to a step_N.pirec or step_N.npy file")
    parser.add_argument(
        "--models",
        default="configs/models.yaml",
        help="Path to models YAML, used to locate record_io inside a model repo",
    )
    parser.add_argument(
        "--sort",
        choices=("key", "size"),
        default="key",
        help="Order fields by name (default) or by memory footprint",
    )
    parser.add_argument(
        "--values", action="store_true", help="Also print the value of scalar and short fields"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")

    args = parser.parse_args()

    try:
        schema = read_schema(args.record, models_path=args.models)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if args.json:
        payload = dataclasses.asdict(schema)
        payload["path"] = str(schema.path)
        payload["nbytes"] = schema.nbytes
        for field, src in zip(payload["fields"], schema.fields, strict=True):
            field["shape"] = list(src.shape) if src.shape is not None else None
            field["ratio"] = src.ratio
            if not isinstance(field["value"], (int, float, str, bool, type(None))):
                field["value"] = repr(src.value)
        print(json.dumps(payload, indent=2))
        return

    print_table(schema, sort=args.sort, show_values=args.values)


if __name__ == "__main__":
    main()
