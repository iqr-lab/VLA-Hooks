#!/usr/bin/env python3
"""Print and inspect a policy record file (`step_N.pirec` or `step_N.npy`).

Examples:
    python scripts/inspect_step_files.py /path/to/run/step_0.pirec
    python scripts/inspect_step_files.py step_0.pirec --sort size
    python scripts/inspect_step_files.py step_0.pirec --inspect hook_records
    python scripts/inspect_step_files.py step_0.pirec --inspect hook_records --max-items 20
    python scripts/inspect_step_files.py step_0.npy --json
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

# Runnable straight from a checkout, without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recorder.records import load_record, read_schema


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def print_nested(
    value: Any,
    *,
    indent: int = 0,
    max_items: int = 10,
    max_depth: int = 8,
) -> None:
    """Recursively print lists, dictionaries, arrays, and scalar values."""
    prefix = "  " * indent

    if indent >= max_depth:
        print(f"{prefix}... maximum depth reached")
        return

    # NumPy arrays and other array-like objects.
    if all(hasattr(value, attr) for attr in ("shape", "dtype", "nbytes")):
        print(
            f"{prefix}{type(value).__name__}("
            f"shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, "
            f"memory={human(value.nbytes)})"
        )

        # Print actual contents only for small arrays.
        size = getattr(value, "size", None)
        if isinstance(size, int) and size <= max_items:
            try:
                print(f"{prefix}values={value.tolist()!r}")
            except (AttributeError, TypeError):
                pass

        return

    # PyTorch tensors.
    if all(hasattr(value, attr) for attr in ("shape", "dtype", "numel", "element_size")):
        nbytes = value.numel() * value.element_size()

        print(
            f"{prefix}{type(value).__name__}("
            f"shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, "
            f"memory={human(nbytes)})"
        )

        if value.numel() <= max_items:
            try:
                print(f"{prefix}values={value.detach().cpu().tolist()!r}")
            except (AttributeError, RuntimeError):
                pass

        return

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        print(f"{prefix}{type(value).__name__}")

        for index, field in enumerate(dataclasses.fields(value)):
            if index >= max_items:
                remaining = len(dataclasses.fields(value)) - max_items
                print(f"{prefix}  ... {remaining} more fields")
                break

            print(f"{prefix}  {field.name}:")
            print_nested(
                getattr(value, field.name),
                indent=indent + 2,
                max_items=max_items,
                max_depth=max_depth,
            )

        return

    if isinstance(value, dict):
        print(f"{prefix}dict ({len(value)} keys)")

        for index, (key, child) in enumerate(value.items()):
            if index >= max_items:
                remaining = len(value) - max_items
                print(f"{prefix}  ... {remaining} more keys")
                break

            print(f"{prefix}  {key!r}:")
            print_nested(
                child,
                indent=indent + 2,
                max_items=max_items,
                max_depth=max_depth,
            )

        return

    if isinstance(value, (list, tuple)):
        print(f"{prefix}{type(value).__name__} ({len(value)} items)")

        for index, child in enumerate(value):
            if index >= max_items:
                remaining = len(value) - max_items
                print(f"{prefix}  ... {remaining} more items")
                break

            print(f"{prefix}  [{index}]:")
            print_nested(
                child,
                indent=indent + 2,
                max_items=max_items,
                max_depth=max_depth,
            )

        return

    # Handle custom Python objects.
    if hasattr(value, "__dict__"):
        attributes = vars(value)
        print(f"{prefix}{type(value).__name__} ({len(attributes)} attributes)")

        for index, (name, child) in enumerate(attributes.items()):
            if index >= max_items:
                remaining = len(attributes) - max_items
                print(f"{prefix}  ... {remaining} more attributes")
                break

            print(f"{prefix}  {name}:")
            print_nested(
                child,
                indent=indent + 2,
                max_items=max_items,
                max_depth=max_depth,
            )

        return

    print(f"{prefix}{type(value).__name__}: {value!r}")


def print_table(schema, *, sort: str, show_values: bool) -> None:
    fields = list(schema.fields)

    if sort == "size":
        fields.sort(key=lambda field: field.nbytes, reverse=True)
    else:
        fields.sort(key=lambda field: field.key)

    has_stored = any(field.stored is not None for field in fields)

    print(f"File      {schema.path}")

    if schema.path.suffix == ".pirec":
        fmt = "pirec container"
    else:
        fmt = "legacy npy (pickled dict)"

    if schema.codec:
        fmt += (
            f"  [codec={schema.codec}, "
            f"float_dtype={schema.float_dtype}]"
        )

    print(f"Format    {fmt}")
    print(f"On disk   {human(schema.file_size)}")

    ratio_text = f"  ({schema.ratio:.2f}x)" if schema.ratio else ""
    print(f"In memory {human(schema.nbytes)}{ratio_text}")

    print(f"Fields    {len(fields)}")
    print()

    key_w = min(
        max(max((len(field.key) for field in fields), default=3), 3),
        70,
    )
    dtype_w = max(
        (len(field.dtype) for field in fields),
        default=5,
    )

    header = (
        f"{'key':<{key_w}}  "
        f"{'dtype':<{dtype_w}}  "
        f"{'shape':<24}  "
        f"{'memory':>10}"
    )

    if has_stored:
        header += f"  {'stored':>10}  {'ratio':>6}"

    print(header)
    print("-" * len(header))

    for field in fields:
        shape = "-" if field.shape is None else str(tuple(field.shape))

        row = (
            f"{field.key:<{key_w}}  "
            f"{field.dtype:<{dtype_w}}  "
            f"{shape:<24}  "
            f"{human(field.nbytes):>10}"
        )

        if has_stored:
            stored = (
                human(field.stored)
                if field.stored is not None
                else "-"
            )
            ratio = f"{field.ratio:.2f}x" if field.ratio else "-"
            row += f"  {stored:>10}  {ratio:>6}"

        print(row)

        if show_values and field.value is not None:
            print(f"{'':<{key_w}}  = {field.value!r}")


def inspect_record_field(
    record_path: str,
    *,
    key: str,
    models_path: str,
    max_items: int,
    max_depth: int,
) -> None:
    """Load the complete record and inspect one field."""
    try:
        record = load_record(
            record_path,
            models_path=models_path,
        )
    except FileNotFoundError as exc:
        print(f"error loading record: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except Exception as exc:
        print(
            f"error loading full record: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None

    print()
    print(f"Contents of {key}")
    print("=" * (12 + len(key)))

    if not isinstance(record, dict):
        print(
            "The loaded record is not a dictionary. "
            f"It is a {type(record).__name__}."
        )
        print_nested(
            record,
            max_items=max_items,
            max_depth=max_depth,
        )
        return

    if key not in record:
        print(f"Field {key!r} was not found.")
        print("Available fields:")

        for available_key in record:
            print(f"  {available_key}")

        return

    print_nested(
        record[key],
        max_items=max_items,
        max_depth=max_depth,
    )



def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "record",
        help="Path to a step_N.pirec or step_N.npy file",
    )

    parser.add_argument(
        "--models",
        default="configs/models.yaml",
        help="Path to models YAML, used to locate record_io inside a model repo",
    )

    parser.add_argument(
        "--sort",
        choices=("key", "size"),
        default="key",
        help="Order fields by name (default) or memory footprint",
    )

    parser.add_argument(
        "--values",
        action="store_true",
        help="Also print scalar and short field values",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a table",
    )

    parser.add_argument(
        "--inspect",
        metavar="KEY",
        help="Load and recursively inspect a field, such as hook_records",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Maximum entries shown per list or dictionary (default: 10)",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum recursive inspection depth (default: 8)",
    )

    args = parser.parse_args()

    try:
        schema = read_schema(
            args.record,
            models_path=args.models,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if args.json:
        payload = dataclasses.asdict(schema)
        payload["path"] = str(schema.path)
        payload["nbytes"] = schema.nbytes

        for field, source in zip(
            payload["fields"],
            schema.fields,
            strict=True,
        ):
            field["shape"] = (
                list(source.shape)
                if source.shape is not None
                else None
            )
            field["ratio"] = source.ratio

            if not isinstance(
                field["value"],
                (int, float, str, bool, type(None)),
            ):
                field["value"] = repr(source.value)

        print(json.dumps(payload, indent=2))
        return

    print_table(
        schema,
        sort=args.sort,
        show_values=args.values,
    )

    if args.inspect:
        inspect_record_field(
            args.record,
            key=args.inspect,
            models_path=args.models,
            max_items=args.max_items,
            max_depth=args.max_depth,
        )


if __name__ == "__main__":
    main()