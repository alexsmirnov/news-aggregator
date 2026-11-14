"""Keyboard-driven editor for tests/evaluation/data/expected_groups_*.yaml.

Usage:
    uv run python tests/evaluation/edit_groups.py \
        <rss_entries.json> <expected_groups.yaml> [--host H] [--port P]
"""

import argparse
import json
from pathlib import Path

import uvicorn
import yaml
from group_editor import GroupEditor, load_entries, load_groups
from group_editor_app import create_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4091
YAML_SUFFIXES = (".yaml", ".yml")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edit_groups")
    parser.add_argument("entries_path", type=Path, help="rss_entries_*.json")
    parser.add_argument("groups_path", type=Path, help="expected_groups.yaml")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def _check_paths(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if not args.entries_path.is_file():
        parser.error(f"entries file not found: {args.entries_path}")
    if args.groups_path.suffix not in YAML_SUFFIXES:
        parser.error(
            f"groups file must end with .yaml or .yml: {args.groups_path}"
        )
    if args.groups_path.resolve() == args.entries_path.resolve():
        parser.error("groups file must differ from the entries file")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _check_paths(parser, args)
    if not args.groups_path.exists():
        print(f"creating new file {args.groups_path}", flush=True)
    try:
        editor = GroupEditor(
            load_entries(args.entries_path),
            load_groups(args.groups_path),
            args.groups_path,
        )
    except (ValueError, OSError, yaml.YAMLError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    uvicorn.run(create_app(editor), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
