import argparse
import asyncio
import sys

import uvicorn

from news.server import run_aggregate

_COMMANDS = ("server", "aggregate")
_HELP_FLAGS = ("-h", "--help")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="news")
    subparsers = parser.add_subparsers(dest="command")

    server_parser = subparsers.add_parser("server")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=4090)
    server_parser.add_argument(
        "--reload", action=argparse.BooleanOptionalAction, default=True
    )

    subparsers.add_parser("aggregate")

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in (*_COMMANDS, *_HELP_FLAGS):
        argv = ["server", *argv]
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    if args.command == "aggregate":
        asyncio.run(run_aggregate())
        return
    uvicorn.run(
        "news.server:app", host=args.host, port=args.port, reload=args.reload
    )
