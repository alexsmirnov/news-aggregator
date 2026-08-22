import argparse
import asyncio
import logging
import sys

import uvicorn

from news.server import run_aggregate

_COMMANDS = ("server", "aggregate")
_HELP_FLAGS = ("-h", "--help")
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(asctime)s %(levelprefix)s %(name)s %(message)s",
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": (
                "%(asctime)s %(levelprefix)s %(name)s "
                '%(client_addr)s - "%(request_line)s" %(status_code)s'
            ),
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {
            "handlers": ["access"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FORMAT,
        force=True,
    )


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
    configure_logging()
    if args.command == "aggregate":
        asyncio.run(run_aggregate())
        return
    uvicorn.run(
        "news.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=UVICORN_LOG_CONFIG,
    )
