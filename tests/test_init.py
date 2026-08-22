import logging
import sys
from unittest.mock import AsyncMock, Mock

import pytest

import news
from news import UVICORN_LOG_CONFIG, _parse_args, configure_logging


def test_parse_args_no_argv_defaults_to_server_with_current_defaults() -> None:
    # Act
    args = _parse_args([])

    # Assert
    assert args.command == "server"
    assert args.host == "0.0.0.0"
    assert args.port == 4090
    assert args.reload is True


def test_parse_args_bare_options_without_subcommand_default_to_server() -> (
    None
):
    # Act
    args = _parse_args(["--port", "8080"])

    # Assert
    assert args.command == "server"
    assert args.port == 8080
    assert args.host == "0.0.0.0"
    assert args.reload is True


def test_parse_args_server_explicit_with_all_overrides() -> None:
    # Act
    args = _parse_args(
        ["server", "--host", "1.2.3.4", "--port", "9000", "--no-reload"]
    )

    # Assert
    assert args.command == "server"
    assert args.host == "1.2.3.4"
    assert args.port == 9000
    assert args.reload is False


def test_parse_args_aggregate_returns_aggregate_command() -> None:
    # Act
    args = _parse_args(["aggregate"])

    # Assert
    assert args.command == "aggregate"


def test_parse_args_aggregate_rejects_server_only_options() -> None:
    # Act / Assert
    with pytest.raises(SystemExit) as exc_info:
        _parse_args(["aggregate", "--host", "1.2.3.4"])
    assert exc_info.value.code == 2


def test_parse_args_unknown_command_raises_system_exit() -> None:
    # Act / Assert
    with pytest.raises(SystemExit) as exc_info:
        _parse_args(["bogus"])
    assert exc_info.value.code == 2


def test_parse_args_help_flag_raises_system_exit_zero() -> None:
    # Act / Assert
    with pytest.raises(SystemExit) as exc_info:
        _parse_args(["--help"])
    assert exc_info.value.code == 0


def test_configure_logging_defaults_to_info_with_timestamp_and_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    basic_config = Mock()
    monkeypatch.setattr(logging, "basicConfig", basic_config)

    # Act
    configure_logging()

    # Assert
    basic_config.assert_called_once_with(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def test_uvicorn_logging_config_preserves_access_logs() -> None:
    # Assert
    access_logger = UVICORN_LOG_CONFIG["loggers"]["uvicorn.access"]
    access_formatter = UVICORN_LOG_CONFIG["formatters"]["access"]["fmt"]
    assert access_logger == {
        "handlers": ["access"],
        "level": "INFO",
        "propagate": False,
    }
    assert "%(asctime)s" in access_formatter
    assert "%(name)s" in access_formatter
    assert "%(request_line)s" in access_formatter


def test_main_no_args_runs_uvicorn_with_default_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    mock_uvicorn = Mock()
    monkeypatch.setattr(sys, "argv", ["news"])
    monkeypatch.setattr(news, "configure_logging", Mock())
    monkeypatch.setattr(news, "uvicorn", mock_uvicorn)

    # Act
    news.main()

    # Assert
    mock_uvicorn.run.assert_called_once_with(
        "news.server:app",
        host="0.0.0.0",
        port=4090,
        reload=True,
        log_config=UVICORN_LOG_CONFIG,
    )


def test_main_server_command_with_overrides_runs_uvicorn_with_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    mock_uvicorn = Mock()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "news",
            "server",
            "--host",
            "1.2.3.4",
            "--port",
            "9000",
            "--no-reload",
        ],
    )
    monkeypatch.setattr(news, "configure_logging", Mock())
    monkeypatch.setattr(news, "uvicorn", mock_uvicorn)

    # Act
    news.main()

    # Assert
    mock_uvicorn.run.assert_called_once_with(
        "news.server:app",
        host="1.2.3.4",
        port=9000,
        reload=False,
        log_config=UVICORN_LOG_CONFIG,
    )


def test_main_aggregate_command_runs_aggregate_and_skips_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    mock_uvicorn = Mock()
    async_mock = AsyncMock()
    monkeypatch.setattr(sys, "argv", ["news", "aggregate"])
    monkeypatch.setattr(news, "configure_logging", Mock())
    monkeypatch.setattr(news, "uvicorn", mock_uvicorn)
    monkeypatch.setattr(news, "run_aggregate", async_mock)

    # Act
    news.main()

    # Assert
    async_mock.assert_awaited_once_with()
    mock_uvicorn.run.assert_not_called()
