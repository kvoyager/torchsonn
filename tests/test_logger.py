import logging
import os
from pathlib import Path

from torchsonn.logger import (
    RunLogHandler,
    TqdmLoggingHandler,
    attach_run_log,
    detach_run_log,
    setup_logger,
)


def _make_record(msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg=msg,
        args=(),
        exc_info=None,
    )


class TestTqdmLoggingHandler:
    def test_emit_writes_via_tqdm(self, monkeypatch):
        captured: list[str] = []
        monkeypatch.setattr("torchsonn.logger.tqdm.write", lambda m: captured.append(m))

        h = TqdmLoggingHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        h.emit(_make_record("hello"))

        assert captured == ["hello"]

    def test_emit_handles_exception_path(self, monkeypatch):
        # Make tqdm.write raise so the except branch executes.
        def boom(_msg: str) -> None:
            raise RuntimeError("nope")

        monkeypatch.setattr("torchsonn.logger.tqdm.write", boom)

        called: list[logging.LogRecord] = []

        h = TqdmLoggingHandler()
        # Replace handleError so the test doesn't actually print a traceback.
        h.handleError = called.append  # type: ignore[assignment]
        h.setFormatter(logging.Formatter("%(message)s"))
        h.emit(_make_record("hello"))

        assert len(called) == 1


class TestSetupLogger:
    def test_setup_default_is_console_only(self):
        logger = setup_logger()
        assert logger is logging.getLogger()
        assert logger.level == logging.INFO
        # One handler: the tqdm console, no file
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], TqdmLoggingHandler)

    def test_setup_explicit_path_and_idempotence(self, tmp_path):
        log_path = tmp_path / "my.log"
        setup_logger(str(log_path))
        first_handlers = list(logging.getLogger().handlers)

        # Second call should clear handlers and add two fresh ones.
        setup_logger(str(log_path))
        second_handlers = list(logging.getLogger().handlers)

        assert len(second_handlers) == 2
        assert set(second_handlers).isdisjoint(set(first_handlers))
        assert log_path.exists()
        for h in second_handlers:
            h.close()


class TestRunLog:
    def test_attach_writes_and_replaces_previous(self, tmp_path):
        setup_logger()
        first, second = tmp_path / "a.log", tmp_path / "b.log"
        attach_run_log(first)
        logging.getLogger("x").info("one")
        attach_run_log(second)
        logging.getLogger("x").info("two")
        detach_run_log()
        root = logging.getLogger()
        assert not any(isinstance(h, RunLogHandler) for h in root.handlers)
        # the console handler from setup_logger is still there
        assert any(isinstance(h, TqdmLoggingHandler) for h in root.handlers)
        assert "one" in first.read_text() and "two" not in first.read_text()
        assert "two" in second.read_text() and "one" not in second.read_text()

    def test_append_keeps_earlier_lines(self, tmp_path):
        log = tmp_path / "train.log"
        attach_run_log(log)
        logging.getLogger("x").info("first part")
        attach_run_log(log, append=True)
        logging.getLogger("x").info("second part")
        detach_run_log()
        text = log.read_text()
        assert "first part" in text and "second part" in text

    def test_lowers_root_level_to_info(self, tmp_path):
        root = logging.getLogger()
        root.setLevel(logging.WARNING)
        attach_run_log(tmp_path / "train.log")
        assert root.level == logging.INFO
        detach_run_log()
