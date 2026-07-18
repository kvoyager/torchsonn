import logging
import os
from pathlib import Path

from torchsonn.logger import TqdmLoggingHandler, setup_logger


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
    def test_setup_default_path(self, tmp_path, monkeypatch):
        # Force the default temp-file path into our temp directory.
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        logger = setup_logger()
        assert logger is logging.getLogger()
        assert logger.level == logging.INFO
        # Two handlers: tqdm console + file
        assert len(logger.handlers) == 2
        # Log file should be created at the predicted path
        assert (tmp_path / "sonn_train.log").exists()
        for h in logger.handlers:
            h.close()

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
