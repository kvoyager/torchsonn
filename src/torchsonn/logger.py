import logging
import os
import tempfile

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Console log handler that writes through `tqdm.write`.

    Plain logging.StreamHandler emits straight to stdout, which collides with
    any active tqdm bar: the bar's last redraw stays on screen and the log
    line gets stamped onto the right of it. `tqdm.write` acquires tqdm's
    lock, clears the bar, prints the message, and re-renders the bar below —
    when no bar is active it falls back to a normal stdout write, so this is
    a safe drop-in replacement for StreamHandler.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(log_path: str | None = None) -> logging.Logger:
    if log_path is None:
        log_path = os.path.join(tempfile.gettempdir(), "sonn_train.log")

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers (avoid duplicate logs if setup_logger is called again)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # Console handler — tqdm-aware so logs don't tear the progress bar.
    ch = TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # File handler — plain, no tqdm involvement.
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

