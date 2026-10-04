import logging
from pathlib import Path

from tqdm import tqdm


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


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
        """Format `record` and write it with `tqdm.write`."""
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class RunLogHandler(logging.FileHandler):
    """File handler for the `train.log` of a training run's folder.

    `Trainer` adds one to the root logger when a run starts or resumes. The
    root logger holds at most one: `attach_run_log` removes the previous
    run's handler, whichever trainer added it, so each `train.log` holds
    only its own run. Other handlers are left alone.
    """


def attach_run_log(path: str | Path, append: bool = False) -> RunLogHandler:
    """Log the root logger's records to a run's `train.log`.

    Replaces any `RunLogHandler` already on the root logger, and lowers the
    root logger's level to INFO if it is above, so the file gets the INFO
    records; other handlers keep their own levels.

    Parameters
    ----------
    path : str or Path
        The log file, `<run folder>/train.log`.
    append : bool
        Append to an existing file (a resumed run) instead of starting a new
        one. Default False.

    Returns
    -------
    RunLogHandler
        The handler added to the root logger.
    """
    root = logging.getLogger()
    detach_run_log()
    if root.getEffectiveLevel() > logging.INFO:
        root.setLevel(logging.INFO)
    handler = RunLogHandler(str(path), mode="a" if append else "w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    return handler


def detach_run_log() -> None:
    """Remove and close every `RunLogHandler` on the root logger."""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if isinstance(h, RunLogHandler)]:
        root.removeHandler(handler)
        handler.close()


def setup_logger(log_path: str | Path | None = None) -> logging.Logger:
    """Configure the root logger for a training script.

    Replaces every handler on the root logger (closing them, a run's
    `train.log` handler included) with a tqdm-aware console handler at INFO
    level, plus a file handler when `log_path` is given. Each training run
    also writes its own `train.log` in its run folder (see
    `Trainer.run_dir`), so a file here is needed only for lines outside
    the runs.

    Parameters
    ----------
    log_path : str, Path or None
        A log file to write as well, overwritten on each call. Default None:
        the console only.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers (avoid duplicate logs if setup_logger is called again)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler — tqdm-aware so logs don't tear the progress bar.
    ch = TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_path is not None:
        # File handler — plain, no tqdm involvement.
        fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
