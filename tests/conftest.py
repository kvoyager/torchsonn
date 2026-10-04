import random

import numpy as np
import pytest
import torch

from torchsonn.logger import detach_run_log


@pytest.fixture(autouse=True)
def _deterministic():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    yield


@pytest.fixture(autouse=True)
def _close_run_log():
    # Trainer.train attaches the run's train.log to the root logger and keeps
    # it until the next run; close it so no test leaves a file open.
    yield
    detach_run_log()
