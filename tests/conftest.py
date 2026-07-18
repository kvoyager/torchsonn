import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    yield
