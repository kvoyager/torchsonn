import torch
from torch import nn

from torchsonn.modules import SONNModule, SoftBinner


class _Demo(SONNModule):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.alpha = 0.5
        self.params_metadata_names.append("alpha")


class TestSONNModule:
    def test_state_dict_includes_params_metadata(self):
        m = _Demo()
        sd = m.state_dict()
        assert "params_metadata" in sd
        assert sd["params_metadata"] == {"alpha": 0.5}

    def test_state_dict_prefix_is_respected(self):
        m = _Demo()
        sd = m.state_dict(prefix="root.")
        assert "root.params_metadata" in sd

    def test_load_state_dict_restores_metadata(self):
        m1 = _Demo()
        m1.alpha = 9.0
        sd = m1.state_dict()

        m2 = _Demo()
        out = m2.load_state_dict(sd, strict=False)
        assert out is m2  # returns self
        assert m2.alpha == 9.0

    def test_set_metadata_dict_skips_unknown_keys(self):
        m = _Demo()
        m._set_metadata_dict({"alpha": 7.0, "ignore_me": 100})
        assert m.alpha == 7.0
        assert not hasattr(m, "ignore_me")


class TestSoftBinner:
    def test_output_shape_2d_input(self):
        sb = SoftBinner(n_bins=5, scale=10.0)
        x = torch.tensor([0.1, 0.5, 0.9])
        out = sb(x)
        assert out.shape == (3, 5)

    def test_centers_evenly_spread(self):
        sb = SoftBinner(n_bins=4)
        # Centers go from 0.05 to 0.95 inclusive
        assert torch.allclose(sb.centers[0], torch.tensor(0.05))
        assert torch.allclose(sb.centers[-1], torch.tensor(0.95))

    def test_logit_is_max_at_nearest_center(self):
        sb = SoftBinner(n_bins=10, scale=100.0)
        x = torch.tensor([0.05])  # exactly at first center
        out = sb(x)
        assert int(out.argmax(dim=-1).item()) == 0

    def test_higher_dim_input_transposes(self):
        sb = SoftBinner(n_bins=3, scale=5.0)
        # input with extra trailing dim (e.g. seq dim) — should transpose internally
        x = torch.rand(4, 6)  # interpreted as (B, T)
        out = sb(x)
        # logits axis moves: from (B, T, n_bins) → (B, n_bins, T)
        assert out.shape == (4, 3, 6)
