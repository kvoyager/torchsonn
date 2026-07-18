import subprocess
import types

import pytest

from torchsonn.plot_model import PlotModel, run_dot_without_conda


class _FakeNeuron:
    def __init__(self, layer_index: int = 0, num_neurons: int = 2):
        self.layer_index = layer_index
        self.num_neurons = num_neurons
        self.src_idxs = []

    def get_short_name(self) -> str:
        return "Fake"


class _FakeLayer:
    def __init__(self, neuron_modules, neuron_idxs=None):
        self.neuron_models = neuron_modules
        self.neuron_idxs = neuron_idxs or []

    def __len__(self) -> int:
        return sum(m.num_neurons for m in self.neuron_models)

    def get_parent_neron_module(self, idx: int):
        # walk-back; only used for layers > 0
        running = 0
        for m in self.neuron_models:
            for i in range(m.num_neurons):
                if running == idx:
                    return i, m
                running += 1
        raise ValueError


class _FakeModel:
    def __init__(self, layers, d_model: int = 4, feature_names=None):
        self.layers = layers
        self.d_model = d_model
        self.feature_names = feature_names

    def get_best_neuron_model(self, layer):
        return 0, 0


@pytest.fixture
def fake_model():
    n1 = _FakeNeuron(layer_index=0, num_neurons=2)
    layer0 = _FakeLayer([n1])
    return _FakeModel([layer0], d_model=3)


class TestPlotModelHelpers:
    def test_get_feature_name_indexed(self, fake_model, tmp_path):
        p = PlotModel(fake_model, filename=tmp_path / "x")
        assert p._get_feature_name(2) == "F2"

    def test_get_feature_name_with_named(self, tmp_path):
        m = _FakeModel([_FakeLayer([_FakeNeuron()])], d_model=3, feature_names=["alpha", "beta", "gamma"])
        p = PlotModel(m, filename=tmp_path / "x")
        assert "alpha" in p._get_feature_name(0)

    def test_get_neuron_name_default(self, fake_model, tmp_path):
        p = PlotModel(fake_model, filename=tmp_path / "x")
        s = p._get_neuron_name(2, _FakeNeuron(layer_index=1), None)
        assert "layer 1" in s

    def test_get_neuron_name_with_short_name(self, fake_model, tmp_path):
        p = PlotModel(fake_model, filename=tmp_path / "x", plot_neuron_name=True)
        s = p._get_neuron_name(2, _FakeNeuron(layer_index=1), None)
        assert "Fake" in s

    def test_get_feature_index_first_layer(self, tmp_path, fake_model):
        n = _FakeNeuron(layer_index=0)
        is_feat, idx = PlotModel._get_feature_index(fake_model.layers, n, 3)
        assert is_feat is True
        assert idx == 3

    def test_get_feature_index_deeper_layer_feature(self, tmp_path):
        # prev layer has len() == 2; u_index >= 2 → original feature
        prev = _FakeLayer([_FakeNeuron(num_neurons=2)])
        curr = _FakeLayer([_FakeNeuron(layer_index=1, num_neurons=1)])
        layers = [prev, curr]
        n = _FakeNeuron(layer_index=1)
        is_feat, idx = PlotModel._get_feature_index(layers, n, 3)
        assert is_feat is True
        assert idx == 1  # 3 - 2

    def test_get_feature_index_deeper_layer_neuron(self):
        prev = _FakeLayer([_FakeNeuron(num_neurons=3)])
        curr = _FakeLayer([_FakeNeuron(layer_index=1, num_neurons=1)])
        layers = [prev, curr]
        n = _FakeNeuron(layer_index=1)
        is_feat, idx = PlotModel._get_feature_index(layers, n, 1)
        assert is_feat is False
        assert idx == 1


class TestPlot:
    def test_plot_empty_model_short_circuits(self, tmp_path):
        m = _FakeModel(layers=[], d_model=2)
        p = PlotModel(m, filename=tmp_path / "out")
        # Should return without raising or calling render
        p.plot()

    def test_plot_missing_dot_raises(self, tmp_path, monkeypatch):
        n = _FakeNeuron(layer_index=0, num_neurons=1)
        # src_idxs needs .tolist() iteration → use a tiny list
        n.src_idxs = MockTensor([[0, 1]])
        layer = _FakeLayer([n], neuron_idxs=None)
        m = _FakeModel([layer], d_model=3)
        p = PlotModel(m, filename=tmp_path / "out")
        monkeypatch.setattr("torchsonn.plot_model.shutil.which", lambda _: None)
        with pytest.raises(RuntimeError, match="Graphviz"):
            p.plot()

    def test_plot_happy_path(self, tmp_path, monkeypatch):
        n = _FakeNeuron(layer_index=0, num_neurons=1)
        n.src_idxs = MockTensor([[0, 1]])
        layer = _FakeLayer([n])
        m = _FakeModel([layer], d_model=3)
        p = PlotModel(m, filename=str(tmp_path / "out"))
        monkeypatch.setattr("torchsonn.plot_model.shutil.which", lambda _: "/usr/bin/dot")
        # graphviz.Digraph.render writes to disk; replace it with a no-op.
        called = {"render": 0}
        monkeypatch.setattr(p.g, "render", lambda **_k: called.update(render=called["render"] + 1))
        p.plot()
        assert called["render"] == 1


class TestAddConnection:
    def test_first_layer_feature_edge(self, tmp_path, fake_model):
        p = PlotModel(fake_model, filename=tmp_path / "x")
        edges: list[tuple[str, str]] = []
        p.add_edge = lambda a, b: edges.append((a, b))  # type: ignore[assignment]
        n = _FakeNeuron(layer_index=0)
        p.add_connection(fake_model.layers, n, neuron_idx=0, u_index=2)
        assert len(edges) == 1
        assert "F2" in edges[0][0]

    def test_deeper_layer_neuron_to_neuron(self, tmp_path):
        prev_n = _FakeNeuron(layer_index=0, num_neurons=2)
        prev = _FakeLayer([prev_n])
        curr_n = _FakeNeuron(layer_index=1, num_neurons=1)
        curr = _FakeLayer([curr_n])
        m = _FakeModel([prev, curr], d_model=3)
        p = PlotModel(m, filename=tmp_path / "x")
        edges: list[tuple[str, str]] = []
        p.add_edge = lambda a, b: edges.append((a, b))  # type: ignore[assignment]
        # u_index=1 lands inside prev's 2 neurons (not a shortcut feature)
        p.add_connection(m.layers, curr_n, neuron_idx=0, u_index=1)
        assert len(edges) == 1


class TestPlotRenderFallback:
    def test_render_exception_triggers_run_dot_without_conda(self, tmp_path, monkeypatch):
        n = _FakeNeuron(layer_index=0, num_neurons=1)
        n.src_idxs = MockTensor([[0, 1]])
        layer = _FakeLayer([n])
        m = _FakeModel([layer], d_model=3)
        p = PlotModel(m, filename=str(tmp_path / "out"))
        monkeypatch.setattr("torchsonn.plot_model.shutil.which", lambda _: "/usr/bin/dot")

        def boom(**_k):
            raise RuntimeError("graphviz rendering failed")

        monkeypatch.setattr(p.g, "render", boom)

        called: dict[str, list[str]] = {"args": []}

        def fake_runner(args):
            called["args"] = list(args)

            class R:
                returncode = 0

            return R()

        monkeypatch.setattr("torchsonn.plot_model.run_dot_without_conda", fake_runner)
        p.plot()
        assert called["args"][0].endswith("dot.exe")


class TestRunDotWithoutConda:
    def test_strips_conda_from_env(self, monkeypatch):
        captured = {}

        def fake_run(args, env, check, capture_output, text):
            captured["args"] = list(args)
            captured["env"] = env
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setenv("CONDA_PREFIX", "/conda")
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
        monkeypatch.setenv("PATH", "C:/conda/bin;C:/anaconda/bin;C:/normal/bin")
        monkeypatch.setattr("torchsonn.plot_model.subprocess.run", fake_run)

        result = run_dot_without_conda(["dot", "-V"])
        assert "CONDA_PREFIX" not in captured["env"]
        assert "conda" not in captured["env"]["PATH"]
        assert "anaconda" not in captured["env"]["PATH"]
        assert "C:/normal/bin" in captured["env"]["PATH"]
        assert result.returncode == 0


class MockTensor:
    """Minimal stand-in for a torch tensor's .tolist() iteration in PlotModel.plot."""
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data
