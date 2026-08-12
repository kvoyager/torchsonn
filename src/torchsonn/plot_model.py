import os
import functools
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

try:
    import graphviz as gv
except ModuleNotFoundError as exc:  # pragma: no cover - depends on install extras
    raise ModuleNotFoundError(
        "torchsonn.plot_model needs the optional 'graphviz' dependency, which "
        "is not part of the base install. Add it with:\n"
        "    pip install \"torchsonn[viz]\"\n"
        "The Python package also requires the system Graphviz binaries ('dot' "
        "on PATH) — see https://graphviz.org/download/."
    ) from exc


def run_dot_without_conda(args: Sequence[str]) -> subprocess.CompletedProcess:
    # Copy current environment
    env = os.environ.copy()

    # Remove conda-related variables
    for var in ["CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_EXE", "CONDA_SHLVL"]:
        env.pop(var, None)

    # Reset PATH to exclude conda paths
    env["PATH"] = ";".join(
        p for p in env["PATH"].split(";")
        if "conda" not in p.lower() and "anaconda" not in p.lower()
    )

    return subprocess.run(args, env=env, check=True, capture_output=True, text=True)


class PlotModel:
    """Plot self-organizing polynomial neural network (multilayered GMDH)
    """
    def __init__(
        self,
        model: "object",
        filename: str | Path,
        plot_neuron_name: bool = False,
        view: bool = False,
    ) -> None:
        self.g = gv.Digraph(format='svg')
        self.output = 'OUTPUT'
        self.model = model
        self.plot_neuron_name = plot_neuron_name
        self.filename = filename
        self.view = view

        '''
        # custom palette
        self.node_color = '#ccd1ff'
        self.io_node_color = '#bed4eb'
        self.io_pen_color = '#535ebe'
        self.pen_color = '#535ebe'
        self.io_font_color = 'black'
        self.connection_color = '#ccd1ff'
        self.connection_fill_color = '#535ebe'
        '''

        # scikit-learn palette
        self.node_color = '#cde8ef'
        self.io_node_color = '#ff9c34'
        self.pen_color = '#cde8ef'
        self.io_pen_color = '#f89939'
        self.io_font_color = 'white'
        self.connection_color = '#cde8ef'
        self.connection_fill_color = '#3499cd'

        self.io_node_param = ({'style': 'filled', 'color': self.io_pen_color, 'fillcolor': self.io_node_color,
                               'fontsize': '12', 'fontcolor': self.io_font_color, 'rank': 'same'})
        self.node_param = ({'style': 'filled', 'shape': 'rect', 'color': self.pen_color, 'fillcolor': self.node_color,
                               'fontsize': '10'})
        # The out_proj head, when there is one. Styled like the I/O nodes rather
        # than the neuron nodes because it is not a discovered polynomial — it
        # is a fitted linear readout, and the diagram should not suggest
        # otherwise.
        self.head_node_param = ({'style': 'filled', 'shape': 'rect', 'color': self.io_pen_color,
                                 'fillcolor': self.io_node_color, 'fontsize': '11',
                                 'fontcolor': self.io_font_color})
        self.g.node(self.output, **self.io_node_param)
        self.g.graph_attr.update(label='Self-organizing deep learning polynomial neural network\n ', labelloc='t', center='true',
                                 fontsize='18')

    def _get_feature_name(self, index: int) -> str:
        s = f"F{index}"
        if self.model.feature_names is not None and len(self.model.feature_names) > 0:
            s += f"\n{self.model.feature_names[index]}"
        return s

    def _get_neuron_name(self, neuron_idx: int, neuron: "object", neuron_idx_map: "object") -> str:
        s = f"layer {neuron.layer_index}\nneuron {neuron_idx}"
        if self.plot_neuron_name:
            s += f"\n{neuron.get_short_name()}"
        return s

    @staticmethod
    def _get_feature_index(layers: "Sequence[object]", neuron: "object", u_index: int) -> tuple[bool, int]:
        if neuron.layer_index == 0:
            return True, u_index
        else:
            prev_layer = layers[neuron.layer_index - 1]
            if u_index < len(prev_layer):
                return False, u_index
            else:
                return True, u_index - len(prev_layer)

    def add_connection(
        self,
        layers: "Sequence[object]",
        neuron: "object",
        neuron_idx: int,
        u_index: int,
    ) -> None:
        input_is_original_feature, feature_index = self._get_feature_index(layers,  neuron, u_index)
        layer = layers[neuron.layer_index]
        if input_is_original_feature:
            name1 = self._get_feature_name(feature_index)
            name2 = self._get_neuron_name(neuron_idx, neuron, layer.neuron_idxs)
        else:
            prev_layer = layers[neuron.layer_index-1]
            parent_neuron_idx, parent_neuron = prev_layer.get_parent_neron_module(feature_index)
            name1 = self._get_neuron_name(parent_neuron_idx, parent_neuron, prev_layer.neuron_idxs)
            name2 = self._get_neuron_name(neuron_idx, neuron, layer.neuron_idxs)
        return self.add_edge(name1, name2)

    def add_edge(self, a: str, b: str) -> None:
        return self.g.edge(a, b, color=self.connection_color, fillcolor=self.connection_fill_color, weight='1')

    @staticmethod
    def _cumulative_column(layer: "object", module_idx: int, neuron_idx: int) -> int:
        """Column of (module_idx, neuron_idx) in the layer's concatenated output.

        Node labels are keyed on this cumulative index — the loop in plot()
        counts straight through every module — whereas `get_best_neuron_model`
        and `module_idxs` report an index *within* a module. Conflating the two
        mislabels the readout edge whenever the target is not in the first
        module.
        """
        col = 0
        for i, module in enumerate(layer.neuron_models):
            if i == int(module_idx):
                return col + int(neuron_idx)
            col += module.num_neurons
        return col

    @staticmethod
    def _module_for_column(layer: "object", col: int) -> "object":
        """The neuron module owning column `col` — it supplies get_short_name()."""
        offset = 0
        for module in layer.neuron_models:
            if col < offset + module.num_neurons:
                return module
            offset += module.num_neurons
        return layer.neuron_models[-1]

    def _add_readout(self, last_layer: "object") -> None:
        """Draw whatever turns the last layer into the model's prediction.

        Without a head that is a single edge from the best-error neuron — the
        column `SONN.infer` returns. With `out_proj` it is a fitted
        `Linear(in_features, out_features)` over the `in_features` lowest-error
        columns, so drawing only the best neuron would show a formula the model
        does not compute.
        """
        out_proj = getattr(self.model, "out_proj", None)
        if out_proj is None:
            module_idx, neuron_idx = self.model.get_best_neuron_model(last_layer)
            col = self._cumulative_column(last_layer, module_idx, neuron_idx)
            module = last_layer.neuron_models[int(module_idx)]
            self.add_edge(self._get_neuron_name(col, module, last_layer.neuron_idxs), self.output)
            return

        cols = self.model._best_neuron_columns(last_layer, out_proj.in_features)
        label = f"out_proj\nLinear({out_proj.in_features}, {out_proj.out_features})"
        if len(cols) < out_proj.in_features:
            # SONN.infer zero-pads when the layer is narrower than the head.
            label += f"\n({out_proj.in_features - len(cols)} inputs zero-padded)"
        self.g.node(label, **self.head_node_param)
        for col in cols:
            module = self._module_for_column(last_layer, col)
            self.add_edge(self._get_neuron_name(col, module, last_layer.neuron_idxs), label)
        self.add_edge(label, self.output)

    def plot(self) -> None:
        if len(self.model.layers) == 0:
            return

        digraph = functools.partial(gv.Digraph, format='svg')

        features_graph = digraph()
        for i in range(0, self.model.d_model):
            features_graph.node(self._get_feature_name(i), **self.io_node_param)
        self.g.subgraph(features_graph)

        for layer in self.model.layers:
            neuron_idx = 0
            layer_graph = digraph()
            for neuron_module in layer.neuron_models:
                for _ in range(neuron_module.num_neurons):
                    s = self._get_neuron_name(neuron_idx, neuron_module, layer.neuron_idxs)
                    layer_graph.node(s, **self.node_param)
                    neuron_idx += 1
            self.g.subgraph(layer_graph)

        for layer in self.model.layers:
            neuron_idx = 0
            for neuron_module in layer.neuron_models:
                for idx in neuron_module.src_idxs.tolist():
                    for u_index in idx:
                        self.add_connection(self.model.layers, neuron_module, neuron_idx, u_index)
                    neuron_idx += 1

        self._add_readout(self.model.layers[-1])

        dot_path = shutil.which("dot")
        if dot_path is None:
            raise RuntimeError("Graphviz 'dot' executable not found in PATH. "
                               "Please install Graphviz and ensure it's in your PATH.")

        try:
            self.g.render(filename=self.filename, view=self.view)
        except Exception as e:

            run_dot_without_conda(f"dot.exe -Kdot -Tsvg -O {str(self.filename)}".split(" "))
