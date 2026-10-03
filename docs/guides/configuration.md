# Configuration

Every setting of a model and its training lives in one configuration with
two sections, `model` and `train`, plus a few top-level flags. The schema is
the dataclass `torchsonn.config.SONNConfig`; it fixes every key, its type
and its default. You change only what you need, from Python or from YAML,
and the rest keeps its default. [Configuration keys](../reference/config.md)
lists every key.

## From Python

`SONN.default_config()` returns the schema with every default filled in.
Merge your changes into it:

```python
from omegaconf import OmegaConf
from torchsonn import SONN

config = OmegaConf.merge(SONN.default_config(), {
    "model": {"type": "regressor", "ref_functions": ["linear_cov"], "nbest_neurons": 8},
    "train": {"max_layer_count": 10},
})
config.train.device = "cuda"   # single keys can be set afterwards too
```

The result is typed: a key the schema does not have, or a value of the
wrong type, is rejected at once, before any training starts.

```text
omegaconf.errors.ConfigKeyError: Key 'nbest_neuron' not in 'ModelConfig'
omegaconf.errors.ValidationError: Value 'eight' of type 'str' could not be converted to Integer
```

Pass this merged configuration to both `SONN` and `Trainer`. The `Trainer`
reads it as given, so it needs every key the schema defines.

## From YAML, with Hydra

The tutorials load their configuration with
[Hydra](https://hydra.cc/). Importing `torchsonn` registers the schema
with Hydra under the name `default`, so a YAML file can start from it and
list only its changes:

```yaml
# my_model.yaml
defaults:
  - default      # the SONNConfig schema with its defaults
  - _self_       # then this file's values

hydra:
  job:
    chdir: false # keep relative paths relative to where you launch

model:
  type: regressor
  ref_functions:
    - linear_cov
    - legendre:
        degree: 3
  nbest_neurons: 8

train:
  checkpoint_dir: checkpoints/my_model
```

A script reads it with `hydra.main`, as the tutorial scripts do:

```python
import hydra
import torchsonn.config  # registers the schema as "default"

@hydra.main(version_base="1.3", config_path=".", config_name="my_model")
def main(config):
    ...

if __name__ == "__main__":
    main()
```

Values are applied in order: the schema's defaults, then the YAML file, then
the command line.

## Command-line overrides

Any key can be overridden when the script runs:

```bash
python my_script.py train.device=cuda train.max_layer_count=20
python my_script.py train.optimizer.name=adam train.optimizer.optimizer_params.lr=5e-3
python my_script.py model.shortcut.prev_layers=2
python my_script.py "model.ref_functions=[linear_cov,quadratic]"
python my_script.py "model.ref_functions=[{legendre:{degree:4}}]"
python my_script.py --config-name other_config
```

Quote overrides that contain brackets or braces, so the shell passes them
unchanged. An unknown key or a value of the wrong type stops the run with
an error naming the key:

```text
Could not override 'model.nbest_neuron'.
Error merging override model.nbest_neurons=eight
```

## Sweeps

`-m` runs the script once per combination of comma-separated values:

```bash
python -m tutorials.ccpp.ccpp -m train.ridge_alpha=0.001,0.01,0.05
```

Hydra gives every run its own output folder. The checkpoint folder is
whatever `train.checkpoint_dir` says, so runs of a sweep that share it
write into the same folder unless the script separates them.

## Neuron family entries

Each entry of `model.ref_functions` is a family name, or a family name with
a mapping of options:

```yaml
model:
  ref_functions:
    - linear_cov
    - polyquad:
        dim: 4
        squares: true
    - rbf: {centers: 8}
```

[Neuron family options](../reference/config.md#neuron-family-options)
lists every option.

## Settings of a tutorial script

The top-level `tutorial` mapping holds settings that only a tutorial script
reads, such as how it splits and prepares its data. The schema accepts any
keys there. The same holds for `train.batch_size`, `train.shuffle`, `resume`
and the `finetune_*` flags: the tutorial scripts read them, the library
does not (see [Read by the tutorial scripts](../reference/config.md#read-by-the-tutorial-scripts)).

<small>Checked against TorchSONN 0.1.5.</small>
