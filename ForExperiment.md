Expirement

```bash
 python run_activation_experiment.py -s ~/datasets/lego -o output/lego_pair_seed0 -r 4 --white_background --seed 0 --pruning_target 10000
```

`~/datasets/lego` is the dataset path

output: `.csv` file

`--sweep` is to use existed training results

dataset:

``` bash
python -m pip install huggingface_hub
```

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path

snapshot_download(
    repo_id="rishitdagli/nerf-gs-datasets",
    repo_type="dataset",
    allow_patterns="lego/*",
    local_dir=str(Path.home() / "datasets")
)
PY
```

change the `allow_patterns="lego/*",` to download other scenes.

https://huggingface.co/datasets/rishitdagli/nerf-gs-datasets/tree/main