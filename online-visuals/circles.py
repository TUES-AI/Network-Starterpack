"""
Colab-ready entry for the interactive MLP decision boundary demo on make_circles.

Original references:
- /mnt/data/circles.py
- /mnt/data/interactive_mlp.py
- /mnt/data/Config.yml  (optional; overrides defaults for network/viz/training)
"""

import yaml
import numpy as np
from sklearn.datasets import make_circles
from IPython.display import display  # noqa: F401  (display is used by the viz module)
from interactive_mlp import create_interactive_mlp_visualization_with_config

# Try to load training section from Config.yml if present.
def _load_training_config():
    paths = [
        "Config.yml",
        "/mnt/data/Config.yml",
    ]
    for p in paths:
        try:
            with open(p, "r") as f:
                cfg = yaml.safe_load(f) or {}
                t = cfg.get("training", {}) or {}
                n_samples = int(t.get("n_samples", 400))
                random_state = int(t.get("random_state", 0))
                noise = float(t.get("noise", 0.10))
                factor = float(t.get("factor", 0.30))
                return dict(n_samples=n_samples, random_state=random_state, noise=noise, factor=factor)
        except FileNotFoundError:
            continue
    # Defaults if no config is present
    return dict(n_samples=400, random_state=0, noise=0.10, factor=0.30)


def main():
    tcfg = _load_training_config()
    X, y = make_circles(
        n_samples=tcfg["n_samples"],
        noise=tcfg["noise"],
        factor=tcfg["factor"],
        random_state=tcfg["random_state"],
    )
    create_interactive_mlp_visualization_with_config(
        X, y, title="MLP decision boundary on Circles Dataset"
    )


if __name__ == "__main__":
    main()

