# interactive_mlp.py  (drop into online-visuals/)
# Runner (in Colab) should do:
#   from google.colab import output; output.enable_custom_widget_manager()
#   %pip -q install ipympl ipywidgets
#   %matplotlib widget
# Do NOT use `%matplotlib inline` with this.

from __future__ import annotations
import warnings, os
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as mw
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    import yaml
except Exception:
    yaml = None

# Keep widget & callback refs alive in notebooks (prevents GC => dead buttons)
_WIDGET_REFS = {}

# --------------------------------------------------------------------------------------
# Public API expected by dataset scripts
# --------------------------------------------------------------------------------------

def create_interactive_mlp_visualization_with_config(*args, **kwargs):
    """
    Flexible signatures:
      - create_interactive_mlp_visualization_with_config(config, X, y, title=...)
      - create_interactive_mlp_visualization_with_config(X, y, title=...)
      - create_interactive_mlp_visualization_with_config(config_dict_or_path, title=...)  # will synthesize data if config includes dataset hints
    Also supports keyword args X=..., Y=..., title=...
    """
    config, X, Y, title = _normalize_signature(*args, **kwargs)
    cfg = _load_config(config)

    # Defaults from config (with safe fallbacks)
    epochs = _get(cfg, ["network", "max_iter"], default=_get(cfg, ["training", "max_iter"], 2000))
    lr     = _get(cfg, ["network", "learning_rate_init"], default=_get(cfg, ["training", "learning_rate"], 1e-3))
    hidden = tuple(_get(cfg, ["network", "hidden_layer_sizes"], default=[10])) or (10,)

    # If no data provided but config hints a dataset, synthesize circles
    if X is None or Y is None:
        ds = str(_get(cfg, ["dataset", "name"], default="")).lower()
        if ds == "circles" or (_get(cfg, ["training", "n_samples"], None) is not None and _get(cfg, ["dataset", "name"], "") == ""):
            from sklearn.datasets import make_circles
            n_samples = int(_get(cfg, ["training", "n_samples"], 400))
            noise     = float(_get(cfg, ["training", "noise"], 0.1))
            factor    = float(_get(cfg, ["training", "factor"], 0.3))
            rs        = int(_get(cfg, ["training", "random_state"], 0))
            X, Y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=rs)
            if title is None:
                title = "Circles"
        else:
            raise ValueError("X/Y not provided and config has no dataset instructions. Pass X, Y.")

    # Visualization bounds
    x_range = _get(cfg, ["visualization", "x_range"], None)
    y_range = _get(cfg, ["visualization", "y_range"], None)
    grid_res = int(_get(cfg, ["visualization", "grid_resolution"], 300))

    params = {
        "initial_epochs": int(epochs),
        "initial_lr":     float(lr),
        "initial_hidden": tuple(int(v) for v in hidden if int(v) > 0) or (10,),
        "grid_resolution": grid_res,
    }
    if title is None:
        title = str(_get(cfg, ["title"], "Interactive MLP"))

    return _launch(X, Y, title=title, x_range=x_range, y_range=y_range, **params)


# Back-compat / convenience entry points
def create_interactive_mlp_visualization(X, Y, title: str = "Interactive MLP"):
    return _launch(X, Y, title=title)

def run(X, Y, title: str = "Interactive MLP"):
    return _launch(X, Y, title=title)

def render(X, Y, title: str = "Interactive MLP"):
    return _launch(X, Y, title=title)

def launch(X, Y, title: str = "Interactive MLP"):
    return _launch(X, Y, title=title)

def start(X, Y, title: str = "Interactive MLP"):
    return _launch(X, Y, title=title)

def interactive_mlp(X, Y, title: str = "Interactive MLP"):
    return _launch(X, Y, title=title)

# --------------------------------------------------------------------------------------
# Signature normalization
# --------------------------------------------------------------------------------------

def _looks_like_X(obj):
    try:
        arr = np.asarray(obj)
        return arr.ndim == 2 and arr.shape[1] == 2 and np.issubdtype(arr.dtype, np.number)
    except Exception:
        return False

def _looks_like_Y(obj):
    try:
        arr = np.asarray(obj)
        return arr.ndim == 1 and np.issubdtype(arr.dtype, np.number)
    except Exception:
        return False

def _normalize_signature(*args, **kwargs):
    """
    Returns (config, X, Y, title) parsed from flexible inputs.
    """
    config = kwargs.pop("config", None)
    X = kwargs.pop("X", kwargs.pop("x", None))
    Y = kwargs.pop("Y", kwargs.pop("y", None))
    title = kwargs.pop("title", None)

    # Positional parsing
    if len(args) >= 1:
        a0 = args[0]
        if _looks_like_X(a0):   # (X, Y, [title])
            X = a0
            if len(args) >= 2 and _looks_like_Y(args[1]):
                Y = args[1]
            if len(args) >= 3 and isinstance(args[2], (str, bytes)):
                title = args[2]
        else:
            config = a0
            if len(args) >= 2 and _looks_like_X(args[1]):
                X = args[1]
            if len(args) >= 3 and _looks_like_Y(args[2]):
                Y = args[2]
            if len(args) >= 4 and isinstance(args[3], (str, bytes)):
                title = args[3]

    if kwargs:
        # anything left is unexpected but harmless
        pass

    return config, X, Y, title

# --------------------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------------------

def _load_config(config):
    if isinstance(config, dict) or config is None:
        return config or {}
    if isinstance(config, str):
        if yaml is None:
            raise ImportError("PyYAML is required to load a YAML path as config.")
        if not os.path.exists(config):
            raise FileNotFoundError(f"Config path not found: {config}")
        with open(config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if hasattr(config, "get"):  # dict-like
        return dict(config)
    raise TypeError("config must be a dict, YAML path string, or None.")

def _get(cfg, keys, default=None):
    cur = cfg
    try:
        for k in keys:
            if cur is None:
                return default
            cur = cur[k]
        return cur if cur is not None else default
    except Exception:
        return default

# --------------------------------------------------------------------------------------
# Core visualization
# --------------------------------------------------------------------------------------

def _launch(
    X, Y,
    title: str = "Interactive MLP",
    initial_epochs: int = 2000,
    initial_lr: float = 1e-3,
    initial_hidden: tuple[int, ...] = (10,),
    x_range=None, y_range=None,
    grid_resolution: int = 300,
):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=int)
    assert X.ndim == 2 and X.shape[1] == 2, "X must be (n, 2)"
    assert Y.ndim == 1 and len(Y) == len(X), "Y must be (n,) aligned with X"

    # Ranges & padding
    pad = 0.2
    if x_range is None:
        x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    else:
        x_min, x_max = float(x_range[0]), float(x_range[1])
    if y_range is None:
        y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    else:
        y_min, y_max = float(y_range[0]), float(y_range[1])

    # User-added points
    X_user0: list[list[float]] = []
    X_user1: list[list[float]] = []

    # ---- Figure & axes
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.28)  # leave space for controls
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    # Original data
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=18, label="Class 0 (original)")
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=18, marker="x", label="Class 1 (original)")
    ax.legend(loc="upper right")

    # On-figure help
    fig.text(
        0.02, 0.02,
        "=== INTERACTIVE ===\n"
        " Left click: Class 0 point (dot)\n"
        " Right click OR Shift+Left click: Class 1 point (cross)\n"
        " TRAIN: retrain & update boundary\n"
        " CLEAR ADDED POINTS: remove user points",
        family="monospace", fontsize=9
    )

    # ---- Controls
    fig.text(0.10, 0.12, "Epochs:", ha="right", va="center")
    fig.text(0.55, 0.12, "Learning Rate:", ha="right", va="center")
    fig.text(0.10, 0.07, "Network Size:", ha="right", va="center")

    ax_epochs = plt.axes([0.12, 0.115, 0.18, 0.05])
    ax_lr     = plt.axes([0.57, 0.115, 0.18, 0.05])
    ax_hid    = plt.axes([0.12, 0.065, 0.28, 0.05])

    tb_epochs = mw.TextBox(ax_epochs, "", initial=str(int(initial_epochs)))
    tb_lr     = mw.TextBox(ax_lr,     "", initial=f"{float(initial_lr):g}")
    tb_hidden = mw.TextBox(ax_hid,    "", initial=" ".join(str(int(h)) for h in initial_hidden))

    ax_btn_clear = plt.axes([0.12, 0.005, 0.28, 0.05])
    ax_btn_train = plt.axes([0.70, 0.055, 0.18, 0.05])
    btn_clear = mw.Button(ax_btn_clear, "CLEAR ADDED POINTS")
    btn_train = mw.Button(ax_btn_train, "TRAIN")

    # Keep strong references
    _WIDGET_REFS.update(
        fig=fig, ax=ax,
        tb_epochs=tb_epochs, tb_lr=tb_lr, tb_hidden=tb_hidden,
        btn_clear=btn_clear, btn_train=btn_train,
        ax_epochs=ax_epochs, ax_lr=ax_lr, ax_hid=ax_hid,
        ax_btn_clear=ax_btn_clear, ax_btn_train=ax_btn_train
    )

    # Internal state
    _STATE = {"contour": None, "scat_user0": None, "scat_user1": None, "clf": None}

    def _parse_hidden(s: str):
        s = (s or "").strip().replace(",", " ")
        vals = []
        for tok in s.split():
            try:
                v = int(tok)
                if v > 0:
                    vals.append(v)
            except Exception:
                pass
        return tuple(vals) or (10,)

    def _params():
        try:
            epochs = int(float(tb_epochs.text))
        except Exception:
            epochs = int(initial_epochs)
        try:
            lr = float(tb_lr.text)
        except Exception:
            lr = float(initial_lr)
        hidden = _parse_hidden(tb_hidden.text)
        return epochs, lr, hidden

    def _combine():
        if X_user0 or X_user1:
            add0 = np.array(X_user0, dtype=float) if X_user0 else np.zeros((0, 2))
            add1 = np.array(X_user1, dtype=float) if X_user1 else np.zeros((0, 2))
            X_all = np.vstack([X, add0, add1])
            Y_all = np.concatenate([Y, np.zeros(len(add0), dtype=int), np.ones(len(add1), dtype=int)])
            return X_all, Y_all
        return X, Y

    def _plot_boundary(clf):
        # remove old contour
        if _STATE["contour"] is not None:
            for c in _STATE["contour"].collections:
                c.remove()
            _STATE["contour"] = None

        gx = np.linspace(x_min, x_max, grid_resolution)
        gy = np.linspace(y_min, y_max, grid_resolution)
        XX, YY = np.meshgrid(gx, gy)
        grid = np.c_[XX.ravel(), YY.ravel()]
        try:
            Z = clf.predict_proba(grid)[:, 1]
        except Exception:
            Z = clf.predict(grid).astype(float)
        Z = Z.reshape(XX.shape)

        _STATE["contour"] = ax.contourf(XX, YY, Z, levels=50, alpha=0.6, cmap="RdBu_r", antialiased=True)

        # refresh user points
        if _STATE["scat_user0"] is not None:
            _STATE["scat_user0"].remove()
            _STATE["scat_user0"] = None
        if _STATE["scat_user1"] is not None:
            _STATE["scat_user1"].remove()
            _STATE["scat_user1"] = None

        if X_user0:
            d0 = np.array(X_user0)
            _STATE["scat_user0"] = ax.scatter(d0[:, 0], d0[:, 1], s=30, edgecolor="k", linewidth=0.5)
        if X_user1:
            d1 = np.array(X_user1)
            _STATE["scat_user1"] = ax.scatter(d1[:, 0], d1[:, 1], s=40, marker="x", linewidth=1.0)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")

    def train_and_plot():
        epochs, lr, hidden = _params()
        X_all, Y_all = _combine()

        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                learning_rate_init=lr,
                max_iter=int(epochs),
                batch_size=min(512, len(X_all)),
                random_state=0,
                verbose=False,
            ),
        )
        clf.fit(X_all, Y_all)
        _STATE["clf"] = clf
        acc = float((clf.predict(X_all) == Y_all).mean())
        ax.set_title(f"{title}  |  hidden={hidden}  epochs={int(epochs)}  lr={lr:g}  acc={acc:.3f}")
        _plot_boundary(clf)
        fig.canvas.draw_idle()

    # Initial render
    train_and_plot()

    # Interactions
    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        # Left click → Class 0 unless Shift is held
        if event.button == 1 and not (event.key and "shift" in str(event.key).lower()):
            X_user0.append([event.xdata, event.ydata])
            print(f"Added Class 0 at ({event.xdata:.2f}, {event.ydata:.2f})")
        # Right click OR Shift+Left click → Class 1
        elif event.button == 3 or (event.button == 1 and event.key and "shift" in str(event.key).lower()):
            X_user1.append([event.xdata, event.ydata])
            print(f"Added Class 1 at ({event.xdata:.2f}, {event.ydata:.2f})")
        else:
            return
        train_and_plot()

    def _on_train_click(_):
        print("[TRAIN]")
        train_and_plot()

    def _on_clear_click(_):
        print("[CLEAR ADDED POINTS]")
        X_user0.clear()
        X_user1.clear()
        train_and_plot()

    # Connect & keep refs
    _WIDGET_REFS["cid_click"]  = fig.canvas.mpl_connect("button_press_event", on_click)
    _WIDGET_REFS["cid_train"]  = btn_train.on_clicked(_on_train_click)
    _WIDGET_REFS["cid_clear"]  = btn_clear.on_clicked(_on_clear_click)

    plt.show()
    return {
        "figure": fig,
        "axis": ax,
        "get_params": _params,
        "train_and_plot": train_and_plot,
        "clf": lambda: _STATE.get("clf", None),
    }

# --------------------------------------------------------------------------------------
# Standalone demo (optional local run)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_circles
    X_demo, Y_demo = make_circles(n_samples=400, noise=0.1, factor=0.3, random_state=0)
    create_interactive_mlp_visualization_with_config(X_demo, Y_demo, title="Circles (demo)")

