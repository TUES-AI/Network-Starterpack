# backprop.py
"""
Minimal JAX trainer for "dumb" Python neural nets.

This module lets students:
  1) define a network as a simple list of dicts:
        {"weights": 2D list [in x out], "biases": 1D list [out]}
     e.g., created from something like your snippet:

        def initialize_network_layer(layer1, layer2):
            weights = []
            biases = []
            for _ in range(layer2):
                biases.append(0)
            for i in range(layer1):
                weights.append([])
                for _ in range(layer2):
                    weights[i].append(0)
            return { "weights": weights, "biases": biases }

        network = []
        network_size = [784, 256, 128, 64, 10]
        for i in range(1, len(network_size)):
            network.append(initialize_network_layer(network_size[i-1],
                                                    network_size[i]))

  2) provide a data function:
        def get_data(split: str, index: int, dir: str="data"):
            # returns (array_like features, int label)
            return x, y

Then just call train(...).

Key features:
- Softmax cross-entropy classification.
- ReLU or Sigmoid hidden activations (last layer is logits).
- JIT-compiled update step with jax.value_and_grad.
- Auto device selection: TPU → GPU → CPU (silent).
- Converts params to/from the simple Python format.

NOTE: No prints/logging in training; keep the classroom output clean.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Any, Sequence, Optional
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp


# -------------------------
# Device selection (silent)
# -------------------------

# backprop.py — replace the old _choose_device() with this

def _choose_device() -> Any:
    """
    Safely pick the first available device without forcing initialization
    of unavailable plugins. Preference: TPU → GPU → METAL → CPU.
    """
    # 1) JAX's default backend is usually the right choice and won't throw.
    try:
        devs = jax.devices()
        if devs:
            return devs[0]
    except Exception:
        pass

    # 2) Otherwise, probe preferred backends, swallowing init failures.
    for platform in ("tpu", "gpu", "metal", "cpu"):
        try:
            devs = jax.devices(platform)
        except Exception:
            devs = []
        if devs:
            return devs[0]

    # 3) Last resort: whatever JAX can give us.
    return jax.devices()[0]


# -------------------------
# Conversions / validation
# -------------------------

Params = List[Dict[str, jnp.ndarray]]

def _to_params_on_device(network: List[Dict[str, Any]],
                         device: Optional[Any] = None,
                         dtype=jnp.float32) -> Params:
    """
    Convert the 'dumb' python network to JAX params on the chosen device.
    Expects each layer to be {"weights": 2D list [in x out], "biases": 1D list [out]}.
    """
    if device is None:
        device = _choose_device()

    params: Params = []
    for i, layer in enumerate(network):
        W = jnp.asarray(layer["weights"], dtype=dtype)
        b = jnp.asarray(layer["biases"], dtype=dtype)
        # Basic shape guards
        if W.ndim != 2:
            raise ValueError(f"Layer {i}: weights must be 2D [in x out], got {W.shape}")
        if b.ndim != 1:
            raise ValueError(f"Layer {i}: biases must be 1D [out], got {b.shape}")
        if W.shape[1] != b.shape[0]:
            raise ValueError(f"Layer {i}: shape mismatch, weights {W.shape}, biases {b.shape}")
        params.append({"W": jax.device_put(W, device), "b": jax.device_put(b, device)})
    return params


def _to_simple_network(params: Params) -> List[Dict[str, Any]]:
    """
    Convert JAX params back to the simple Python (lists) format
    to hand back to students (or to save to JSON).
    """
    net: List[Dict[str, Any]] = []
    for layer in params:
        W = np.asarray(layer["W"])  # move to host
        b = np.asarray(layer["b"])
        net.append({"weights": W.tolist(), "biases": b.tolist()})
    return net


def _validate_first_last_dims(params: Params,
                              get_data: Callable[[str, int, str], Tuple[Any, int]],
                              split: str,
                              num_classes: Optional[int] = None,
                              data_dir: str = "data") -> Tuple[int, int]:
    """
    Peek one sample to validate input dimension and infer (#features, #classes).
    """
    x0, y0 = get_data(split, 0, data_dir)
    x0 = np.asarray(x0)
    if x0.ndim != 1:
        raise ValueError(f"get_data must return a 1D feature vector; got shape {x0.shape}")
    in_dim = x0.shape[0]
    first_W = np.asarray(params[0]["W"])
    if first_W.shape[0] != in_dim:
        raise ValueError(f"Input dim mismatch: sample has {in_dim}, "
                         f"but first layer expects {first_W.shape[0]}")

    out_dim = int(np.asarray(params[-1]["b"]).shape[0])
    if num_classes is not None and num_classes != out_dim:
        raise ValueError(f"num_classes ({num_classes}) != last layer out dim ({out_dim})")
    return in_dim, out_dim


# -------------------------
# Activations
# -------------------------

def _get_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if name.lower() == "relu":
        return jax.nn.relu
    elif name.lower() == "sigmoid":
        return jax.nn.sigmoid
    else:
        raise ValueError(f"Unsupported activation_function '{name}'. "
                         f"Use 'relu' or 'sigmoid'.")


# -------------------------
# Core model pieces
# -------------------------

def _forward(params: Params,
             x: jnp.ndarray,
             activation_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """
    Forward pass: for hidden layers apply activation; last layer returns logits.
    x can be shape [D] or [B, D].
    """
    z = x
    last = len(params) - 1
    for i, layer in enumerate(params):
        z = jnp.dot(z, layer["W"]) + layer["b"]
        if i != last:
            z = activation_fn(z)
    return z


def _cross_entropy_logits(logits: jnp.ndarray, y_idx: jnp.ndarray) -> jnp.ndarray:
    """
    Mean cross-entropy between logits and integer labels.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(y_idx, num_classes=log_probs.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def _make_loss_fn(activation_fn: Callable[[jnp.ndarray], jnp.ndarray]):
    def loss_fn(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        logits = _forward(params, x, activation_fn)
        return _cross_entropy_logits(logits, y)
    return loss_fn


def _make_update_fn(activation_fn: Callable[[jnp.ndarray], jnp.ndarray]):
    loss_fn = _make_loss_fn(activation_fn)

    @jax.jit
    def update(params: Params,
               x: jnp.ndarray,
               y: jnp.ndarray,
               lr: float) -> Tuple[Params, jnp.ndarray]:
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        return new_params, loss
    return update

def _make_batch_accuracy_fn(activation_fn: Callable[[jnp.ndarray], jnp.ndarray]):
    @jax.jit
    def batch_accuracy(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        logits = _forward(params, x, activation_fn)
        y_pred = jnp.argmax(logits, axis=-1)
        return jnp.mean((y_pred == y).astype(jnp.float32))
    return batch_accuracy



# -------------------------
# Data helpers
# -------------------------

def _fetch_batch(get_data: Callable[[str, int, str], Tuple[Any, int]],
                 split: str,
                 indices: Sequence[int],
                 data_dir: str,
                 device: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a batch by calling user-provided get_data() for each index.
    (Kept simple on purpose; students don't need to vectorize their loader.)
    """
    xs = []
    ys = []
    for idx in indices:
        x_i, y_i = get_data(split, int(idx), data_dir)
        xs.append(np.asarray(x_i, dtype=np.float32))
        ys.append(int(y_i))
    X = jnp.asarray(np.stack(xs, axis=0))
    Y = jnp.asarray(np.array(ys, dtype=np.int32))
    # Place on selected device
    return jax.device_put(X, device), jax.device_put(Y, device)


# -------------------------
# Public API
# -------------------------

def train(
    network: List[Dict[str, Any]],
    get_data: Callable[[str, int, str], Tuple[Any, int]],
    *,
    learning_rate: float = 1e-2,
    number_of_epochs: int = 3,
    data_to_train_on: int = 10_000,
    size_data_set: int = 60_000,
    activation_function: str = "relu",
    batch_size: int = 64,
    data_dir: str = "data",
    split: str = "train",
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Train the given 'dumb' network on classification with softmax x-entropy.

    Returns:
      (updated_network_in_simple_format, metrics_dict)

    metrics_dict contains:
      - "final_loss"
      - "device"
      - "epochs"
      - "batches_per_epoch"
    """
    if data_to_train_on <= 0 or size_data_set <= 0:
        raise ValueError("data_to_train_on and size_data_set must be positive.")
    if data_to_train_on > size_data_set:
        raise ValueError("data_to_train_on cannot exceed size_data_set.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    device = _choose_device()
    params = _to_params_on_device(network, device=device)
    activation_fn = _get_activation(activation_function)
    update = _make_update_fn(activation_fn)

    # Validate shapes vs first sample; infer input and output dims.
    _validate_first_last_dims(params, get_data, split, data_dir=data_dir)

    rng = np.random.default_rng(seed)
    batches_per_epoch = max(1, (data_to_train_on + batch_size - 1) // batch_size)

    last_loss = None
    for _epoch in range(number_of_epochs):
        # Pick a subset of dataset indices for this epoch, size = data_to_train_on
        subset = rng.choice(size_data_set, size=data_to_train_on, replace=False)
        # Shuffle for batching
        rng.shuffle(subset)

        # Mini-batch SGD
        for s in range(0, data_to_train_on, batch_size):
            batch_indices = subset[s : s + batch_size]
            Xb, Yb = _fetch_batch(get_data, split, batch_indices, data_dir, device)
            params, last_loss = update(params, Xb, Yb, learning_rate)

    updated_network = _to_simple_network(params)
    metrics = {
        "final_loss": float(last_loss) if last_loss is not None else None,
        "device": str(device),
        "epochs": number_of_epochs,
        "batches_per_epoch": batches_per_epoch,
    }
    return updated_network, metrics


def evaluate(
    network: List[Dict[str, Any]],
    get_data: Callable[[str, int, str], Tuple[Any, int]],
    *,
    size_data_set: int,
    batch_size: int = 256,
    activation_function: str = "relu",
    data_dir: str = "data",
    split: str = "test",
) -> float:
    device = _choose_device()
    params = _to_params_on_device(network, device=device)
    activation_fn = _get_activation(activation_function)

    # NEW: compile the accuracy fn with activation baked in
    batch_accuracy = _make_batch_accuracy_fn(activation_fn)

    correct = 0.0
    total = 0
    for s in range(0, size_data_set, batch_size):
        idxs = range(s, min(s + batch_size, size_data_set))
        Xb, Yb = _fetch_batch(get_data, split, idxs, data_dir, device)
        # CHANGED: call the compiled closure
        acc = batch_accuracy(params, Xb, Yb)
        correct += float(acc) * (Xb.shape[0])
        total += int(Xb.shape[0])
    return correct / max(1, total)


# -------------------------
# Optional: predict helper
# -------------------------

def predict_proba(
    network: List[Dict[str, Any]],
    x: np.ndarray,
    *,
    activation_function: str = "relu",
) -> np.ndarray:
    """
    Predict class probabilities for a single example x (1D feature vector).
    """
    device = _choose_device()
    params = _to_params_on_device(network, device=device)
    activation_fn = _get_activation(activation_function)
    logits = _forward(params, jax.device_put(jnp.asarray(x, dtype=jnp.float32), device), activation_fn)
    probs = jax.nn.softmax(logits, axis=-1)
    return np.asarray(probs)

