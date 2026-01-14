"""
Hyperparameter‑tuning & epoch‑sensitivity script for MLP side‑channel attack
============================================================================
* **Dynamic input dimension** — determined from trace file at runtime.
* Uses **Keras‑Tuner** (Bayesian Optimization) to search:
  - hidden depth (2–4) & width (512‒4096)
  - dropout rate (0.2–0.5)
  - activation ∈ {relu, swish, gelu}
  - optimizer ∈ {adam, rmsprop, sgd}
  - learning‑rate ∈ {1e‑4, 5e‑4, 1e‑3}
  - loss ∈ {categorical_crossentropy, mean_squared_error}
* Retrains the best HP on **train + val** once。
* `epoch_sweep()`：以 epochs ∈ [50,100,150,200,300] 重新訓練並比較泛化。

Run ───────────────────────────────────────────────────────────────────
$ pip install --upgrade keras-tuner
$ python hyperparameter_tuning_mlp.py
"""

from __future__ import annotations
import os, time, json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import keras_tuner as kt

# ─── Experiment flags ────────────────────────────────────────────────
TRACE_NUM    = 500
GROUP        = '2'
OPT_LEVEL    = "o3"           # "o0" or "o3"
ALGO         = "PQClean"
PLATFORM     = "CW308_STM32F3"  # CW308_STM32F4 | CWLITEARM
OSCILLOSCOPE = "picoscope"       # chipwhisperer | picoscope
TEST_SIZE    = 0.2
RANDOM_STATE = 42
EPOCH_LIST   = [50, 100, 150, 200, 300]

INPUT_DIM: int | None = None  # set after loading data

# ─── Data I/O ─────────────────────────────────────────────────────────

def load_trace():
    """Load trace/label numpy arrays and standardize traces."""
    global INPUT_DIM
    trace_dir = f"traces_train/{TRACE_NUM}/traces_unfixed_message_{TRACE_NUM}_{OSCILLOSCOPE}_{PLATFORM}_{OPT_LEVEL}_{ALGO}_{GROUP}"
    t = np.load(os.path.join(trace_dir, "trace.npy"), mmap_mode="r")
    l = np.load(os.path.join(trace_dir, "label.npy"), mmap_mode="r")
    INPUT_DIM = t.shape[1]
    t_std = StandardScaler().fit_transform(t)
    l_oh  = to_categorical(l, num_classes=256)
    return t_std, l_oh


def split_data(t, l):
    return train_test_split(t, l, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=l.argmax(axis=1))

# ─── Model builder (for Keras‑Tuner) ──────────────────────────────────

def model_builder(hp: kt.HyperParameters):
    model = Sequential()
    act = hp.Choice("activation", ["relu", "swish", "gelu"], default="relu")

    for i in range(hp.Int("num_layers", 2, 4, default=3)):
        units = hp.Choice(f"units_{i}", [4096, 2048, 1024, 512], default=1024)
        model.add(
            Dense(units, activation=act, kernel_initializer="he_uniform", input_dim=INPUT_DIM if i == 0 else None)
        )
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1, default=0.3)))

    model.add(Dense(256, activation="softmax"))

    lr = hp.Choice("lr", [1e-4, 5e-4, 1e-3], default=1e-4)
    opt = {
        "adam": Adam(lr),
        "rmsprop": RMSprop(lr),
        "sgd": SGD(lr, momentum=0.9, nesterov=True),
    }[hp.Choice("optimizer", ["adam", "rmsprop", "sgd"], default="adam")]

    loss_fn = CategoricalCrossentropy() if hp.Choice("loss", ["categorical", "mse"], default="categorical") == "categorical" else "mse"

    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    return model

# ─── Tuner & helpers ─────────────────────────────────────────────────

def run_tuner(t_train, l_train):
    tuner = kt.BayesianOptimization(model_builder, objective="val_accuracy", max_trials=50, directory="keras_tuner_logs", project_name=f"mlp_{PLATFORM.lower()}_{OSCILLOSCOPE}")
    callbacks = [EarlyStopping("val_accuracy", patience=20, restore_best_weights=True, mode="max"), ReduceLROnPlateau("val_loss", factor=0.5, patience=7, min_lr=1e-5)]
    tuner.search(t_train, l_train, epochs=200, batch_size=512, validation_split=0.2, callbacks=callbacks, verbose=2)
    return tuner, tuner.get_best_hyperparameters(1)[0]


def retrain_best(best_hp, t_full, l_full, *, epochs=300):
    model = model_builder(best_hp)
    cb = [EarlyStopping("val_accuracy", patience=30, restore_best_weights=True, mode="max"), ReduceLROnPlateau("val_loss", factor=0.5, patience=10, min_lr=1e-5)]
    history = model.fit(t_full, l_full, epochs=epochs, batch_size=best_hp.Choice("batch_size", [256,512,1024], default=512), validation_split=0.1, callbacks=cb, verbose=2)
    return model, history

# ─── Epoch sweep ─────────────────────────────────────────────────────

def epoch_sweep(best_hp, t_train, t_val, l_train, l_val):
    res = {}
    for ep in EPOCH_LIST:
        print(f"\n[Epoch Sweep] {ep} epochs")
        model, _ = retrain_best(best_hp, t_train, l_train, epochs=ep)
        _, acc = model.evaluate(t_val, l_val, verbose=0)
        res[ep] = float(acc)
        print(f"  val_acc = {acc:.4f}")
    return res

# ─── Plot helpers ────────────────────────────────────────────────────

def plot_history(hist, path="figure/tuner/training_history.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.plot(hist.history["loss"], label="train"); plt.plot(hist.history["val_loss"], label="val"); plt.title("Loss"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(hist.history["accuracy"], label="train"); plt.plot(hist.history["val_accuracy"], label="val"); plt.title("Accuracy"); plt.legend()
    plt.savefig(path); plt.close()


def plot_epoch_sweep(res, path="figure/tuner/epoch_sweep.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ks, vs = zip(*sorted(res.items()))
    plt.figure(); plt.plot(ks, vs, marker="o"); plt.xlabel("epochs"); plt.ylabel("val acc"); plt.title("Epoch sweep"); plt.grid(True); plt.savefig(path); plt.close()

# ─── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    traces, labels = load_trace()
    t_tr, t_val, l_tr, l_val = split_data(traces, labels)

    tuner, best_hp = run_tuner(t_tr, l_tr)
    print("\nBest HP:\n", json.dumps(best_hp.values, indent=2))

    best_model, hist = retrain_best(best_hp, np.vstack([t_tr, t_val]), np.vstack([l_tr, l_val]))
    v_loss, v_acc = best_model.evaluate(t_val, l_val, verbose=0)
    print(f"\nHold‑out acc={v_acc:.4f}, loss={v_loss:.4f}")
    plot_history(hist)

    preds = best_model.predict(t_val)
    print(classification_report(l_val.argmax(1), preds.argmax(1), digits=4))

    sweep_res = epoch_sweep(best_hp, t_tr, t_val, l_tr, l_val)
    plot_epoch_sweep(sweep_res)
    print("Epoch sweep:", sweep_res)

    print(f"Total runtime: {(time.time()-t0)/60:.2f} min")
