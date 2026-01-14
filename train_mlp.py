#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    Callback                      # ← for custom metrics
)
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score, precision_score, recall_score
)

# ---------- 全域設定 ----------
trace_num    = 500
samples_num  = trace_num * 32
order        = '2'
opt          = 'o3'                 # o0 or o3
algo         = 'PQClean'            # PQClean
platform     = 'CW308_STM32F4'      # CW308_STM32F3 / CW308_STM32F4 / CWLITEARM
oscilloscope = 'picoscope'          # chipwhisperer / picoscope

os.makedirs('model', exist_ok=True)
os.makedirs('figure/train_model', exist_ok=True)


# ---------- 自訂 Callback：每 epoch 計算 P / R / F1 ----------
class MetricsLogger(Callback):
    """在驗證集上計算 macro-precision / recall / F1，並寫入 logs"""
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data          # (X_val, y_val)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        X_val, y_val = self.validation_data
        y_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
        y_true = np.argmax(y_val, axis=1)

        macro_p  = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_r  = recall_score   (y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score       (y_true, y_pred, average='macro', zero_division=0)

        logs['val_precision'] = macro_p
        logs['val_recall']    = macro_r
        logs['val_f1']        = macro_f1


# ---------- 資料載入 ----------
def load_trace():
    trace_path = (f'traces_train/{trace_num}/'
                  f'traces_unfixed_message_{trace_num}_{oscilloscope}_{platform}_{opt}_{algo}_{order}')

    t = np.load(f'{trace_path}/trace.npy')
    l = np.load(f'{trace_path}/label.npy')

    scaler = StandardScaler()
    t = scaler.fit_transform(t)
    l_encode = to_categorical(l, num_classes=256)

    print('size of trace :', t.shape)
    print('size of label :', l.shape)
    print('size of one-hot label:', l_encode.shape)
    return t, l_encode


def split_data(t, l, test_size=0.2, random_state=42):
    t_train, t_val, l_train, l_val = train_test_split(
        t, l,
        test_size=test_size,
        random_state=random_state,
        stratify=l.argmax(axis=1)
    )

    print('t_train:', t_train.shape, ' | t_val:', t_val.shape)
    print('l_train:', l_train.shape, ' | l_val:', l_val.shape)
    label_distribution(l_train.argmax(axis=1), 'Training Set')
    label_distribution(l_val.argmax(axis=1), 'Validation Set')
    return t_train, t_val, l_train, l_val


# ---------- 建模 ----------
def build_model(input_dim=7313, num_classes=256,
                layer_units=[1024, 512, 256], dropout_rate=0.5):
    model = Sequential()
    model.add(Dense(layer_units[0], input_shape=(input_dim,),
                    activation='relu', kernel_initializer='he_uniform',
                    kernel_regularizer=regularizers.l2(1e-3)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for units in layer_units[1:]:
        model.add(Dense(units, activation='relu',
                        kernel_initializer='he_uniform',
                        kernel_regularizer=regularizers.l2(1e-3)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation='softmax'))
    return model


def compile_model(model, learning_rate=1e-3):
    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    return model


# ---------- 訓練 ----------
def train_model(model, t_train, l_train, t_val, l_val,
                epochs=50, batch_size=256):

    es = EarlyStopping(
        monitor='val_accuracy', patience=20,
        restore_best_weights=True, verbose=1, mode='max'
    )
    mc = ModelCheckpoint(
        'model/best_model.keras', monitor='val_accuracy',
        save_best_only=True, verbose=1, mode='max'
    )
    metrics_logger = MetricsLogger(validation_data=(t_val, l_val))

    history = model.fit(
        t_train, l_train,
        validation_data=(t_val, l_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[es, mc, metrics_logger]
    )
    return history


# ---------- 評估 ----------
def evaluate_model(model, t_val, l_val):
    val_loss, val_acc = model.evaluate(t_val, l_val, verbose=0)

    preds = model.predict(t_val, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(l_val, axis=1)

    macro_p  = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_r  = recall_score   (y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score       (y_true, y_pred, average='macro', zero_division=0)

    print(f"\n===== Validation Metrics =====")
    print(f"Loss      : {val_loss:.4f}")
    print(f"Accuracy  : {val_acc:.4f}")
    print(f"Macro P   : {macro_p :.4f}")
    print(f"Macro R   : {macro_r :.4f}")
    print(f"Macro F1  : {macro_f1:.4f}\n")

    print(classification_report(y_true, y_pred, digits=4))


# ---------- 視覺化 ----------
def label_distribution(labels, name):
    counts = np.bincount(labels, minlength=256)
    plt.figure(figsize=(12, 6))
    plt.bar(range(256), counts, width=1, edgecolor='black', align='center')
    plt.title(f"Label Distribution – {name}")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.savefig(f'figure/train_model/label_distribution_{name}.png')
    plt.close()


def plot_history(history):
    loss      = history.history['loss']
    val_loss  = history.history['val_loss']
    acc       = history.history['accuracy']
    val_acc   = history.history['val_accuracy']
    val_f1    = history.history.get('val_f1')       # ← 可能不存在 (早停前)
    epochs    = range(1, len(loss) + 1)

    plt.figure(figsize=(16, 8))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss,     'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy + F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, 'r-', label='Val Acc')
    if val_f1 is not None:
        plt.plot(epochs, val_f1, 'g-', label='Val F1')
    plt.title('Val Accuracy / F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('figure/train_model/training_history.png')
    plt.close()


# ---------- main ----------
if __name__ == "__main__":
    start = time.time()

    t, l = load_trace()
    t_train, t_val, l_train, l_val = split_data(t, l, test_size=0.2)

    model = build_model(
        input_dim=t.shape[1],
        num_classes=256,
        layer_units=[1024],       # 可自行修改
        dropout_rate=0.3
    )
    model = compile_model(model, learning_rate=1e-4)

    history = train_model(
        model, t_train, l_train, t_val, l_val,
        epochs=300, batch_size=512
    )

    evaluate_model(model, t_val, l_val)
    plot_history(history)

    elapsed = (time.time() - start) / 60
    print(f"\nTotal running time: {elapsed:.2f} minutes")
    print("done")
