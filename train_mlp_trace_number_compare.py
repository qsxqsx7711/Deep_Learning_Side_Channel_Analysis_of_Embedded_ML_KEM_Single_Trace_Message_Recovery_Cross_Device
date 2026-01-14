#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

# ---------- 全域參數 ----------
TRACE_NUMS    = [1000, 1500, 2000]
MODES         = ['PoI'] # , PoI, Segment
OPT           = 'o3'
ALGO          = 'PQClean'
PLATFORM      = 'CW308_STM32F4'
OSCILLOSCOPE  = 'picoscope'
BATCH_SIZE    = 512
EPOCHS        = 150
LR            = 1e-4
LAYER_UNITS   = [2048, 512]
DROPOUT_RATE  = 0.5

# ---------- 自訂 Callback：每 epoch 計算 val P / R / F1 ----------
class MetricsLogger(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        X_val, y_val = self.validation_data
        y_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
        y_true = np.argmax(y_val, axis=1)

        logs['val_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        logs['val_recall']    = recall_score   (y_true, y_pred, average='macro', zero_division=0)
        logs['val_f1']        = f1_score       (y_true, y_pred, average='macro', zero_division=0)

# ---------- 工具函式 ----------
def load_trace(trace_num, mode):
    """
    mode == 'poi'     => folder = traces_unfixed_message_{trace_num}_...
    mode == 'segment' => folder = traces_unfixed_message_{trace_num}_..._segment
    """
    base   = './traces_train/best_traces'
    suffix = '_segment' if mode == 'Segment' else '_poi'
    folder = (
        f"traces_unfixed_message_{trace_num}"
        f"_{OSCILLOSCOPE}_{PLATFORM}_{OPT}_{ALGO}{suffix}"
    )
    path = os.path.join(base, folder)

    X_raw = np.load(os.path.join(path, 'trace.npy'))
    y_raw = np.load(os.path.join(path, 'label.npy'))

    scaler = StandardScaler().fit(X_raw)
    X      = scaler.transform(X_raw)
    y      = to_categorical(y_raw, num_classes=256)
    return X, y, scaler

def split_data(X, y, test_size=0.2):
    return train_test_split(
        X, y, test_size=test_size,
        stratify=np.argmax(y, axis=1),
        random_state=42
    )

def build_model(input_dim, num_classes=256):
    model = Sequential()
    model.add(Dense(
        LAYER_UNITS[0],
        input_shape=(input_dim,),
        use_bias=False,
        kernel_initializer='he_uniform',
        kernel_regularizer=regularizers.l2(1e-3)
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))

    # 如果 LAYER_UNITS 還有第二、第三個元素，就用迴圈新增更多層
    for units in LAYER_UNITS[1:]:
        model.add(Dense(
            units,
            use_bias=False,
            kernel_initializer='he_uniform',
            kernel_regularizer=regularizers.l2(1e-3)
        ))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT_RATE))

    # 最後加上輸出層
    model.add(Dense(num_classes, activation='softmax'))
    return model

def compile_model(model):
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=LR),
        metrics=['accuracy']
    )
    return model

def train_one_trace_num(trace_num, mode):
    print(f"\n=== Training {mode.upper()} | TRACE_NUM={trace_num} ===")
    X, y, scaler = load_trace(trace_num, mode)
    Xt, Xv, yt, yv = split_data(X, y)

    # Save scaler
    scaler_dir = f'model/{PLATFORM}/train_mlp_trace_number_compare/compare_{mode}/{trace_num}/scalers'
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, 'scaler.pkl'))

    model = compile_model(build_model(X.shape[1]))

    print("\n--- Model Architecture ---")
    model.summary()
    print("--------------------------\n")
    
    # Callbacks
    model_dir = f'model/{PLATFORM}/train_mlp_trace_number_compare/compare_{mode}/{trace_num}'
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, 'best_model.keras')

    metrics_cb = MetricsLogger(validation_data=(Xv, yv))
    mc = ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                         save_best_only=True, mode='max', verbose=1)

    history = model.fit(
        Xt, yt,
        validation_data=(Xv, yv),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[metrics_cb, mc],
        verbose=0
    )

    # Save history
    hist_dir = f'figure/train_mlp_trace_number_compare/{PLATFORM}/compare_{mode}/{trace_num}'
    os.makedirs(hist_dir, exist_ok=True)
    df_hist = pd.DataFrame(history.history)
    df_hist.to_csv(os.path.join(hist_dir, 'history.csv'), index_label='epoch')

    best_acc = df_hist['val_accuracy'].max()
    best_f1  = df_hist['val_f1'].max()
    print(f"-> best val_acc: {best_acc:.4f} | best F1: {best_f1:.4f}")

    K.clear_session()
    return best_acc, best_f1, df_hist

# ---------- main ----------
if __name__ == "__main__":
    start = time.time()

    best_acc, best_f1, histories = {}, {}, {}

    for mode in MODES:
        for tn in TRACE_NUMS:
            acc, f1, hist = train_one_trace_num(tn, mode)
            best_acc[(mode, tn)] = acc
            best_f1 [(mode, tn)] = f1
            histories[(mode, tn)] = hist

    # Export summary CSV
    idx = pd.MultiIndex.from_tuples(best_acc.keys(), names=['mode','TRACE_NUM'])
    summary_df = pd.DataFrame({
        'best_val_accuracy': best_acc,
        'best_val_f1'      : best_f1
    }, index=idx)
    summary_df.to_csv(f'figure/train_mlp_trace_number_compare/{PLATFORM}/summary_by_mode_and_trace_num.csv', float_format='%.4f')
    print("\n=== Summary ===")
    print(summary_df)

    # Plot all curves together
    fig_dir = f'figure/train_mlp_trace_number_compare/{PLATFORM}/compare_summary'
    os.makedirs(fig_dir, exist_ok=True)

    # 定義不同的標記符號和顏色 (為3條線準備)
    markers = ['o', 's', '^']  # 圓形, 正方形, 三角形(上)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 藍色, 橙色, 綠色

    # Accuracy
    plt.figure(figsize=(8, 6))
    for i, ((mode, tn), hist) in enumerate(histories.items()):
        plt.plot(range(1, len(hist['val_accuracy'])+1),
                 hist['val_accuracy'],
                 label=f'{tn}Traces_{mode}',
                 marker=markers[i],
                 color=colors[i],
                 markersize=6,
                 markevery=5,  # 每5個點顯示一個標記
                 linewidth=2)
    
    plt.title('Validation Accuracy – 1000 vs 1500 vs 2000', fontsize=20)
    plt.rc('xtick', labelsize=10)   # x 軸刻度字體大小
    plt.rc('ytick', labelsize=10)   # y 軸刻度字體大小
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'val_acc_all.png'), dpi=200)
    plt.close()

    # F1-score
    plt.figure(figsize=(8, 6))
    for i, ((mode, tn), hist) in enumerate(histories.items()):
        plt.plot(range(1, len(hist['val_f1'])+1),
                 hist['val_f1'],
                 label=f'{tn} Traces ({mode})',
                 marker=markers[i],
                 color=colors[i],
                 markersize=6,
                 markevery=5,  # 每5個點顯示一個標記
                 linewidth=2)
    
    plt.title('Validation F1-score – 1000 vs 1500 vs 2000', fontsize=20)
    plt.rc('xtick', labelsize=10)   # x 軸刻度字體大小
    plt.rc('ytick', labelsize=10)   # y 軸刻度字體大小
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('F1-score', fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'val_f1_all.png'), dpi=200)
    plt.close()

    print(f"\nTotal runtime: {(time.time()-start)/60:.2f} minutes")