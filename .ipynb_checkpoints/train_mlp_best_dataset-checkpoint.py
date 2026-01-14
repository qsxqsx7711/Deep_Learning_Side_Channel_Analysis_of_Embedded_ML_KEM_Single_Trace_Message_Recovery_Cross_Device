#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report
)
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    Callback                    # ← 已移除 ReduceLROnPlateau
)
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

# ---------- 全域參數 ----------
TRACE_NUM    = 2000
OPT          = 'o3'
POI          = 'poi' # poi or segment
ALGO         = 'PQClean'
PLATFORM     = 'CW308_STM32F4'
OSCILLOSCOPE = 'picoscope'
BATCH_SIZE   = 512
EPOCHS       = 150
LR           = 1e-4
LAYER_UNITS  = [2048, 512]
DROPOUT_RATE = 0.5
GROUPS       = ['0', '1', '2', '3', '4']

# ---------- 自訂 Callback：每 epoch 計算 val P / R / F1 ----------
class MetricsLogger(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data   # (X_val, y_val)

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

# ---------- 工具函式 ----------
def load_trace(group):
    base   = f'traces_train/{TRACE_NUM}/'
    folder = f'traces_unfixed_message_{TRACE_NUM}_{OSCILLOSCOPE}_{PLATFORM}_{OPT}_{ALGO}_{POI}_{group}'
    path   = os.path.join(base, folder)

    X_raw  = np.load(os.path.join(path, 'trace.npy'))
    y_raw  = np.load(os.path.join(path, 'label.npy'))

    scaler = StandardScaler().fit(X_raw)
    X      = scaler.transform(X_raw)
    y      = to_categorical(y_raw, num_classes=256)

    return X, y, scaler, path

def split_data(X, y, test_size=0.2):
    return train_test_split(
        X, y, test_size=test_size,
        stratify=np.argmax(y, axis=1),
        random_state=42
    )

def build_model(input_dim, num_classes=256):
    model = Sequential()
    model.add(Dense(LAYER_UNITS[0], input_shape=(input_dim,),
                    use_bias=False, kernel_initializer='he_uniform',
                    kernel_regularizer=regularizers.l2(1e-3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_RATE))

    for units in LAYER_UNITS[1:]:
        model.add(Dense(units, use_bias=False, kernel_initializer='he_uniform',
                        kernel_regularizer=regularizers.l2(1e-3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(num_classes, activation='softmax'))
    return model

def compile_model(model):
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=LR),
        metrics=['accuracy']
    )
    return model

# ---------- 訓練單一 group ----------
def train_one(group):
    print(f"\n=== Training group={group} ===")
    X, y, scaler, trace_path = load_trace(group)

    # 儲存 scaler
    scaler_dir = f'model/{PLATFORM}/{TRACE_NUM}/scalers'
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, f'scaler_group{group}.pkl'))

    Xt, Xv, yt, yv = split_data(X, y)
    model = compile_model(build_model(X.shape[1]))

    # callbacks
    model_dir = f'model/{PLATFORM}/{TRACE_NUM}'
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, f'best_model_group{group}.keras')

    es = EarlyStopping(monitor='val_accuracy', patience=30,
                       min_delta=1e-4, restore_best_weights=True,
                       mode='max', verbose=0)

    metrics_cb = MetricsLogger(validation_data=(Xv, yv))

    mc = ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                         save_best_only=True, mode='max', verbose=0)

    history = model.fit(
        Xt, yt,
        validation_data=(Xv, yv),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[metrics_cb, mc],
        # callbacks=[es, metrics_cb, mc],
        verbose=0
    )
    model.load_weights(ckpt_path)
    
    # === 保存歷史 ===
    hist_dir = f'figure/train_mlp_best_dataset/{PLATFORM}/{TRACE_NUM}/results'
    os.makedirs(hist_dir, exist_ok=True)
    df_hist = pd.DataFrame(history.history)
    df_hist.to_csv(os.path.join(hist_dir, f'history_group{group}.csv'), index_label='epoch')

    best_acc = df_hist['val_accuracy'].max()
    best_f1  = df_hist['val_f1'].max()
    print(f"-> best val_acc: {best_acc:.4f} | best F1: {best_f1:.4f}")

    # optional: 最佳模型下的報告
    y_pred = np.argmax(model.predict(Xv, verbose=0), axis=1)
    y_true = np.argmax(yv, axis=1)
    # print(classification_report(y_true, y_pred, digits=4))

    K.clear_session()
    return best_acc, best_f1, trace_path

# ---------- main ----------
if __name__ == "__main__":

    best_acc_dict, best_f1_dict, trace_paths = {}, {}, {}
    
    for grp in GROUPS:
        start = time.time()
        acc, f1, path = train_one(grp)
        best_acc_dict[grp] = acc
        best_f1_dict [grp] = f1
        trace_paths   [grp] = path
        print(f"\nTotal runtime: {(time.time()-start)/60:.2f} minutes")
        
    # ---- 以 accuracy 為基準選最佳 group ----
    best_group = max(best_acc_dict, key=best_acc_dict.get)
    print(f"\n>>> Best group: {best_group} | val_acc={best_acc_dict[best_group]:.4f} "
          f"| F1={best_f1_dict[best_group]:.4f}")

    # ---- 複製最佳 trace ----
    dst = (f'traces_train/best_traces/'
           f'traces_unfixed_message_{TRACE_NUM}_{OSCILLOSCOPE}_{PLATFORM}_{OPT}_{ALGO}_{POI}')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(trace_paths[best_group], dst, dirs_exist_ok=True)

    # ---- 複製最佳模型 & scaler ----
    best_dir = f'model/{PLATFORM}/{TRACE_NUM}/best_model'
    os.makedirs(best_dir, exist_ok=True)
    shutil.copy2(
        os.path.join(f'model/{PLATFORM}/{TRACE_NUM}', f'best_model_group{best_group}.keras'),
        os.path.join(best_dir, 'best_model.keras')
    )
    shutil.copy2(
        os.path.join(f'model/{PLATFORM}/{TRACE_NUM}/scalers',
                     f'scaler_group{best_group}.pkl'),
        os.path.join(best_dir, 'scaler.pkl')
    )

    # ---------- 繪圖：分開畫 val_accuracy 與 val_F1 ----------
    all_acc, all_f1 = {}, {}
    for grp in GROUPS:
        df = pd.read_csv(
            f'figure/train_mlp_best_dataset/{PLATFORM}/{TRACE_NUM}/results/'
            f'history_group{grp}.csv',
            index_col='epoch'
        )
        all_acc[f'{TRACE_NUM}Traces_POI_Dataset{grp}'] = df['val_accuracy'].tolist()
        all_f1 [f'{TRACE_NUM}Traces_POI_Dataset{grp}'] = df['val_f1'].tolist()

    max_len = max(len(v) for v in all_acc.values())
    for d in (all_acc, all_f1):
        for k, v in d.items():
            d[k] += [np.nan] * (max_len - len(v))

    fig_dir = f'figure/train_mlp_best_dataset/{PLATFORM}/{TRACE_NUM}'
    os.makedirs(fig_dir, exist_ok=True)

    # 定義不同的標記符號和顏色
    markers = ['o', 's', '^', 'D', 'v']  # 圓形, 正方形, 三角形(上), 菱形, 三角形(下)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 不同顏色
    
    # --- 圖 1：Validation Accuracy ---
    plt.figure(figsize=(8, 6))
    for i, (k, v) in enumerate(all_acc.items()):
        plt.plot(range(1, len(v) + 1), v, 
                label=k, 
                marker=markers[i], 
                color=colors[i],
                markersize=6,
                markevery=10,  # 每5個點顯示一個標記，避免過於密集
                linewidth=2)
    
    plt.title(f'Validation Accuracy ({TRACE_NUM} Traces)', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    acc_png = os.path.join(fig_dir, 'val_accuracy_comparison.png')
    plt.savefig(acc_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy plot to {acc_png}")

    # --- 圖 2：Validation F1-score ---
    plt.figure(figsize=(8, 6))
    for i, (k, v) in enumerate(all_f1.items()):
        plt.plot(range(1, len(v) + 1), v, 
                label=k, 
                marker=markers[i], 
                color=colors[i],
                markersize=6,
                markevery=10,  # 每5個點顯示一個標記，避免過於密集
                linewidth=2)
    
    plt.title(f'Validation F1-score ({TRACE_NUM} Traces)', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('F1-score', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    f1_png = os.path.join(fig_dir, 'val_f1_comparison.png')
    plt.savefig(f1_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved F1 plot to {f1_png}")

    # ---- 匯出比較 CSV ----
    pd.DataFrame(all_acc).to_csv(
        f'{fig_dir}/results/val_accuracy_comparison.csv', index_label='epoch')
    pd.DataFrame(all_f1 ).to_csv(
        f'{fig_dir}/results/val_f1_comparison.csv', index_label='epoch')

    