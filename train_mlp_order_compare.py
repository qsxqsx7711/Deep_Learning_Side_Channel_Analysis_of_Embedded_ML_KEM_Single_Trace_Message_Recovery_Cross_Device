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

# 固定使用截止頻率 (MHz)
CUTOFF_FREQ = 500e3
cutoff_freq_MHz = int(CUTOFF_FREQ / 1e3)
# 比較不同的 filter 階數
FILTER_ORDERS = [1, 5, 10, 15, 20, 25, 30] # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

# 其餘超參數可照原本設定
OPT            = 'o3'
ALGO           = 'PQClean'
FT             = 'ChebyshevI_lowpass' # butterworth ChebyshevI
PLATFORM       = 'CW308_STM32F4'
OSCILLOSCOPE   = 'picoscope'
BATCH_SIZE     = 512
EPOCHS         = 300
LR             = 1e-4
LAYER_UNITS    = [2048, 512]
DROPOUT_RATE   = 0.5

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
def load_trace_by_order(filter_order):
    """
    根據 filter 階數讀取濾波後的 trace.npy 與 label.npy。
    目錄結構：traces_{FT}_train/cf{CUTOFF_FREQ}M_order{filter_order}/
    """
    
    base_folder = f'./traces_{FT}_train'
    folder_name = f'cf{cutoff_freq_MHz}K_order{filter_order}'
    path = os.path.join(base_folder, folder_name)

    # 讀取 npy
    X_raw = np.load(os.path.join(path, 'trace.npy'))
    y_raw = np.load(os.path.join(path, 'label.npy'))

    # 標準化
    scaler = StandardScaler().fit(X_raw)
    X      = scaler.transform(X_raw)
    # one-hot 編碼 (256 類別)
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

    # 如果 LAYER_UNITS 還有其他 layer，就繼續加
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

    # 輸出層
    model.add(Dense(num_classes, activation='softmax'))
    return model

def compile_model(model):
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=LR),
        metrics=['accuracy']
    )
    return model

def train_one_order(filter_order):
    print(f"\n=== Training filter order={filter_order} (cutoff={CUTOFF_FREQ} MHz) ===")
    X, y, scaler = load_trace_by_order(filter_order)
    Xt, Xv, yt, yv = split_data(X, y)

    # Save scaler - 新的路徑結構：PLATFORM/filter_name/cutoff_freq/order{N}/scalers/
    scaler_dir = f'model/{PLATFORM}/{FT}/{cutoff_freq_MHz}K/order{filter_order}/scalers'
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, 'scaler.pkl'))

    # 建立並編譯模型
    model = compile_model(build_model(X.shape[1]))

    # Callbacks：儲存驗證集最佳模型 - 新的路徑結構
    model_dir = f'model/{PLATFORM}/{FT}/{cutoff_freq_MHz}K/order{filter_order}'
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

    # Save history - 新的路徑結構
    hist_dir = f'figure/train_mlp_order_compare/{PLATFORM}/{FT}/{cutoff_freq_MHz}K/order{filter_order}'
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

    # 針對不同 filter 階數進行訓練
    for order in FILTER_ORDERS:
        acc, f1, hist = train_one_order(order)
        best_acc[order]     = acc
        best_f1[order]      = f1
        histories[order]    = hist

    # Export summary CSV - 新的路徑結構：放在該截止頻率的根目錄下
    summary_df = pd.DataFrame({
        'best_val_accuracy': pd.Series(best_acc),
        'best_val_f1'      : pd.Series(best_f1)
    }, index=pd.Index(FILTER_ORDERS, name='filter_order'))
    
    out_summary_path = f'figure/train_mlp_order_compare/{PLATFORM}/{FT}/{cutoff_freq_MHz}K/summary_by_order.csv'
    os.makedirs(os.path.dirname(out_summary_path), exist_ok=True)
    summary_df.to_csv(out_summary_path, float_format='%.4f')

    print("\n=== Summary ===")
    print(summary_df)

    # Plot all curves together - 新的路徑結構
    fig_dir = f'figure/train_mlp_order_compare/{PLATFORM}/{FT}/{cutoff_freq_MHz}K/compare_summary'
    os.makedirs(fig_dir, exist_ok=True)

    # 1) Validation Accuracy - 只修改圖表寬度和圖例位置
    plt.figure(figsize=(8, 6))  # 只增加寬度
    for order, hist in histories.items():
        plt.plot(
            range(1, len(hist['val_accuracy'])+1),
            hist['val_accuracy'],
            label=f'Order {order}'
        )
    plt.title(f'Validation Accuracy – {FT} (Cutoff {cutoff_freq_MHz} KHz)', fontsize=20)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.grid(True)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'val_acc_order_all.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2) Validation F1-score - 同樣的修改
    plt.figure(figsize=(8, 6))  # 只增加寬度
    for order, hist in histories.items():
        plt.plot(
            range(1, len(hist['val_f1'])+1),
            hist['val_f1'],
            label=f'Order {order}'
        )
    plt.title(f'Validation F1-score – {FT} (Cutoff {cutoff_freq_MHz} KHz)', fontsize=20)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('F1-score', fontsize=20)
    plt.grid(True)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'val_f1_order_all.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\nTotal runtime: {(time.time()-start)/60:.2f} minutes")