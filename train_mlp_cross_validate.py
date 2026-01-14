import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.saving import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K
import os
import joblib
import seaborn as sns  # 新增的引入

trace_num = 2000
samples_num = trace_num * 32

def load_trace():
    trace_path = f'traces_train/traces_unfixed_message_{trace_num}_chipwhisperer'
    
    t = np.load(f'{trace_path}/trace.npy')
    l = np.load(f'{trace_path}/label.npy')

    scaler = StandardScaler()
    t = scaler.fit_transform(t)
    l_encode = to_categorical(l, num_classes=256) # one-hot encoding

    # 保存 scaler
    os.makedirs('scaler', exist_ok=True)  # 確保資料夾存在
    joblib.dump(scaler, 'scaler/scaler.joblib')
    
    print('size of trace:', t.shape)
    print('size of label:', l.shape)
    print('size of one-hot encoding label:', l_encode.shape)
    print('label:', l)
    print('label one-hot encoding', l_encode)
    
    return t, l, l_encode

def build_model(input_dim=6940, num_classes=256, layer_units=[1024, 512, 256], dropout_rate=0.5):
    """
    建立並回傳MLP模型的函式。
    可以自由調整layer_units以改變各層神經元數量。
    """
    model = Sequential()
    # 第一層需指定input_shape
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # 中間層
    for units in layer_units[1:]:
        model.add(Dense(units, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-3))) # , kernel_regularizer=regularizers.l2(1e-3)
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
    # 輸出層
    model.add(Dense(num_classes, activation='softmax'))
    return model

def compile_model(model, learning_rate=1e-3):
    """
    編譯模型的函式，可以自由調整優化器或學習率。
    """
    optimizer=RMSprop(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, t_train, l_train, t_val, l_val, epochs=50, batch_size=256):
    """
    訓練模型的函式，加入EarlyStopping與ModelCheckpoint。
    """
    # 定義 EarlyStopping 和 ModelCheckpoint 回調函式
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1, mode='max')
    model_checkpoint = ModelCheckpoint('model/best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
    
    # 訓練模型並傳入回調函式
    history = model.fit(
        t_train, l_train,
        validation_data=(t_val, l_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    return history

def evaluate_model(model, t_val, l_val):
    """
    評估模型的函式。
    """
    val_loss, val_acc = model.evaluate(t_val, l_val, verbose=0)
    print("Validation Loss:", val_loss)
    print("Validation Accuracy:", val_acc)

def label_distribution(labels, dataset_name):
    counts = np.bincount(labels, minlength=256)
    print(f"{dataset_name} label distribution:", counts)
    plt.figure(figsize=(12, 6))
    plt.rc('xtick', labelsize=20)   # x 軸刻度字體大小
    plt.rc('ytick', labelsize=20)   # y 軸刻度字體大小
    plt.bar(range(256), counts, width=1, edgecolor='black', align='center')
    plt.title(f"Count of Label in {dataset_name}")
    plt.xlabel("Value", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.xlim(0, 255)
    plt.savefig(f'figure/train_model/label_distribution_{dataset_name}.png', dpi=200, bbox_inches='tight')  # 可將圖形儲存成檔案
    plt.close()

def plot_history(history, fold):
    """
    將訓練過程中的loss與accuracy繪出圖表，並根據折疊編號儲存。
    """
    # 取得訓練與驗證數據
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    
    # 繪製 Loss 圖
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title(f'Fold {fold} - Training and Validation Loss', fontsize=20)
    plt.rc('xtick', labelsize=20)   # x 軸刻度字體大小
    plt.rc('ytick', labelsize=20)   # y 軸刻度字體大小
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend()

    # 繪製 Accuracy 圖
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title(f'Fold {fold} - Training and Validation Accuracy', fontsize=20)
    plt.rc('xtick', labelsize=20)   # x 軸刻度字體大小
    plt.rc('ytick', labelsize=20)   # y 軸刻度字體大小
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'figure/cross_validation/training_history_fold_{fold}.png', dpi=200, bbox_inches='tight')  # 儲存圖形
    plt.close()

def generate_confusion_matrix(y_true, y_pred, fold, normalize=False):
    """
    生成並儲存混淆矩陣圖表。
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(256))
    
    plt.figure(figsize=(20, 16))
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues')
        title = f'Fold {fold} - Normalized Confusion Matrix'
        filename = f'figure/cross_validation/confusion_matrix_fold_{fold}_normalized.png'
    else:
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        title = f'Fold {fold} - Confusion Matrix'
        filename = f'figure/cross_validation/confusion_matrix_fold_{fold}.png'
    
    plt.title(title, fontsize=20)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()

def generate_classification_report_and_confusion_matrix(model, t_val, l_val_labels, fold):
    """
    生成並儲存分類報告和混淆矩陣。
    """
    predictions = model.predict(t_val)
    y_pred = np.argmax(predictions, axis=1)
    y_true = l_val_labels
    report = classification_report(y_true, y_pred, digits=4)
    print(f"Fold {fold} - Classification Report:\n", report)
    
    # 儲存分類報告到檔案
    with open(f'figure/cross_validation/classification_report_fold_{fold}.txt', 'w') as f:
        f.write(f"Fold {fold} - Classification Report:\n")
        f.write(report)
    
    # 生成並儲存混淆矩陣（未歸一化）
    generate_confusion_matrix(y_true, y_pred, fold, normalize=False)
    
    # 生成並儲存混淆矩陣（歸一化）
    generate_confusion_matrix(y_true, y_pred, fold, normalize=True)

def cross_validate_model(t, l, l_encode, n_splits=5, epochs=100, batch_size=512):
    """
    使用 StratifiedKFold 進行交叉驗證，並在每個折疊中訓練和評估模型。
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    val_accuracies = []
    val_losses = []
    
    # 創建存放模型和歷史的資料夾
    os.makedirs('model/cross_validation', exist_ok=True)
    os.makedirs('figure/cross_validation', exist_ok=True)
    
    for train_index, val_index in skf.split(t, l):
        print(f'\n--- Fold {fold} ---')
        t_train, t_val = t[train_index], t[val_index]
        l_train, l_val = l_encode[train_index], l_encode[val_index]
        l_val_labels = l[val_index]  # 用於 classification_report
        
        K.clear_session()
        # 建立並編譯模型
        model = build_model(input_dim=t.shape[1], num_classes=256, layer_units=[512], dropout_rate=0.3) # 2048 512
        model = compile_model(model, learning_rate=1e-5)
        
        # 訓練模型
        history = train_model(model, t_train, l_train, t_val, l_val, epochs=epochs, batch_size=batch_size)
        
        # 評估模型
        val_loss, val_acc = model.evaluate(t_val, l_val, verbose=0)
        print(f"Fold {fold} - Validation Loss: {val_loss}")
        print(f"Fold {fold} - Validation Accuracy: {val_acc}")
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        
        # 繪製並儲存訓練歷史
        plot_history(history, fold)
        
        # 儲存最佳模型
        model.save(f'model/cross_validation/best_model_fold_{fold}.keras')
        
        # 生成並儲存分類報告和混淆矩陣
        generate_classification_report_and_confusion_matrix(model, t_val, l_val_labels, fold)
        
        fold += 1
    
    print(f"\nCross-Validation Results:")
    print(f"Average Validation Accuracy: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
    print(f"Average Validation Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")

if __name__ == "__main__":
    start = time.time()
    
    # 載入資料
    t, l, l_encode = load_trace()
    
    # 執行交叉驗證
    cross_validate_model(t, l, l_encode, n_splits=5, epochs=300, batch_size=128)
    
    end = time.time()
    
    print("The total running time was: ", ((end - start) / 60), " minutes.") 

print('done')
