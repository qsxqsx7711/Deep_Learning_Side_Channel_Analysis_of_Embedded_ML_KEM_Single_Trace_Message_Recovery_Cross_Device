import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix 

trace_num = 2000
samples_num = trace_num * 32
opt = 'o3'
algo = 'PQClean' # orgPoly or PQClean
platform = 'CW308_STM32F3' # CW308_STM32F3 or CW308_STM32F4
oscilloscope = "picoscope" # chipwhisperer or picoscope

def load_trace():
    trace_path = f'traces_train/traces_unfixed_message_{trace_num}_{oscilloscope}_{platform}_{opt}_{algo}'
    
    t = np.load(f'{trace_path}/trace.npy')
    l = np.load(f'{trace_path}/label.npy')

    scaler = StandardScaler()
    t = scaler.fit_transform(t)
    l_encode = to_categorical(l, num_classes=256) # one-hot encoding
    # l_encode = np.zeros((samples_num,256))
    # for i in range(samples_num):
    #     l_encode[i] = to_categorical(l[i], num_classes=256)
    # print(l_encode)
    
    print('size of trace:', t.shape)
    print('size of label:', l.shape)
    print('size of one-hot encoding label:', l_encode.shape)
    print('label:', l)
    return t, l_encode

def split_data(t, l, test_size=0.2, random_state=42):
    """
    將資料以8:2比例分割為訓練集與驗證集。
    test_size=0.2表示20%作為驗證集，80%作為訓練集。
    """
    t_train, t_val, l_train, l_val = train_test_split(
        t, l, test_size=test_size, random_state=random_state, stratify=l.argmax(axis=1)
    )

    print('size of t_train:', t_train.shape)
    print('size of t_val:  ', t_val.shape)
    print('size of l_train:', l_train.shape)
    print('size of l_val:  ', l_val.shape)
    # 檢查標籤分佈
    label_distribution(l_train.argmax(axis=1), 'Training Set')
    label_distribution(l_val.argmax(axis=1), 'Validation Set')
    return t_train, t_val, l_train, l_val

def apply_pca(t_train, t_val, variance_threshold=0.90):
    pca = PCA(n_components=variance_threshold, svd_solver='full', random_state=42)
    t_train_pca = pca.fit_transform(t_train)
    t_val_pca = pca.transform(t_val)
    print(f"PCA 解釋的總變異比例: {np.sum(pca.explained_variance_ratio_):.2f}")
    print(f"主成分數量: {pca.n_components_}")
    return t_train_pca, t_val_pca, pca

def build_model(input_dim=7313, num_classes=256, layer_units=[1024, 512, 256], dropout_rate=0.5):
    """
    建立並回傳MLP模型的函式。
    可以自由調整layer_units以改變各層神經元數量。
    """
    model = Sequential()
    # 第一層需指定input_shape
    model.add(Dense(layer_units[0], input_shape=(input_dim,), activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # 中間層
    for units in layer_units[1:]:
        model.add(Dense(units, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
    # 輸出層
    model.add(Dense(num_classes, activation='softmax'))
    return model
    
def compile_model(model, learning_rate=1e-4):
    """
    編譯模型的函式，可以自由調整優化器或學習率。
    """
    optimizer = RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-07, centered=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, t_train, l_train, t_val, l_val, epochs=1000, batch_size=512):
    """
    訓練模型的函式，加入EarlyStopping、ModelCheckpoint及ReduceLROnPlateau回調函式。
    """
    # 計算類別權重
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(np.argmax(l_train, axis=1)),
        y=np.argmax(l_train, axis=1)
    )
    class_weights = dict(enumerate(class_weights_array))
    print("Class weights:", class_weights)
    
    # 定義 EarlyStopping 和 ModelCheckpoint 回調函式
    early_stopping = EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint('model/best_model_pca_rmsprop.keras', monitor='val_loss', save_best_only=True, verbose=1)
    
    # 定義 ReduceLROnPlateau 回調函式
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    # 訓練模型並傳入回調函式
    history = model.fit(
        t_train, l_train,
        validation_data=(t_val, l_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weights
    )
    # , reduce_lr
    
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
    plt.savefig(f'figure/train_model/label_distribution_{dataset_name}.png')  # 可將圖形儲存成檔案
    plt.close()

def plot_history(history):
    """
    將訓練過程中的loss與accuracy繪出圖表。
    """
    # 取得訓練與驗證數據
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    
    # 繪製 Loss 圖
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 繪製 Accuracy 圖
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('figure/train_model/training_history_pca_rmsprop.png')  # 可將圖形儲存成檔案

def inspect_predictions(model, t_val, l_val, num_samples=10):
    predictions = model.predict(t_val[:num_samples])
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(l_val[:num_samples], axis=1)
    
    for i in range(num_samples):
        print(f"Sample {i+1}: True Label = {y_true[i]}, Predicted Label = {y_pred[i]}")

def check_data_integrity(t, l):
    print("Checking for NaNs in trace data:", np.isnan(t).sum())
    print("Checking for NaNs in labels:", np.isnan(l).sum())
    print("Checking for infinite values in trace data:", np.isinf(t).sum())
    print("Checking for infinite values in labels:", np.isinf(l).sum())

if __name__ == "__main__":
    start = time.time()
    t, l = load_trace()
    check_data_integrity(t, l)
    t_train, t_val, l_train, l_val = split_data(t, l, test_size=0.2)
    t_train_pca, t_val_pca, pca = apply_pca(t_train, t_val, variance_threshold=0.99)
    model = build_model(input_dim=t_train_pca.shape[1], num_classes=256, layer_units=[1024, 512], dropout_rate=0.2)
    model = compile_model(model, learning_rate=1e-3)
    history = train_model(model, t_train_pca, l_train, t_val_pca, l_val, epochs=500, batch_size=256)
    evaluate_model(model, t_val_pca, l_val)
    plot_history(history)
    inspect_predictions(model, t_val_pca, l_val, num_samples=10)
    end = time.time()

    print("The total running time was: ", ((end - start) / 60), " minutes.") 

print('done')
