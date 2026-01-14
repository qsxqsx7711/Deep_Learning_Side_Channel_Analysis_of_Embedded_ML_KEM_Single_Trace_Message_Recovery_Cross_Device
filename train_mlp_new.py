import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report

# === 參數設定 ===
TRACE_NUM    = 1000
ORDER        = '1'
OPT          = 'o3'                   # o0 or o3
ALGO         = 'PQClean'              # or 'orgPoly'
PLATFORM     = 'CW308_STM32F4'        # or other
OSCILLOSCOPE = 'picoscope'        # or 'picoscope' or 'chipwhisperer'
BATCH_SIZE   = 512
EPOCHS       = 500
LR           = 5e-4
LAYER_UNITS  = [1024, 512]
DROPOUT_RATE = 0.3

def load_trace():
    path = f'traces_train/{TRACE_NUM}/traces_unfixed_message_{TRACE_NUM}_{OSCILLOSCOPE}_{PLATFORM}_{OPT}_{ALGO}_{ORDER}'
    X = np.load(f'{path}/trace.npy')
    y = np.load(f'{path}/label.npy')
    X = StandardScaler().fit_transform(X)
    y = to_categorical(y, num_classes=256)
    print(f"[Load] X: {X.shape}, y: {y.shape}")
    return X, y

def split_data(X, y, test_size=0.2):
    Xt, Xv, yt, yv = train_test_split(
        X, y, test_size=test_size,
        stratify=np.argmax(y, axis=1), random_state=42
    )
    print(f"[Split] Train: {Xt.shape}, {yt.shape} | Val: {Xv.shape}, {yv.shape}")
    return Xt, Xv, yt, yv

def build_model(input_dim, num_classes=256, layer_units=LAYER_UNITS, dropout_rate=DROPOUT_RATE):
    model = Sequential()
    # 第一層
    model.add(Dense(layer_units[0], input_shape=(input_dim,), use_bias=False, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    # 中間層
    for units in layer_units[1:]:
        model.add(Dense(units, use_bias=False, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-3)))# , kernel_regularizer=regularizers.l2(1e-3)
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))
    # 輸出層
    model.add(Dense(num_classes, activation='softmax'))
    return model

def compile_model(model, learning_rate=LR):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, Xt, yt, Xv, yv, epochs=EPOCHS, batch_size=BATCH_SIZE):
    es = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max', verbose=1)
    mc = ModelCheckpoint('model/best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    return model.fit(
        Xt, yt,
        validation_data=(Xv, yv),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, mc],
        verbose=1
    )

def evaluate_model(model, Xv, yv):
    loss, acc = model.evaluate(Xv, yv, verbose=0)
    print(f"[Eval] Val loss: {loss:.4f}, Val acc: {acc:.4f}")
    preds = model.predict(Xv, batch_size=BATCH_SIZE, verbose=0)
    y_true = np.argmax(yv, axis=1)
    y_pred = np.argmax(preds, axis=1)
    print("\n" + classification_report(y_true, y_pred, digits=4))

def plot_history(history):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    start = time.time()
    X, y       = load_trace()
    Xt, Xv, yt, yv = split_data(X, y)
    model = build_model(input_dim=X.shape[1])
    model = compile_model(model)
    history = train_model(model, Xt, yt, Xv, yv)
    # evaluate_model(model, Xv, yv)
    plot_history(history)
    print(f"Total runtime: {(time.time() - start)/60:.2f} minutes")
