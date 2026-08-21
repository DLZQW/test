import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers

FEATURE_COLS = [
    '1週報酬率',
    '4週報酬率',
    '8週報酬率',
    '5日均線偏離率',
    '20日均線偏離率',
    '成交量增幅',
    '外資淨買賣超/成交量',
    '融資融券比率'
]
SEQUENCE_LENGTH = 5
TARGET_COL = '未來4週報酬%'


def build_sequences(df):
    records = []
    df = df.sort_values(['股票代號', '進場日期']).copy()

    for stock_id, group in df.groupby('股票代號', sort=False):
        group = group.sort_values('進場日期').reset_index(drop=True)
        if len(group) < SEQUENCE_LENGTH + 1:
            continue

        for i in range(SEQUENCE_LENGTH, len(group)):
            window = group.iloc[i - SEQUENCE_LENGTH:i][FEATURE_COLS].to_numpy(dtype=np.float32)
            target = float(group.iloc[i][TARGET_COL])
            end_date = pd.to_datetime(group.iloc[i]['進場日期'])
            records.append({
                'end_date': end_date,
                'window': window,
                'target': target
            })

    if not records:
        return np.empty((0, SEQUENCE_LENGTH, len(FEATURE_COLS)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    records_df = pd.DataFrame(records).sort_values('end_date').reset_index(drop=True)
    X = np.stack(records_df['window'].to_numpy()).astype(np.float32)
    y = records_df['target'].to_numpy().astype(np.float32)
    return X, y


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_training_data.csv")

    if not os.path.exists(data_path):
        print(f"❌ 找不到訓練資料 {data_path}，請先執行回測程式。")
        return

    print("讀取訓練資料...")
    df = pd.read_csv(data_path)
    target_column = TARGET_COL if TARGET_COL in df.columns else '是否獲利'
    df = df.dropna(subset=FEATURE_COLS + [target_column]).copy()

    if len(df) < 50:
        print("⚠️ 警告：有效樣本數少於 50 筆，模型可能無法有效學習。")

    if target_column == '是否獲利':
        df[TARGET_COL] = df['是否獲利'].astype(float)

    X, y = build_sequences(df)
    if len(X) == 0:
        print("⚠️ 依照序列設定，沒有足夠的訓練樣本。")
        return

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"⏳ 正在使用 LSTM 進行『未來 4 週報酬率』回歸預測，序列長度={SEQUENCE_LENGTH}, 樣本數={len(X)}...")
    model = keras.Sequential([
        keras.Input(shape=(SEQUENCE_LENGTH, len(FEATURE_COLS))),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=60,
        batch_size=16,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )

    y_pred = model.predict(X_test, verbose=0).reshape(-1)
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

    print("\n📊 模型驗證報告:")
    print(f"平均絕對誤差 (MAE): {mae:.3f}%")
    print(f"均方根誤差 (RMSE): {rmse:.3f}%")
    print(f"實際平均 4 週報酬: {np.mean(y_test):+.3f}%")
    print(f"預測平均 4 週報酬: {np.mean(y_pred):+.3f}%")
    print("-" * 30)

    model_filename = 'lstm_trading_model.keras'
    model.save(model_filename)
    print(f"\n✅ 模型已成功儲存為 {model_filename}")


if __name__ == "__main__":
    main()