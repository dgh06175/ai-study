import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers

# 1. 데이터 생성 (노이즈 증가)
np.random.seed(0)
X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = 2 * X + 3 + 1.0 * np.random.randn(100, 1)  # 더 많은 노이즈로 일반화 어려움 증가

# 훈련 데이터 축소 (과적합 유도)
X_train = X[:20]
y_train = y[:20]
X_test = X
y_test = y

# 2. 모델 생성 함수 (복잡한 모델로 오버피팅 유도)
def build_model(dropout = 0.0):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(1,)),
        layers.Dropout(dropout),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 3. 모델 1: 단일 학습 모델 (오버피팅 발생)
model_single = build_model()
model_single.fit(X_train, y_train, epochs=500, verbose=0)
pred_single = model_single.predict(X_test)
mse_single = mean_squared_error(y_test, pred_single)

# 4. 모델 2: K-Fold 교차검증 후 평균 예측
kf = KFold(n_splits=5, shuffle=True, random_state=1)
preds_cv = np.zeros_like(y_test)

for train_idx, val_idx in kf.split(X_train):
    X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
    model_cv = build_model()
    model_cv.fit(X_fold_train, y_fold_train, epochs=500, verbose=0)
    preds_cv += model_cv.predict(X_test)

preds_cv /= 5
mse_cv = mean_squared_error(y_test, preds_cv)


# 5. 모델 3:
kf_2 = KFold(n_splits=5, shuffle=True, random_state=1)
preds_cv_2 = np.zeros_like(y_test)

for train_idx, val_idx in kf_2.split(X_train):
    X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
    model_cv = build_model(0.3)
    model_cv.fit(X_fold_train, y_fold_train, epochs=500, verbose=0)
    preds_cv_2 += model_cv.predict(X_test)

preds_cv_2 /= 5
mse_dropout = mean_squared_error(y_test, preds_cv_2)



# 5. 결과 출력
print("MSE (Single Model):", mse_single)
print("MSE (K-Fold CV Averaged):", mse_cv)
print("MSE (K-Fold + Dropout): ", mse_dropout)