from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 생성
X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)

# 2. 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 3. SVM 분류 모델 생성 및 학습
model = SVC(kernel='rbf', gamma='scale')
model.fit(X_train, y_train)

# 4. 테스트 데이터 예측
y_pred = model.predict(X_test)

# 5. 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"테스트 정확도: {accuracy:.4f}")

# 6. 결정 경계 시각화 함수
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('SVM Decision Boundary on make_moons')
    plt.show()

# 7. 결정 경계 시각화
plot_decision_boundary(X_test, y_test, model)
