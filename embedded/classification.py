import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

class_0 = np.array([[1, 1], [2, 2], [3, 3], [4, 3], [5, 2]])
class_1 = np.array([[9, 6], [8, 7] ,[10, 8], [7, 9], [9, 7]])

X_test = np.array([[3, 6], [6, 3], [8, 8]])

X = np.vstack((class_0, class_1))
y = np.hstack((np.zeros(len(class_0)), np.ones(len(class_1))))

k = 3
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X, y)


y_train_pred = knn_model.predict(X)
y_test_pred = knn_model.predict(X_test)

# 예측 결과 출력
print("학습 데이터 예측 결과:")
for i, (point, pred) in enumerate(zip(X, y_train_pred)):
    print(f"Point {i+1}: {point}, Predicted Class: {int(pred)}")

print("\n테스트 데이터 예측 결과:")
for i, (point, pred) in enumerate(zip(X_test, y_test_pred)):
    print(f"Test Point {i+1}: {point}, Predicted Class: {int(pred)}")


plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Class 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Class 1')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

plt.scatter(X_test[:, 0], X_test[:, 1], color='green', marker='x', label='Test data')

plt.title('KNN 분류 결과 및 분류 경계')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.legend()

plt.show()
