import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 데이터 로딩
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# LDA 적용
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# 시각화 준비
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 1. 원본 데이터 (Feature 1 vs Feature 2)
for label, color in zip(range(3), ['red', 'green', 'blue']):
    ax1.scatter(X[y == label, 0], X[y == label, 1], label=target_names[label], color=color)
ax1.set_title('Original Data (Feature 1 vs 2)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend()

# 2. LDA 변환 결과
for label, color in zip(range(3), ['red', 'green', 'blue']):
    ax2.scatter(X_lda[y == label, 0], X_lda[y == label, 1], label=target_names[label], color=color)
ax2.set_title('LDA Result (2D)')
ax2.set_xlabel('LDA1')
ax2.set_ylabel('LDA2')
ax2.legend()

plt.tight_layout()
plt.show()
