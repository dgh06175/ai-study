import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 아이리스 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target  # 입력 데이터 (꽃의 특징들)

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 최적의 클러스터 개수 찾기 (최대 5까지)
max_clusters = 5
silhouette_scores = []

for i in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)

# 실루엣 점수 시각화
plt.figure(figsize=(8, 6))
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Each Cluster')
plt.show()

# 최적의 클러스터 개수 결정
optimal_n_clusters = np.argmax(silhouette_scores) + 2  # argmax는 0-based index라 +2 보정
print('Optimal number of clusters:', optimal_n_clusters)

# --- 1. 최적의 클러스터 개수로 K-Means 클러스터링 ---
kmeans_optimal = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
kmeans_optimal.fit(X_scaled)
labels_optimal = kmeans_optimal.labels_

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_optimal, cmap='viridis', label='Data Points')
plt.scatter(kmeans_optimal.cluster_centers_[:, 0],
            kmeans_optimal.cluster_centers_[:, 1],
            c='red', marker='x', s=200, label='Cluster Centers (Optimal)')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.title('K-Means Clustering with Optimal k')
plt.legend()
plt.show()

# --- 2. k=4로 K-Means 클러스터링 ---
different_k = 10
kmeans_diff = KMeans(n_clusters=different_k, random_state=42, n_init=10)
kmeans_diff.fit(X_scaled)
labels_diff = kmeans_diff.labels_

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_diff, cmap='viridis', label='Data Points')
plt.scatter(kmeans_diff.cluster_centers_[:, 0],
            kmeans_diff.cluster_centers_[:, 1],
            c='red', marker='x', s=200, label='Cluster Centers (k=10)')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.title('K-Means Clustering with k=10')
plt.legend()
plt.show()
