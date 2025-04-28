import numpy as np
import matplotlib.pyplot as plt

# 각 x에 대한 정답 t
x_in = np.array([[0,0], [1,0], [0,1], [1,1]])
t_in = np.array([-1, 1, 1, 1])

# W, b 초기값
W = np.array([-0.5, 0.75])
b = 0.375
alpha = 0.2

# 판별 함수 d
def d(W, X, b):
    sum_ = np.dot(W, X) + b
    if sum_ >= 0.0:
        return 1
    else:
        return -1

# 손실함수 j
def j(W, Xset, b):
    pred_set = []
    for index in range(len(Xset)):
        pred = d(W, Xset[index], b)
        pred_set.append(pred)
    return pred_set

cnt = 0
Yset = None

while True:

    pred = j(W, x_in, b)
    Yset = (pred != t_in)

    # True 인게 있으면 즉, 정답이 아닌 Yset 이 있으면
    cnt += 1
    print(cnt)
    if np.count_nonzero(Yset):
        # 가중치 조절
        sum_w = 0
        sum_b = 0
        for flag, ti, xi in zip(Yset, t_in, x_in):
            if flag: # 틀린 데이터에 대해서
                sum_w += ti * xi # tiXi 곱한 값을 다 더한값을 W에 더함
                sum_b += ti # ti ㄷㄷㅓㅎㅏㄴ
        W += alpha * sum_w #
        b += alpha * sum_b
    else:
        break

print(W)
print(b)

plt.figure(figsize=(8, 6))

# 클래스별로 색상 다르게 그리기
for i, (x, t, y) in enumerate(zip(x_in, t_in, Yset)):
    color = 'blue' if t == 1 else 'red'
    marker = 'o' if t == y else 'x'  # 정답: 동그라미, 오답: X표
    label = f"Class {t}" if i == t_in.tolist().index(t) else ""
    plt.scatter(x[0], x[1], color=color, marker=marker, s=100, label=label)
    plt.text(x[0] + 0.02, x[1] + 0.02, f"{i}", fontsize=9)

# 결정 경계 그리기: W[0]*x + W[1]*y + b = 0 → y = -(W[0]*x + b) / W[1]
x_vals = np.linspace(-0.2, 1.2, 100)
y_vals = -(W[0] * x_vals + b) / W[1]
plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

# 꾸미기
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptron 결과 시각화')
plt.legend()
plt.grid(True)
plt.show()