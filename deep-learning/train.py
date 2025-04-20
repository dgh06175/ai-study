import numpy as np

# 입력/정답 정의
x_in = np.array([[0,0], [1,0], [0,1], [1,1]])
t_in = np.array([-1, 1, 1, 1])
W = np.array([-0.5, 0.75])
b = 0.375
alpha = 0.4

def perceptron_1_(W, X, b):
    y = np.dot(W, X) + b
    return 1 if y >= 0 else -1

def perceptron_predict(W, Xset, b):
    return [perceptron_1_(W, x, b) for x in Xset]

epoch = 0

while True:  # 충분한 반복
    epoch += 1
    pred = perceptron_predict(W, x_in, b)
    Yset = pred != t_in
    print(f"\n[Epoch {epoch + 1}]")
    print("예측값:", pred)
    print("정답과 다른 인덱스:", np.where(Yset)[0])

    if not np.any(Yset):
        print("→ 모든 예측이 맞았습니다. 학습 종료.")
        print(f"W: {W}")
        print(f"b: {b}")
        break

    # 업데이트할 값 계산
    sum_w = sum(ti * xi for flag, ti, xi in zip(Yset, t_in, x_in) if flag)
    sum_b = sum(ti for flag, ti in zip(Yset, t_in) if flag)

    print("W 업데이트량:", alpha * sum_w)
    print("b 업데이트량:", alpha * sum_b)

    # 파라미터 업데이트
    W += alpha * sum_w
    b += alpha * sum_b

    print("업데이트된 W:", W)
    print("업데이트된 b:", b)
