import time
import numpy as np

# 데이터 크기
N = 100_000_000

# 배열 순회 시간 측정
array = np.arange(N)  # numpy 배열 생성 (연속된 메모리)
start_time = time.time()
sum_array = np.sum(array)  # 배열을 순회하면서 합 계산
array_time = time.time() - start_time

# Map (딕셔너리) 접근 시간 측정
map_data = {i: i for i in range(N)}  # Python 딕셔너리 생성 (해시 테이블)
start_time = time.time()
sum_map = sum(map_data.values())  # 딕셔너리 값들을 순회하면서 합 계산
map_time = time.time() - start_time

# 결과 출력
print(array_time)
print(map_time)
