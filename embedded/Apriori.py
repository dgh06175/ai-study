from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# 1. 거래 데이터 예제
transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread']
]

# 2. 데이터 변환 (One-hot encoding)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 3. 자주 등장하는 항목집합 찾기 (min_support는 최소 지지도)
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("자주 등장하는 항목 집합:\n", frequent_itemsets)

# 4. 연관 규칙 추출 (신뢰도 기준)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
print("\n연관 규칙:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
