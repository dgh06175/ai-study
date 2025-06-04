from transformers import pipeline

# 텍스트 생성 파이프라인 로딩(기본 모델은 distilgpt2)
generator = pipeline("text-generation", model="distilgpt2")

# 프롬프트 입력
prompt = "Once upon a time"

# 텍스트 생성
result = generator(prompt, max_length=50, num_return_sequences=1)

# 출력
print(result[0]["generated_text"])
