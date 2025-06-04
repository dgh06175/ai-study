import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

alphabet = list("abcdefghijklmnopqrstuvwxyz ")
char2idx = {ch: i for i, ch in enumerate(alphabet)}
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(alphabet)

def encode_string(s, max_len=10):
    s = s.lower()[:max_len].ljust(max_len)
    one_hot = torch.zeros(max_len, vocab_size)
    for i, ch in enumerate(s):
        if ch in char2idx:
            one_hot[i, char2idx[ch]] = 1
    return one_hot

def decode_tensor(tensor):
    indices = torch.argmax(tensor, dim=1)
    return ''.join([idx2char[i.item()] for i in indices])

class CharAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

input_size = vocab_size
hidden_size = 16
model = CharAutoEncoder(input_size, hidden_size)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

text = "hello"
x_train = encode_string(text)

for epoch in range(500):
    optimizer.zero_grad()
    output, _ = model(x_train)
    loss = loss_fn(output, x_train)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    decoded, encoded = model(x_train)
    print(f"원본 입력: {text}")
    print(f"복원 결과: {decode_tensor(decoded)}")

    # 임의의 노이즈 추가
    noise = torch.randn_like(encoded) * 0.2
    noisy_encoded = encoded + noise
    noisy_decoded = model.decoder(noisy_encoded)
    print(f"노이즈 추가 후 복원 결과: {decode_tensor(noisy_decoded)}")

    # 완전 무작위 벡터
    random_encoded = torch.randn_like(encoded)
    random_decoded = model.decoder(random_encoded)
    print(f"무작위 벡터 디코딩 결과: {decode_tensor(random_decoded)}")

