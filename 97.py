import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel


# 事前学習済み言語モデルを読み込む
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

encoder = AutoModel.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)
encoder.eval()

# 97番では言語モデル自体は更新せず、文章のベクトル化に利用する
for param in encoder.parameters():
    param.requires_grad = False


# SST-2をダウンロードして展開する
data_dir = Path("data")
train_path = data_dir / "SST-2" / "train.tsv"
dev_path = data_dir / "SST-2" / "dev.tsv"
data_dir.mkdir(exist_ok=True)

if not train_path.exists() or not dev_path.exists():
    zip_path = data_dir / "SST-2.zip"

    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
        zip_path,
    )

    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(data_dir)

train = pd.read_csv(train_path, sep="\t")
dev = pd.read_csv(dev_path, sep="\t")


# Llamaの最終層の出力を平均し、1文を1本のベクトルに変換する
def encode_sentences(sentences, batch_size=64):
    vectors = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(encoder.device)

        with torch.inference_mode():
            outputs = encoder(**inputs)

        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)

        # PAD部分を除いて平均を取る
        sentence_vectors = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1)
        )

        vectors.append(sentence_vectors.float().cpu())

        processed = min(start + batch_size, len(sentences))
        print(f"エンコード済み: {processed}/{len(sentences)}")

    return torch.cat(vectors)


print("訓練データをエンコードします")
train_x = encode_sentences(train["sentence"].tolist())

print("\n開発データをエンコードします")
dev_x = encode_sentences(dev["sentence"].tolist())

train_y = torch.tensor(train["label"].values, dtype=torch.long)
dev_y = torch.tensor(dev["label"].values, dtype=torch.long)


# 文ベクトルからpositive / negativeを予測するフィードフォワード層
classifier = nn.Sequential(
    nn.Linear(train_x.shape[1], 256),
    nn.ReLU(),
    nn.Linear(256, 2),
).to(encoder.device)

train_loader = DataLoader(
    TensorDataset(train_x, train_y),
    batch_size=256,
    shuffle=True,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)


# フィードフォワード層を学習する
epochs = 5

for epoch in range(epochs):
    classifier.train()
    total_loss = 0

    for x, y in train_loader:
        x = x.to(encoder.device)
        y = y.to(encoder.device)

        optimizer.zero_grad()

        logits = classifier(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"loss={total_loss / len(train_loader):.4f}"
    )


# 開発データで正解率を測定する
classifier.eval()

with torch.inference_mode():
    logits = classifier(dev_x.to(encoder.device))
    predictions = logits.argmax(dim=1).cpu()

accuracy = (predictions == dev_y).float().mean().item()

print("\n=== 評価結果 ===")
print(f"正解数: {(predictions == dev_y).sum().item()}/{len(dev_y)}")
print(f"正解率: {accuracy:.4f} ({accuracy * 100:.2f}%)")