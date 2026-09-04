import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# GPT型の事前学習済み言語モデルを読み込む
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# パープレキシティを比較する文
sentences = [
    "The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises",
]


# 1文のパープレキシティを計算する関数
def calculate_perplexity(sentence):
    # 文をトークン化してGPUへ移動する
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
    ).to(model.device)

    # Causal Language Modelでは、labelsにinput_idsを渡すと
    # 各位置で「次のトークン」を正解としてCross Entropy Lossを計算する
    with torch.inference_mode():
        outputs = model(
            **inputs,
            labels=inputs["input_ids"],
        )

    # outputs.lossは各次トークン予測の平均Cross Entropy Loss
    loss = outputs.loss.float()

    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()

    return loss.item(), perplexity


print("各文のパープレキシティ\n")

for sentence in sentences:
    loss, perplexity = calculate_perplexity(sentence)

    print(sentence)
    print(f"  loss       : {loss:.6f}")
    print(f"  perplexity : {perplexity:.6f}")
    print()