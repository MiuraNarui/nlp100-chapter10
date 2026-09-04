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

# プロンプト
prompt = "The movie was full of"

# プロンプトをトークンID列に変換する
inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

# 実際にモデルへ入力されるトークン列を確認する
input_ids = inputs["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print("プロンプト:")
print(prompt)

print("\nモデルに入力されるトークン列:")
print(tokens)

print("\nトークンID列:")
print(input_ids.tolist())

# 入力をモデルと同じデバイスへ移動する
inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}

# 次トークンの予測を行う
with torch.inference_mode():
    outputs = model(**inputs)

# 最後の入力トークン位置における、語彙全体の予測スコアを取り出す
next_token_logits = outputs.logits[0, -1, :]

# 数値計算を安定させるためfloat32に変換してから確率へ変換する
next_token_probs = torch.softmax(next_token_logits.float(), dim=-1)

# 確率が高い上位10トークンを取得する
top_probs, top_ids = torch.topk(next_token_probs, k=10)

print("\n次に続くトークンの上位10個:")
for rank, (token_id, prob) in enumerate(zip(top_ids, top_probs), start=1):
    token_id = token_id.item()

    # トークナイザ内部でのトークン表現
    token = tokenizer.convert_ids_to_tokens(token_id)

    # 人が読みやすい文字列へ戻した表現
    text = tokenizer.decode(
        [token_id],
        clean_up_tokenization_spaces=False,
    )

    print(
        f"{rank:2d}. "
        f"token={token!r:<18} "
        f"text={text!r:<15} "
        f"probability={prob.item():.6f}"
    )