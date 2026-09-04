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

# 生成する最大トークン数
max_new_tokens = 10

# プロンプトをトークン化してGPUへ移動する
inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.device)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 生成されたトークンと、その生成確率を保存する
generated_tokens = []
generated_probs = []

# 1トークンずつ生成し、そのとき選ばれたトークンの確率を記録する
with torch.inference_mode():
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # 現在の文脈に対する次トークンの予測スコア
        next_token_logits = outputs.logits[:, -1, :]

        # スコアを確率に変換
        next_token_probs = torch.softmax(
            next_token_logits.float(),
            dim=-1,
        )

        # Greedy decoding:
        # 最も確率が高いトークンを次のトークンとして選ぶ
        next_token_id = torch.argmax(
            next_token_probs,
            dim=-1,
            keepdim=True,
        )

        # 選ばれたトークンの確率を取得する
        next_token_prob = next_token_probs.gather(
            dim=-1,
            index=next_token_id,
        ).item()

        token_id = next_token_id.item()
        token = tokenizer.convert_ids_to_tokens(token_id)
        text = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )

        generated_tokens.append(
            {
                "token": token,
                "text": text,
                "probability": next_token_prob,
            }
        )

        # 生成したトークンを入力の末尾に追加する
        input_ids = torch.cat(
            [input_ids, next_token_id],
            dim=-1,
        )

        # attention_maskにも新しいトークン分の1を追加する
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.size(0), 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=-1,
        )

        # EOSトークンが生成されたら終了する
        if token_id == tokenizer.eos_token_id:
            break

# 生成された全文を表示する
generated_text = tokenizer.decode(
    input_ids[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("プロンプト:")
print(prompt)

print("\n生成されたテキスト:")
print(generated_text)

print("\n生成された各トークンの尤度:")
for i, item in enumerate(generated_tokens, start=1):
    print(
        f"{i:2d}. "
        f"token={item['token']!r:<18} "
        f"text={item['text']!r:<15} "
        f"probability={item['probability']:.6f}"
    )
