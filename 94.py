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

# ユーザーからの問いかけ
messages = [
    {
        "role": "user",
        "content": "What do you call a sweet eaten after dinner?",
    }
]

# モデル専用のチャットテンプレートを適用して、
# user / assistant などの役割情報を含むプロンプトを作成する
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("チャットテンプレート適用後のプロンプト:")
print(prompt)

# チャットテンプレートには特殊トークンが含まれているため、
# add_special_tokens=Falseとして二重に追加されるのを防ぐ
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=False,
).to(model.device)

# プロンプト部分の長さを記録する
input_length = inputs["input_ids"].shape[1]

# 応答を生成する
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )

# 生成結果から、入力したプロンプト部分を除いて応答部分だけを取り出す
response_ids = output_ids[0, input_length:]

response = tokenizer.decode(
    response_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\nモデルの応答:")
print(response)