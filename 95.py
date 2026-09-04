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

# 問題94でのユーザーの質問
first_user_message = "What do you call a sweet eaten after dinner?"

# 問題94で生成された応答
first_assistant_message = (
    "A sweet often eaten after dinner is typically referred to as dessert."
)

# 追加の問いかけ
second_user_message = (
    "Please give me the plural form of the word "
    "with its spelling in reverse order."
)

# これまでの会話履歴を含めてmessagesを作成する
messages = [
    {
        "role": "user",
        "content": first_user_message,
    },
    {
        "role": "assistant",
        "content": first_assistant_message,
    },
    {
        "role": "user",
        "content": second_user_message,
    },
]

# 会話履歴全体にチャットテンプレートを適用する
# add_generation_prompt=Trueにより、次がassistantの発言であることを示す
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("言語モデルに与えるプロンプト:")
print(prompt)

# チャットテンプレート内に特殊トークンが含まれているため、
# add_special_tokens=Falseとして二重追加を防ぐ
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=False,
).to(model.device)

# 入力したプロンプト部分の長さを記録する
input_length = inputs["input_ids"].shape[1]

# 2回目のユーザーの問いかけに対する応答を生成する
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )

# 入力プロンプト部分を除き、新しく生成された応答だけを取り出す
response_ids = output_ids[0, input_length:]

response = tokenizer.decode(
    response_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\nモデルの応答:")
print(response)