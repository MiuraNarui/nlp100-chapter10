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

# プロンプトをトークン化してGPUへ移動する
inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.device)


# 生成結果を表示する関数
def show_results(title, output_ids):
    print(f"\n=== {title} ===")

    for i, ids in enumerate(output_ids, start=1):
        text = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(f"{i}. {text}")


# 1. Greedy decoding
# 各ステップで最も確率が高いトークンを選ぶ
with torch.inference_mode():
    greedy_outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

show_results("Greedy decoding", greedy_outputs)


# 2. Beam search
# 複数の候補系列を保持しながら、系列全体の確率が高い文章を探索する
with torch.inference_mode():
    beam_outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        num_beams=5,
        num_return_sequences=3,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
    )

show_results("Beam search", beam_outputs)


# 3. Sampling
# 確率分布に従ってトークンを選び、temperatureによる生成結果の変化を観察する
temperatures = [0.5, 1.0, 1.5]

for temperature in temperatures:
    # 各temperatureを同じ乱数条件から比較する
    torch.manual_seed(42)

    with torch.inference_mode():
        sampled_outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            num_return_sequences=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    show_results(
        f"Sampling (temperature={temperature})",
        sampled_outputs,
    )
