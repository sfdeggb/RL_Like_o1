from transformers import AutoTokenizer,AutoModelForCausalLM 

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2B-it")
model = AutoModelForCausalLM.from_pretrained("./trained_self_correcting_model").to("cuda")

input_text = "酸度的尺度是什么?"


def change_inference_chat_format(input_text):
    return [
    {"role": "user", "content": f"{input_text}"},
    {"role": "assistant", "content": ""}
    ]
prompt = change_inference_chat_format(input_text)
# tokenizer 
inputs = tokenizer.apply_chat_template(prompt, tokenize=True, 
                                       add_generation_prompt=True, 
                                       return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=128, use_cache=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))