from transformers import GPT2LMHeadModel, BertTokenizerFast
import os

if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    model_path = os.path.join("model_output", "checkpoint-24000")

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to("cuda")

    while True:
        input_text = input("请输如：")
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        input_ids = input_ids.to("cuda")

        output = model.generate(input_ids, max_length=20, num_beams=5, repetition_penalty=1, early_stopping=True)

        output_text = tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"{output_text}")
