from glob import glob
import os
from torch.utils.data import Dataset
from datasets import load_dataset
import random
from transformers import BertTokenizerFast
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 定义处理一条数据的逻辑
def tokenize(element):
    # 分词，index
    outputs = tokenizer(element['content'], truncation=True, max_length=context_length, return_overflowing_tokens=True,
                        return_length=True)

    input_batch = []
    for length, input_idx in zip(outputs["length"], outputs["input_ids"]):
        # 如果文本太小直接舍去
        if length == context_length:
            input_batch.append(input_idx)

    return {"input_ids": input_batch}


if __name__ == '__main__':
    random.seed(1001)
    test_rate = 0.15
    context_length = 128

    data_path = os.path.join("gpt2_data_mini", "*", "**")  # *代表所有文件夹，**代表所有文件
    all_files = glob(pathname=data_path)  # 如果不用这个包需要写多个循环

    test_file_list = random.sample(all_files, int(len(all_files) * test_rate))
    train_file_list = [i for i in all_files if i not in test_file_list]

    # 构建dataset，数据集的加载
    raw_datasts = load_dataset("csv", data_files={"train": train_file_list, "valid": test_file_list},
                               cache_dir="cache_data")
    # 常见的是对文件进行分词，填充、裁剪
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    tokenizer.add_special_tokens({"bos_token": "[begin]", "eos_token": "[end]"})

    tokenize(raw_datasts['train'][0])
    # 数据构建完，接下来开始调用模型
    tokenize_datasets = raw_datasts.map(tokenize, batched=True, remove_columns=raw_datasts['train'].column_names)

    config = GPT2Config.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id)

    model = GPT2LMHeadModel(config)
    # model_size = sum([t.numel() for t in model.parameters()])
    # print(f"model_size: {model_size / 1000 / 1000}M")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        learning_rate=1e-5,
        num_train_epochs=10,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        eval_steps=2000,
        logging_steps=2000,
        gradient_accumulation_steps=5,
        weight_decay=0.1,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        save_steps=2000,
        output_dir="model_output",
        fp16=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenize_datasets['train'],
        eval_dataset=tokenize_datasets['valid']
    )
    trainer.train()
    # print("")
