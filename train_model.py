import json
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

with open("squad.json", "r") as f:
    squad_data = json.load(f)

examples = []
for item in squad_data["data"]:
    for para in item["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            if qa["answers"]:  
                examples.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answers": qa["answers"][0] 
                })

dataset = Dataset.from_list(examples)

def preprocess(example):
    tokenized = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offsets = tokenized["offset_mapping"][0]
    start_char = example["answers"]["answer_start"]
    end_char = start_char + len(example["answers"]["text"])

    start_token = end_token = 0
    for i, (start, end) in enumerate(offsets):
        if start <= start_char < end:
            start_token = i
        if start < end_char <= end:
            end_token = i
            break

    tokenized = {k: v.squeeze() for k, v in tokenized.items() if k != "offset_mapping"}
    tokenized["start_positions"] = start_token
    tokenized["end_positions"] = end_token
    return tokenized

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions"]
)

training_args = TrainingArguments(
    output_dir="./qa_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    logging_dir="./logs",
    logging_steps=5,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

trainer.train()

model.save_pretrained("./qa_finetuned")
tokenizer.save_pretrained("./qa_finetuned")
