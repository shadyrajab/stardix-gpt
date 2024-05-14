import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")


def format_examples(example):
    inputs = tokenizer(
        example["Perguntas"],
        example["Respostas"],
        truncation=True, 
        padding="max_length",
        max_length=512, 
    )
    sep_index = inputs.input_ids.index(tokenizer.sep_token_id)
    inputs["start_positions"] = sep_index + 1 
    inputs["end_positions"] = sep_index + 2  
    return inputs


df = pd.read_excel("perguntas-respostas-stardix.xlsx")
df.dropna(inplace=True)

dataset = Dataset.from_pandas(df)
train_dataset = dataset.map(format_examples, batched=True)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
)


class MyTrainer(Trainer):
    def training_step(self, model, inputs):
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_positions = inputs["start_positions"]
        end_positions = inputs["end_positions"]
        loss_fct = torch.nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        self.log("train_loss", total_loss)
        return {"loss": total_loss}


trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
model.save_pretrained("./stardix")
