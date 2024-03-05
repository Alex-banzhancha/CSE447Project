# %%
import transformers
print(transformers.__version__)

from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
import numpy as np
import torch

# %%
model_checkpoint = "t5-base-finetuned-multi-news/checkpoint-112000"

# %%
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# %%
raw_datasets = load_dataset("multi_news")
metric = load("rouge")

# %%
max_input_length = 2048
max_target_length = 256

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# %%
prefix = "summarize: "
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# %%
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# %%
batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-multi-news",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=False,
)

# %%
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# %%
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# %%
results = trainer.predict(tokenized_datasets["test"], metric_key_prefix="test", do_sample=True, max_new_tokens=256)

f = open("test.txt", "w")
for i in range(len(results.predictions)):
    f.write(tokenizer.decode(results.predictions[i])[8:])
    f.write("\n")
f.close()

results = trainer.predict(tokenized_datasets["validation"], metric_key_prefix="validation", do_sample=True, max_new_tokens=256)

f = open("val.txt", "w")
for i in range(len(results.predictions)):
    f.write(tokenizer.decode(results.predictions[i])[8:])
    f.write("\n")
f.close()

results = trainer.predict(tokenized_datasets["train"], metric_key_prefix="train", do_sample=True, max_new_tokens=256)

f = open("train.txt", "w")
for i in range(len(results.predictions)):
    f.write(tokenizer.decode(results.predictions[i])[8:])
    f.write("\n")
f.close()
