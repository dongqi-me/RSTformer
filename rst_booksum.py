import pandas as pd
import numpy as np
import random
import torch
from datasets import Dataset
from datasets import load_dataset, load_metric
from booksum_transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from booksum_transformers import LEDForConditionalGeneration

# load metric
rouge = load_metric("rouge")

train_dataset = load_dataset("kmfoda/booksum", split="train")
val_dataset = load_dataset("kmfoda/booksum", split="validation")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")

# max encoder length for led
encoder_max_length = 16384
decoder_max_length = 1024
batch_size = 1
learning_rate = 3e-9
weight_decay = 0.1
num_train_epochs = 30
random_seed = 3407

def set_seed(seed: int = 3407):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
set_seed(random_seed)

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["chapter"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
    )
    outputs = tokenizer(
        batch["summary"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch

# map train data
train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["chapter", "summary"],
)

# map val data
val_dataset = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["chapter", "summary"],
)

# set Python list to PyTorch tensor
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# set Python list to PyTorch tensor
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)


# enable fp16 apex training
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    num_train_epochs = num_train_epochs,
    fp16=False,
    # fp16_backend="apex",
    output_dir="./",
    logging_steps=1,
    eval_steps=1500,
    save_steps=1500,
    lr_scheduler_type="cosine",
    warmup_steps=1500,
    save_total_limit=5,
    gradient_accumulation_steps=1,
    optim= "adafactor",
    load_best_model_at_end = True,
    group_by_length=True,
    gradient_checkpointing= True,
    seed=3407
)

# compute Rouge score during validation
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge1_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1"])["rouge1"].mid
    rouge2_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    rougeL_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rougeL"])["rougeL"].mid
    rougeLsum_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rougeLsum"])["rougeLsum"].mid
    return {
        "rouge1_precision": round(rouge1_output.precision, 4),
        "rouge1_recall": round(rouge1_output.recall, 4),
        "rouge1_fmeasure": round(rouge1_output.fmeasure, 4),
        
        "rouge2_precision": round(rouge2_output.precision, 4),
        "rouge2_recall": round(rouge2_output.recall, 4),
        "rouge2_fmeasure": round(rouge2_output.fmeasure, 4),
        
        "rougeL_precision": round(rougeL_output.precision, 4),
        "rougeL_recall": round(rougeL_output.recall, 4),
        "rougeL_fmeasure": round(rougeL_output.fmeasure, 4),
        
        "rougeLsum_precision": round(rougeLsum_output.precision, 4),
        "rougeLsum_recall": round(rougeLsum_output.recall, 4),
        "rougeLsum_fmeasure": round(rougeLsum_output.fmeasure, 4),
    }


# load model + enable gradient checkpointing & disable cache for checkpointing
led = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384", gradient_checkpointing=True, use_cache=False)

# set generate hyperparameters
led.config.num_beams = 4
led.config.max_length = 1024
led.config.min_length = 256
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 3


# instantiate trainer
trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# start training
# torch.autograd.set_detect_anomaly(True)
trainer.train()
