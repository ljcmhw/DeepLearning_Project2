#!/usr/bin/env python
import os
import argparse
import torch
import pickle
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from src.data_utils import CustomDataset, DataCollatorWithIdx
from src.lora_model import make_lora_target_modules, distillation_loss, compute_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="results/model_checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load and tokenize AG News
    raw = load_dataset("ag_news", split="train")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    tokenized = raw.map(lambda b: tokenizer(b["text"], truncation=True), batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    # 2. Create train/eval splits
    split = tokenized.train_test_split(test_size=640, seed=42)
    train_ds = CustomDataset(split["train"])
    eval_ds  = CustomDataset(split["test"])
    collator = DataCollatorWithIdx(tokenizer, return_tensors="pt")

    num_labels = raw.features["label"].num_classes
    id2label  = {i: n for i, n in enumerate(raw.features["label"].names)}

    # 3. Teacher logit generation
    teacher = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", id2label=id2label, label2id={v:k for k,v in id2label.items()}
    ).cuda().eval()

    all_logits = []
    loader = DataLoader(train_ds, batch_size=64, collate_fn=collator)
    for batch in loader:
        idxs = batch.pop("idx")
        inputs = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            logits = teacher(**inputs).logits.cpu()
        all_logits.append((idxs, logits))

    teacher_logits = torch.zeros(len(train_ds), num_labels)
    for idxs, logits in all_logits:
        teacher_logits[idxs] = logits

    # 4. Student + LoRA setup
    student = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", id2label=id2label, label2id={v:k for k,v in id2label.items()}
    )
    target_mods = make_lora_target_modules(6, 11)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=target_mods, bias="none", task_type="SEQ_CLS"
    )
    peft_model = get_peft_model(student, lora_cfg)
    peft_model.print_trainable_parameters()

    # 5. TrainingArguments & Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        load_best_model_at_end=True,
        report_to=[]
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        # override loss to include distillation
        compute_loss=lambda model, inputs: distillation_loss(
            model(**inputs).logits,
            teacher_logits[inputs["idx"]].to(model.device),
            inputs["labels"].to(model.device),
            temperature=2.0, alpha=0.7
        )
    )

    # 6. Train & save
    trainer.train()
    metrics = trainer.evaluate()
    print(f"Final Eval Accuracy: {metrics['eval_accuracy']:.4f}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()

