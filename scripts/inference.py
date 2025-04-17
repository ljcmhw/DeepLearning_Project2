#!/usr/bin/env python
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_csv", default="submission.csv")
    return p.parse_args()

def main():
    args = parse_args()

    # Load tokenizer and base model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    base_model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    model = PeftModel.from_pretrained(base_model, args.checkpoint).cuda().eval()

    # Load test texts
    with open(args.test_file, "rb") as f:
        test_texts = pickle.load(f)

    # Tokenize
    enc = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt").to("cuda")

    # Predict
    with torch.no_grad():
        logits = model(**enc).logits.cpu().numpy()
    preds = np.argmax(logits, axis=-1)

    # Save to CSV
    df = pd.DataFrame({"ID": np.arange(len(preds)), "Label": preds})
    df.to_csv(args.output_csv, index=False)
    print(f"Submission saved to {args.output_csv}")

if __name__ == "__main__":
    main()

