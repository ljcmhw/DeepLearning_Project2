import torch.nn.functional as F
from transformers import EvalPrediction
import evaluate

def make_lora_target_modules(start: int, end: int):
    """
    Generate module names for LoRA injection in RoBERTa encoder layers [start, end].
    """
    base_modules = ["attention.self.query", "attention.self.value", "output.dense"]
    targets = []
    for layer in range(start, end + 1):
        for mod in base_modules:
            targets.append(f"encoder.layer.{layer}.{mod}")
    return targets

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Combine KL(student||teacher) on softened logits with cross-entropy on true labels.
    """
    # Soft KL divergence
    kl = F.kl_div(
        input=F.log_softmax(student_logits / temperature, dim=-1),
        target=F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)

    # Hard cross-entropy
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kl + (1 - alpha) * ce

def compute_metrics(pred: EvalPrediction):
    """
    Compute accuracy for Trainer evaluations.
    """
    preds = pred.predictions.argmax(-1)
    acc = evaluate.load("accuracy")
    result = acc.compute(predictions=preds, references=pred.label_ids)
    return {"accuracy": result["accuracy"]}

