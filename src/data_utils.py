from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
import torch

class CustomDataset(Dataset):
    """
    Wrap a Hugging Face Dataset to include a stable 'idx' field for distillation.
    """
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.indices = list(range(len(hf_dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        item["idx"] = self.indices[idx]
        return item

class DataCollatorWithIdx(DataCollatorWithPadding):
    """
    Inherit HF's DataCollatorWithPadding, but preserve the 'idx' field in the batch.
    """
    def __call__(self, features):
        batch = super().__call__(features)
        batch["idx"] = torch.tensor([f["idx"] for f in features], dtype=torch.long)
        return batch

