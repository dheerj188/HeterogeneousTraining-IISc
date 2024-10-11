import os
from typing import List, Tuple

from datasets import load_dataset
from transformers import GPT2TokenizerFast, DataCollatorForLanguageModeling

import torch.profiler
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def initialize_data_loader(batch_size, num_data_workers) -> Tuple[DataLoader, DataLoader]:
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.special_tokens_map["eos_token"]

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    #dataset = load_dataset("glue", "mrpc")
    dataset = load_dataset("/scratch/kevinmahesh/shell-env/datasets/c4/en", streaming = True)
    dataset = dataset.map(
        lambda samples: tokenizer(samples["text"], padding=True, truncation=True, max_length=32),
        batched=True,
        remove_columns=["text", "url", "timestamp"]
    ).with_format(type="torch")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        persistent_workers=num_data_workers > 0,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        persistent_workers=num_data_workers > 0,
        pin_memory=True,
        collate_fn=collator,
    )
    return train_loader, val_loader