import time

import torch

import datasets

from transformers import AutoTokenizer

from torch.utils.data import DataLoader

class DummyProfiler:
    def __init__(self):
        self.step_number = 0

    def step(self):
        self.step_number += 1

class DataManager:
    def __init__(self, dataset = "glue", partition = "mrpc", split = "train"):
        self.dataset_name = dataset
        self.partition = partition
        self.split = split

    def load_data(self, seq_len):

        ds = datasets.load_dataset(self.dataset_name, self.partition, split = self.split)
        print(f"Succesfully Loaded {self.dataset_name} dataset")
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.special_tokens_map["eos_token"]

        def tokenize_function(sample):
            return tokenizer(sample['sentence1'], sample['sentence2'], padding = True, truncation=True, return_tensors="pt", max_length=seq_len)

        tokenized_datasets = ds.map(tokenize_function, batched=True)

        train_dataset = tokenized_datasets

        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        return train_dataloader

#Randomly Generated Data

def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def get_time_stamp():
    cur_time = time.strftime("%d-%H:%M", time.localtime())
    return cur_time