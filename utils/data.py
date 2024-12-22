# Helper functions for downloading and processing the data
import os
import re
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
from dataset_processors import load_dataset

# Downloading text data
def download_alice_wonderland_txt_data():
    file_path = "alice-wonderland.txt"
    url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
            # Extract just the story text between the specified start and end points
            start = text_data.find("Alice was beginning to get very tired of sitting by her sister on the")
            end = text_data.find("*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***")
            text_data = text_data[start:end]
            # Replace all newlines with a space, remove "*" and redundant spaces
            text_data = re.sub(r'\r\n|\r|\n|\*', ' ', text_data)
            text_data = re.sub(r'\s+', ' ', text_data)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    return text_data

# Downloading English wikipedia dataset from HuggingFace
def download_eng_wiki():
    dataset_path = "wikimedia/wikipedia"
    dataset_subpath = "20231101.en"

    print(f"\nStarting downloading dataset from {dataset_path}, {dataset_subpath}")
    ds = load_dataset(dataset_path, dataset_subpath)

    print("Done downloading dataset.")
    return ds

# Processing data
def get_wiki_dataset():
    ds = download_eng_wiki()
    ds = ds.remove_columns(["id", "url", "title"])
    return ds["train"]

# Dataset class for text data
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Dataloader for the text data
def create_dataloader_v1(tokenizer, txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

# Helper function for the wikipedia dataset tokenization and chunking
def tokenize_and_chunk(examples, tokenizer, max_length, stride):
    all_input_ids = []
    all_target_ids = []

    for text in examples["text"]:
        # Replace multiple newlines with single space and remove trailing spaces
        text = re.sub(r"\s+", " ", text).strip()
        t_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        # Slice into chunks
        for i in range(0, len(t_ids) - max_length, stride):
            input_chunk = t_ids[i:i + max_length]
            target_chunk = t_ids[i + 1:i + max_length + 1]
            all_input_ids.append(input_chunk)
            all_target_ids.append(target_chunk)

    return {"input_ids": all_input_ids, "target_ids": all_target_ids}

# Helper function for batching wikipedia dataset
def collate_fn(batch):
    # batch is a list of dictionaries, each with "input_ids" and "target_ids"
    input_ids = torch.stack([item["input_ids"] for item in batch])
    target_ids = torch.stack([item["target_ids"] for item in batch])
    return input_ids, target_ids

# Creating dataloader for the wikipedia dataset
def create_dataloader_HFD(tokenizer, ds, batch_size=4, max_length=256,
                          stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Map tokenization over the dataset
    tokenized_ds = ds.map(
        lambda examples: tokenize_and_chunk(examples, tokenizer, max_length, stride),
        batched=True,
        remove_columns=ds.column_names
    )

    # Filter out any incomplete sequences (so that all batches are same length)
    tokenized_ds = tokenized_ds.filter(lambda x: len(x["input_ids"]) == max_length)

    # Convert dataset to torch format
    tokenized_ds.set_format(type="torch", columns=["input_ids", "target_ids"])

    # Create DataLoader with a custom collate function to return tensors
    dataloader = DataLoader(
        tokenized_ds, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers, collate_fn=collate_fn
    )

    return dataloader