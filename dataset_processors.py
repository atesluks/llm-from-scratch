import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from downloaders import CorpusDownloader

@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader creation."""
    batch_size: int = 4
    max_length: int = 256
    stride: int = 128
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 0

class TextDataset(Dataset):
    """Dataset for processing continuous text data with sliding window."""
    
    def __init__(self, text: str, tokenizer: Any, max_length: int, stride: int):
        """
        Initialize the text dataset.
        
        Args:
            text (str): Input text to process
            tokenizer: Tokenizer instance
            max_length (int): Maximum sequence length
            stride (int): Stride for sliding window
        """
        self.input_ids = []
        self.target_ids = []
        
        # Tokenize the entire text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        # Create sliding window chunks
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]

class DatasetProcessor:
    """Handles processing of datasets into format suitable for training."""
    
    @staticmethod
    def create_text_dataloader(
        tokenizer: Any,
        text: str,
        config: DataLoaderConfig
    ) -> DataLoader:
        """
        Create DataLoader for continuous text data.
        
        Args:
            tokenizer: Tokenizer instance
            text (str): Input text
            config (DataLoaderConfig): DataLoader configuration
            
        Returns:
            DataLoader: PyTorch DataLoader instance
        """
        dataset = TextDataset(
            text, 
            tokenizer, 
            config.max_length, 
            config.stride
        )
        
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            drop_last=config.drop_last,
            num_workers=config.num_workers
        )

    @staticmethod
    def create_wiki_dataloader(
        tokenizer: Any,
        dataset: Any,
        config: DataLoaderConfig
    ) -> DataLoader:
        """
        Create DataLoader for Wikipedia dataset.
        
        Args:
            tokenizer: Tokenizer instance
            dataset: HuggingFace dataset
            config (DataLoaderConfig): DataLoader configuration
            
        Returns:
            DataLoader: PyTorch DataLoader instance
        """
        # Process dataset
        tokenized_ds = dataset.map(
            lambda examples: DatasetProcessor._tokenize_and_chunk(
                examples, tokenizer, config.max_length, config.stride
            ),
            batched=True,
            remove_columns=dataset.column_names
        )

        # Filter incomplete sequences
        tokenized_ds = tokenized_ds.filter(
            lambda x: len(x["input_ids"]) == config.max_length
        )

        # Convert to PyTorch format
        tokenized_ds.set_format(type="torch", columns=["input_ids", "target_ids"])

        return DataLoader(
            tokenized_ds,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            drop_last=config.drop_last,
            num_workers=config.num_workers,
            collate_fn=DatasetProcessor._collate_fn
        )

    @staticmethod
    def _tokenize_and_chunk(
        examples: Dict[str, List[str]],
        tokenizer: Any,
        max_length: int,
        stride: int
    ) -> Dict[str, List[List[int]]]:
        """
        Tokenize and chunk text examples.
        
        Args:
            examples: Dictionary containing text examples
            tokenizer: Tokenizer instance
            max_length (int): Maximum sequence length
            stride (int): Stride for sliding window
            
        Returns:
            Dict: Dictionary containing input and target ids
        """
        all_input_ids = []
        all_target_ids = []

        for text in examples["text"]:
            # Clean and tokenize text
            text = CorpusDownloader._clean_text(text)
            t_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            
            # Create chunks
            for i in range(0, len(t_ids) - max_length, stride):
                input_chunk = t_ids[i:i + max_length]
                target_chunk = t_ids[i + 1:i + max_length + 1]
                all_input_ids.append(input_chunk)
                all_target_ids.append(target_chunk)

        return {
            "input_ids": all_input_ids,
            "target_ids": all_target_ids
        }

    @staticmethod
    def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function for batching examples.
        
        Args:
            batch: List of dictionaries containing input and target ids
            
        Returns:
            Tuple: Input and target tensors
        """
        input_ids = torch.stack([item["input_ids"] for item in batch])
        target_ids = torch.stack([item["target_ids"] for item in batch])
        return input_ids, target_ids
