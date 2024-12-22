import os
import re
import urllib.request
import logging
from datasets import load_dataset
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

class CorpusDownloader:
    """
    Module for downloading and processing text corpora for GPT model training.
    Handles both local text files and HuggingFace datasets.
    """
    @staticmethod
    def create_default_logger(log_level: int = logging.INFO) -> logging.Logger:
        """
        Create a default logger if none is provided.
        
        Args:
            log_level (int): Logging level to use
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger("CorpusDownloader")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(log_level)
        return logger
    
    def __init__(self, logger: Optional[logging.Logger] = None,
                 log_level: int = logging.INFO):
        """
        Initialize the CorpusDownloader.
        
        Args:
            logger (Optional[logging.Logger]): External logger instance
            log_level (int): Logging level (used only if logger is not provided)            
        """
        self.logger = logger or self.create_default_logger(log_level)        
            
    def download_alice_wonderland(self,file_path:str = 'corpuses/alice-wonderland.txt') -> str:
        """
        Download and process Alice in Wonderland text.
        
        Returns:
            str: Processed text content
        """
        url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"

        if not os.path.exists(file_path):
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode('utf-8')
                # Extract story text
                start = text_data.find("Alice was beginning to get very tired of sitting by her sister on the")
                end = text_data.find("*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***")
                text_data = text_data[start:end]
                # Clean text
                text_data = CorpusDownloader._clean_text(text_data)
                
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()

        return text_data

    def download_wikipedia(self) -> Any:
        """
        Download English Wikipedia dataset from HuggingFace.
        
        Returns:
            Dataset: HuggingFace dataset object
        """
        dataset_path = "wikimedia/wikipedia"
        dataset_subpath = "20231101.en"

        print(f"\nDownloading dataset from {dataset_path}, {dataset_subpath}")
        ds = load_dataset(dataset_path, dataset_subpath)
        print("Dataset download complete.")
        
        # Clean dataset
        ds = ds.remove_columns(["id", "url", "title"])
        return ds["train"]

    def _clean_text(self,text: str) -> str:
        """
        Clean text by removing special characters and extra whitespace.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        text = re.sub(r'\r\n|\r|\n|\*', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

