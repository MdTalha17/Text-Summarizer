import os
from textSummarizer.entity import DataTransformationConfig
from pathlib import Path
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    def convert_examples_to_features(self, example_batch):
        print("pad_token_id:", self.tokenizer.pad_token_id)
        input_encodings = self.tokenizer(
            example_batch["dialogue"],
            max_length=1024,
            truncation=True,
            padding="max_length"
        )
        target_encodings = self.tokenizer(
            text_target=example_batch["summary"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        
        labels = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
            for label in target_encodings["input_ids"]
        ]
        
        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": labels
        }

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))