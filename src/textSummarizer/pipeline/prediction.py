from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline
import torch


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

        device = 0 if torch.cuda.is_available() else -1

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)

        self.pipe = pipeline(
            "summarization",
            model=self.config.model_path,
            tokenizer=self.tokenizer,
            device=device
        )

        print("Model loaded successfully")

    def predict(self, text, length="medium"):
        if length == "short":
            max_len, min_length = 60, 20
        elif length == "long":
            max_len, min_length = 180, 80
        else:
            max_len, min_length = 120, 50

        gen_kwargs = {
            "length_penalty": 1.0,
            "num_beams": 2,   
            "max_length": max_len,
            "min_length": min_length,
            "do_sample": False
        }

        print("Dialogue:")
        print(text)

        with torch.no_grad():
            output = self.pipe(text, **gen_kwargs)[0]["summary_text"]

        print("\nModel Summary:")
        print(output)

        return output