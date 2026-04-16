from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline
import torch


class PredictionPipeline:
    def __init__(self):
        HF_MODEL = "Md-Talha017/bart-samsum"

        #self.config = ConfigurationManager().get_model_evaluation_config()

        device = 0 if torch.cuda.is_available() else -1

        #self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)

        self.pipe = pipeline(
            "summarization",
            model=HF_MODEL,
            #tokenizer=self.tokenizer,
            device=device
        )

        print("Model loaded successfully")

    def predict(self, text, length="medium"):
        if length == "short":
            max_len, min_length = 40, 10
            length_penalty = 0.5
        elif length == "long":
            max_len, min_length = 300, 150
            length_penalty = 2.0
        else:
            max_len, min_length = 120, 50
            length_penalty = 1.0

        gen_kwargs = {
            "length_penalty": length_penalty,
            "num_beams": 4,
            "max_length": max_len,
            "min_length": min_length,
            "do_sample": False,
            "early_stopping": True
        }

        with torch.no_grad():
            output = self.pipe(text, **gen_kwargs)[0]["summary_text"]

        return output