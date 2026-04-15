from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HF_USERNAME = "Md-Talha017"
REPO_NAME = "bart-samsum"

model = AutoModelForSeq2SeqLM.from_pretrained("artifacts/model_trainer/bart-samsum-model")
tokenizer = AutoTokenizer.from_pretrained("artifacts/model_trainer/tokenizer")

model.push_to_hub(f"{HF_USERNAME}/{REPO_NAME}")
tokenizer.push_to_hub(f"{HF_USERNAME}/{REPO_NAME}")

print(f"Done! Model at: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")