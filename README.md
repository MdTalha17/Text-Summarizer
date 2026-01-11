# 📝 Text Summarizer

An end-to-end text summarization system built with BART (Bidirectional and Auto-Regressive Transformers) model, featuring a complete ML pipeline from data ingestion to model deployment with a web interface.

## 🚀 Features

- **End-to-End ML Pipeline**: Complete workflow from data ingestion to model evaluation
- **BART Model**: State-of-the-art transformer-based summarization using Facebook's BART-base
- **SAMSum Dataset**: Trained on dialogue summarization dataset
- **RESTful API**: FastAPI-based web service for real-time predictions
- **Web Interface**: Simple HTML frontend for easy text summarization
- **Model Evaluation**: Automatic ROUGE score calculation for model performance
- **Docker Support**: Containerized deployment ready
- **Modular Architecture**: Clean, maintainable codebase with separation of concerns

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.11+**
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Pre-trained models and tokenizers
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server

### Key Libraries
- `transformers` - Hugging Face transformers library
- `datasets` - Dataset handling and processing
- `rouge-score` - ROUGE metric for evaluation
- `sacrebleu` - BLEU score calculation
- `pandas` - Data manipulation
- `nltk` - Natural language processing
- `PyYAML` - Configuration management

## 📋 Prerequisites

- Python 3.11 or higher
- pip or conda package manager
- CUDA-capable GPU (optional, for faster training)
- At least 8GB RAM (16GB recommended)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MdTalha17/TEXT-SUMMARIZER.git
cd TEXT-SUMMARIZER
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 📁 Project Structure

```
Text-Summarizer/
├── artifacts/                    # Generated artifacts
│   ├── data_ingestion/           # Raw and processed datasets
│   ├── data_transformation/      # Transformed datasets
│   ├── data_validation/          # Validation status
│   ├── model_trainer/            # Trained models and tokenizers
│   └── model_evaluation/         # Evaluation metrics
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
├── logs/                         # Application logs
├── research/                     # Jupyter notebooks for experimentation
├── src/                         # Source code
│   └── textSummarizer/
│       ├── components/           # Core components
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       ├── config/              # Configuration management
│       ├── constants/           # Constants
│       ├── entity/             # Data entities
│       ├── logging/            # Logging setup
│       ├── pipeline/           # Training and prediction pipelines
│       └── utils/              # Utility functions
├── templates/                   # HTML templates
│   └── index.html              # Web interface
├── app.py                      # FastAPI application
├── main.py                     # Training pipeline entry point
├── params.yaml                 # Training hyperparameters
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── Dockerfile                  # Docker configuration
```

## 🔄 ML Pipeline Workflow

The project follows a structured 5-stage pipeline:

### Stage 1: Data Ingestion
- Downloads SAMSum dataset from remote source
- Extracts and organizes data into train/validation/test splits
- Saves data in structured format

### Stage 2: Data Validation
- Validates data integrity and structure
- Checks for required files and formats
- Generates validation status report

### Stage 3: Data Transformation
- Tokenizes text using BART tokenizer
- Preprocesses dialogue and summary pairs
- Prepares data for model training

### Stage 4: Model Training
- Fine-tunes BART-base model on SAMSum dataset
- Implements training with configurable hyperparameters
- Saves model checkpoints and tokenizer

### Stage 5: Model Evaluation
- Evaluates model performance using ROUGE metrics
- Generates evaluation reports
- Saves metrics for analysis

## 🚀 Usage

### Training the Model

Run the complete training pipeline:

```bash
python main.py
```

Or trigger training via API:

```bash
curl http://localhost:8080/train
```

### Running the Web Application

Start the FastAPI server:

```bash
python app.py
```

The application will be available at:
- **Web Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Alternative Docs**: http://localhost:8080/redoc

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:8080`
2. Enter the text you want to summarize in the text area
3. Click "Summarize" to generate the summary
4. View the generated summary on the results page

### API Endpoints

#### POST `/predict`
Generate a summary for the provided text.

**Request:**
```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=Your text to summarize here"
```

**Response:**
```html
HTML page with original text and generated summary
```

#### GET `/train`
Trigger model training pipeline.

**Response:**
```
Training successful !!
```

#### GET `/`
Redirects to web interface (index.html)

## ⚙️ Configuration

### Model Configuration (`config/config.yaml`)

```yaml
model_trainer:
  root_dir: artifacts/model_trainer
  data_path: artifacts/data_transformation/samsum_dataset
  model_ckpt: facebook/bart-base

model_evaluation:
  model_path: artifacts/model_trainer/bart-samsum-model
  tokenizer_path: artifacts/model_trainer/tokenizer
```

### Training Parameters (`params.yaml`)

Key hyperparameters:
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Gradient accumulation steps
- `learning_rate`: Learning rate (if specified)
- `fp16`: Mixed precision training

## 📊 Model Performance

The model is evaluated using ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics:
- **ROUGE-1**: Measures overlap of unigrams
- **ROUGE-2**: Measures overlap of bigrams
- **ROUGE-L**: Measures longest common subsequence

Evaluation metrics are saved in `artifacts/model_evaluation/metrics.csv`

## 🐳 Docker Deployment

Build the Docker image:

```bash
docker build -t text-summarizer .
```

Run the container:

```bash
docker run -p 8080:8080 text-summarizer
```

## 🔍 Development

### Running Tests

```bash
# Add test commands here when tests are implemented
```

### Code Structure

- **Components**: Modular components for each pipeline stage
- **Pipeline**: Orchestrates the training workflow
- **Config**: Centralized configuration management
- **Utils**: Reusable utility functions

## 📝 Notes

- The model uses BART-base which requires significant computational resources for training
- For inference, the model can run on CPU but GPU is recommended for faster processing
- The SAMSum dataset is automatically downloaded during data ingestion
- Model checkpoints are saved during training for resuming or evaluation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 👤 Author

**MdTalha17**

- GitHub: [@MdTalha17](https://github.com/MdTalha17)
- Email: talhamoh017@gmail.com

## 🙏 Acknowledgments

- Hugging Face for the transformers library and BART model
- Facebook AI Research for the BART architecture
- SAMSum dataset creators
- FastAPI community for the excellent web framework

## 📚 References

- [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SAMSum Dataset](https://huggingface.co/datasets/samsum)

---

⭐ If you find this project helpful, please consider giving it a star!
