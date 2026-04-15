from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging

from textSummarizer.pipeline.prediction import PredictionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor: PredictionPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading model...")
    try:
        predictor = PredictionPipeline()
        
        predictor.predict("Warmup text", "short")
        
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError("Model initialization failed.") from e
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="AI Text Summarizer",
    description="Summarize text using a BART-based model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

MAX_TEXT_LENGTH = 4000

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_route(request: Request, text: str = Form(...), length: str = Form("medium")):
    text = text.strip()

    if not text:
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "text": text,
                "summary": None,
                "error": "Input text cannot be empty.",
            },
        )

    if len(text) > MAX_TEXT_LENGTH:
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "text": text,
                "summary": None,
                "error": f"Input text exceeds the {MAX_TEXT_LENGTH:,} character limit.",
            },
        )

    if predictor is None:
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "text": text,
                "summary": None,
                "word_count": 0,
                "error": "Model is still loading. Please try again in a few seconds.",
            },
        )

    try:
        logger.info(f"Summarizing text of length {len(text)}...")
        summary = predictor.predict(text, length)
        logger.info("Summarization successful.")

        word_count = len(summary.strip().split()) if summary else 0

        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "text": text,
                "summary": summary,
                "word_count": word_count,
                "length": length,
                "error": None,
            },
        )

    except Exception as e:
        logger.exception("Prediction failed.")
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "text": text,
                "summary": None,
                "word_count": 0,
                "error": f"Prediction failed: {str(e)}",
            },
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)