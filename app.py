from fastapi import FastAPI, Form, Request
import uvicorn
import os
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from textSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict", response_class=HTMLResponse)
async def predict_route(request: Request, text: str = Form(...)):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text)
        return f"""
        <html>
            <body>
                <h1>Text Summarizer</h1>
                <h2>Original Text:</h2>
                <p>{text}</p>
                <h2>Summary:</h2>
                <p>{summary}</p>
                <br>
                <a href="/">Back to Home</a>
            </body>
        </html>
        """
    except Exception as e:
        return f"""
        <html>
            <body>
                <h1>Error</h1>
                <p>Error: {str(e)}</p>
                <a href="/">Back to Home</a>
            </body>
        </html>
        """

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)