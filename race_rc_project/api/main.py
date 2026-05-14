from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from typing import Optional
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
from preprocessing import load_race_data
from inference import RCInferenceEngine


app = FastAPI(title="NeuraQuiz API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = None
train_df = None


@app.on_event("startup")
async def startup_event():
    global engine, train_df
    try:
        engine = RCInferenceEngine(PROJECT_ROOT)
        print("[API] Inference Engine loaded successfully.")
    except Exception as e:
        print(f"[API] Error loading Inference Engine: {e}")

    try:
        train_df, _, _ = load_race_data(
            os.path.join(PROJECT_ROOT, "..", "dataset"))
        print("[API] Dataset loaded successfully.")
    except Exception as e:
        print(f"[API] Error loading Dataset: {e}")


class QuizRequest(BaseModel):
    article: str
    question: Optional[str] = ""
    correct_answer: Optional[str] = ""


class VerifyRequest(BaseModel):
    article: str
    question: str
    selected_option: str


@app.get("/api/random_article")
async def get_random_article():
    if train_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    sample = train_df.sample(1).iloc[0]
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    correct_idx = label_map.get(sample["answer"], 0)
    correct_answer = sample[["A", "B", "C", "D"]][correct_idx]

    return {
        "article": sample["article"],
        "question": sample["question"],
        "correct_answer": correct_answer,
    }


@app.post("/api/quiz")
async def generate_quiz(req: QuizRequest):
    if engine is None:
        raise HTTPException(
            status_code=500,
            detail="Inference engine not loaded")
    try:
        quiz = engine.generate_quiz(
            req.article, req.question, req.correct_answer)
        return quiz
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verify")
async def verify_answer(req: VerifyRequest):
    if engine is None:
        raise HTTPException(
            status_code=500,
            detail="Inference engine not loaded")
    try:
        result = engine.verify_answer(
            req.article, req.question, req.selected_option)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
