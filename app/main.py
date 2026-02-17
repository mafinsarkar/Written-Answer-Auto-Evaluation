import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.evaluator import evaluate_answer

app = FastAPI(title="Answer Evaluation System")
QUESTIONS_PATH = Path("questions.json")

# CORS (allow frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnswerRequest(BaseModel):
    question_id: int | None = None
    model_answer: str
    student_answer: str
    pass_threshold: float = 0.70


def load_questions() -> list[dict[str, Any]]:
    if not QUESTIONS_PATH.exists():
        return []

    raw = QUESTIONS_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        data = data.get("questions", [])

    if not isinstance(data, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        question_id = item.get("id", item.get("question_id"))
        try:
            question_id = int(question_id) if question_id is not None else None
        except (TypeError, ValueError):
            question_id = None

        normalized.append(
            {
                "id": question_id,
                "question": str(item.get("question", "")),
                "model_answer": str(item.get("model_answer", "")),
            }
        )

    return normalized


@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path("frontend/index.html")
    return html_path.read_text(encoding="utf-8")


@app.get("/questions")
def questions():
    return {"questions": load_questions()}


@app.get("/questions/{question_id}")
def question_by_id(question_id: int):
    for question in load_questions():
        if question.get("id") == question_id:
            return question
    raise HTTPException(status_code=404, detail="Question not found")


@app.post("/evaluate")
def evaluate(req: AnswerRequest):
    result = evaluate_answer(
        model_answer=req.model_answer,
        student_answer=req.student_answer,
        threshold=req.pass_threshold,
    )
    result["question_id"] = req.question_id
    return result
