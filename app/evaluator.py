# [file name]: evaluator.py

import json
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

# ❌ DO NOT LOAD MODEL HERE
model = None   # GLOBAL EMPTY MODEL


def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading ML model for first time...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


# Get the absolute path to questions.json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_FILE = os.path.join(os.path.dirname(BASE_DIR), "questions.json")


def load_questions():
    """Load questions from JSON file"""
    print(f"Looking for questions at: {QUESTIONS_FILE}")

    if not os.path.exists(QUESTIONS_FILE):
        print("Questions file not found, creating sample...")
        sample_data = {
            "questions": [
                {
                    "id": 1,
                    "question": "What is Machine Learning?",
                    "model_answer": "Machine Learning is a subset of Artificial Intelligence that enables systems to learn from data."
                }
            ]
        }
        with open(QUESTIONS_FILE, 'w') as f:
            json.dump(sample_data, f, indent=2)
        return sample_data

    with open(QUESTIONS_FILE, 'r') as f:
        data = json.load(f)
        print(f"Loaded {len(data.get('questions', []))} questions")
        return data


def get_model_answer_by_id(question_id: int):
    data = load_questions()
    for question in data.get("questions", []):
        if question["id"] == question_id:
            return question["model_answer"]
    return None


def get_question_text_by_id(question_id: int):
    data = load_questions()
    for question in data.get("questions", []):
        if question["id"] == question_id:
            return question["question"]
    return None


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def evaluate_answer(model_answer: str, student_answer: str, threshold: float = 0.70):
    """Evaluate similarity between model answer and student answer"""

    model_local = get_model()   # 🔥 LOAD MODEL ONLY WHEN API IS CALLED

    model_answer = clean_text(model_answer)
    student_answer = clean_text(student_answer)

    emb1 = model_local.encode([model_answer])
    emb2 = model_local.encode([student_answer])

    score = float(cosine_similarity(emb1, emb2)[0][0])
    percentage = round(score * 100, 2)

    status = "PASS" if score >= threshold else "FAIL"

    return {
        "similarity_score": round(score, 4),
        "similarity_percentage": percentage,
        "status": status
    }


def evaluate_answer_by_id(question_id: int, student_answer: str, threshold: float = 0.70):
    model_answer = get_model_answer_by_id(question_id)
    question_text = get_question_text_by_id(question_id)

    if model_answer is None:
        raise ValueError(f"Question with ID {question_id} not found")

    result = evaluate_answer(model_answer, student_answer, threshold)
    result["question_text"] = question_text
    result["model_answer"] = model_answer

    return result
