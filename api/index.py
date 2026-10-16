import os
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def get_active_model():
    """Returns the recommended latest model, with dynamic fallback if needed."""
    try:
        available = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        # Prefer gemini-3.8-flash first
        for m in available:
            if "3.8-flash" in m:
                return m
        # Fallback to any other flash model that is not retired
        for m in available:
            if "flash" in m.lower() and "2.5" not in m:
                return m
        if available:
            return available[0]
    except Exception:
        pass
    return "models/gemini-3.8-flash"

class LectureRequest(BaseModel):
    direct_text: str

@app.post("/api/generate")
async def generate_notes(req: LectureRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing in Vercel settings.")

    text = req.direct_text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Text is too short. Please provide at least a couple of sentences.")

    chosen_model = get_active_model()

    prompt = f"""You are an elite academic educator and study tool architect.
Analyze the following lecture material:
\"\"\"{text[:10000]}\"\"\"

Generate:
1. A structured summary highlighting key concepts and actionable takeaways.
2. Exactly 5 active-recall Flashcards (clear question and concise answer).

Format response strictly as valid JSON:
{{
  "summary": "markdown formatted notes",
  "flashcards": [
    {{"question": "Question 1", "answer": "Answer 1"}},
    {{"question": "Question 2", "answer": "Answer 2"}},
    {{"question": "Question 3", "answer": "Answer 3"}},
    {{"question": "Question 4", "answer": "Answer 4"}},
    {{"question": "Question 5", "answer": "Answer 5"}}
  ]
}}
"""

    try:
        model = genai.GenerativeModel(model_name=chosen_model)
        resp = model.generate_content(prompt)
        raw_text = resp.text.strip()

        # Clean JSON markdown code blocks if present
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        return json.loads(raw_text)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Generation failed on model '{chosen_model}': {str(e)}"
        )
