import os
import json
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

class LectureRequest(BaseModel):
    direct_text: str

@app.post("/api/generate")
async def generate_notes(req: LectureRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing in Vercel settings.")

    text = req.direct_text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Text is too short. Please provide at least a couple of sentences.")

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
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )
        resp = model.generate_content(prompt)
        return json.loads(resp.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
