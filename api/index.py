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

def is_youtube_url(text: str) -> bool:
    return bool(re.search(r'(?:youtube\.com\/(?:watch\?v=|embed\/|shorts\/)|youtu\.be\/)[0-9A-Za-z_-]{11}', text))

class LectureRequest(BaseModel):
    content: str

@app.post("/api/generate")
async def generate_notes(req: LectureRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing in Vercel settings.")

    user_input = req.content.strip()
    if len(user_input) < 10:
        raise HTTPException(status_code=400, detail="Please enter a valid YouTube link or lecture text.")

    # Agar YouTube link hai, toh Gemini ko direct video target karne do
    if is_youtube_url(user_input):
        prompt = f"""You are an elite academic educator.
Watch and analyze this YouTube video thoroughly: {user_input}

Generate:
1. A detailed, structured lecture summary specific to this video's unique discussion, key insights, and takeaways.
2. Exactly 5 active-recall Flashcards directly testing concepts explained in this specific video.

Return ONLY valid JSON matching this schema:
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
    else:
        # Agar user ne custom text paste kiya hai
        prompt = f"""You are an elite academic educator.
Analyze this study lecture material:
\"\"\"{user_input[:10000]}\"\"\"

Generate:
1. A detailed structured summary with key concepts and takeaways.
2. Exactly 5 active-recall Flashcards (question and concise answer).

Return ONLY valid JSON matching this schema:
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
            model_name="models/gemini-3.8-flash",
            generation_config={"response_mime_type": "application/json"}
        )
        resp = model.generate_content(prompt)
        raw = resp.text.strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        return json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
