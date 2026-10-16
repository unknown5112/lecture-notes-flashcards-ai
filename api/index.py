import os
import re
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

def extract_clean_url(url: str) -> str:
    match = re.search(r'(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})', url)
    if match:
        video_id = match.group(1)
        return f"https://www.youtube.com/watch?v={video_id}"
    return None

class LectureRequest(BaseModel):
    url: str

@app.post("/api/generate")
async def generate_notes(req: LectureRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable missing on Vercel.")

    clean_yt_url = extract_clean_url(req.url)
    if not clean_yt_url:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL. Please provide a valid YouTube video link.")

    prompt = f"""You are an expert academic educator.
Analyze this YouTube video: {clean_yt_url}

Generate:
1. A structured summary highlighting the main concepts and key takeaways.
2. 5 interactive revision flashcards (questions and answers).

Return the response in strict JSON format matching this schema:
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
        response = model.generate_content([clean_yt_url, prompt])
        
        if not response or not response.text:
            raise Exception("Empty response from AI model.")

        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
