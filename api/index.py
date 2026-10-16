import os
import re
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
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

def extract_video_id(url: str) -> str:
    match = re.search(r'(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})', url)
    if match:
        return match.group(1)
    return None

def get_best_model():
    # Dynamically find the exact active model name available for your API key
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Prefer flash first, then pro
        flash_models = [m for m in models if 'flash' in m.lower()]
        if flash_models:
            return flash_models[0]
        return models[0] if models else "gemini-pro"
    except Exception:
        return "gemini-1.5-flash-latest"

class LectureRequest(BaseModel):
    url: str

@app.post("/api/generate")
async def generate_notes(req: LectureRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing on Vercel environment.")

    video_id = extract_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

    # Fetch Transcript using YouTubeTranscriptApi instance
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)
        
        # Try finding english or hindi
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'hi'])
        except Exception:
            transcript = next(iter(transcript_list))
            
        data = transcript.fetch()
        transcript_text = " ".join([item['text'] for item in data])
    except Exception as e:
        # Fallback to direct get_transcript call
        try:
            data = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([item['text'] for item in data])
        except Exception:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract subtitles for this video. Please ensure the video has English/Hindi captions enabled."
            )

    if not transcript_text.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    transcript_text = transcript_text[:12000]

    prompt = f"""You are an expert academic educator. Analyze this YouTube lecture transcript:
{transcript_text}

Generate:
1. A structured summary highlighting the main concepts and key takeaways.
2. 5 interactive revision flashcards (questions and answers).

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

    active_model_name = get_best_model()

    try:
        model = genai.GenerativeModel(
            model_name=active_model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed on {active_model_name}: {str(e)}")
