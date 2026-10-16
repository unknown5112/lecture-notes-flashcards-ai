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
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

class LectureRequest(BaseModel):
    url: str

@app.post("/api/generate")
async def generate_notes(req: LectureRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing on Vercel environment.")

    video_id = extract_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL provided.")

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([t['text'] for t in transcript_list])
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not retrieve transcript. Ensure video has English/Hindi subtitles enabled.")

    # Transcript ko limit karte hain taaki tokens optimize rahein
    transcript_text = transcript_text[:12000]

    prompt = f"""You are an expert academic educator. Analyze this YouTube lecture transcript:
{transcript_text}

Generate:
1. A structured summary with bulleted Key Takeaways.
2. 5 interactive Flashcards formatted as valid JSON:
[
  {{"question": "Q1 text", "answer": "A1 text"}},
  ...
]

Return response in strict JSON format:
{{
  "summary": "markdown formatted summary text",
  "flashcards": [
    {{"question": "Q1", "answer": "A1"}}
  ]
}}
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type": "application/json"})
        resp = model.generate_content(prompt)
        data = json.loads(resp.text)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
