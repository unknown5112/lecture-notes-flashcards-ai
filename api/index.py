import os
import re
import json
import urllib.request
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

def extract_video_id(url: str) -> str:
    match = re.search(r'(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})', url)
    return match.group(1) if match else None

def get_transcript_via_web(video_id: str) -> str:
    # Lightweight web scraper that fetches timed text without blocked scrapers
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    url = f"https://www.youtube.com/watch?v={video_id}"
    req = urllib.request.Request(url, headers=headers)
    html = urllib.request.urlopen(req, timeout=6).read().decode('utf-8')
    
    # Check if caption tracks exist in player response
    match = re.search(r'"captionTracks":\s*(\[.*?\])', html)
    if not match:
        return ""
    
    tracks = json.loads(match.group(1))
    base_url = tracks[0].get('baseUrl', '')
    if not base_url:
        return ""
        
    xml_req = urllib.request.Request(base_url, headers=headers)
    xml_data = urllib.request.urlopen(xml_req, timeout=6).read().decode('utf-8')
    text_pieces = re.findall(r'<text[^>]*>(.*?)</text>', xml_data)
    
    # Clean XML entities
    clean_text = " ".join(text_pieces)
    clean_text = clean_text.replace('&amp;', '&').replace('&quot;', '"').replace('&#39;', "'")
    return clean_text

class LectureRequest(BaseModel):
    url: str = ""
    direct_text: str = ""

@app.post("/api/generate")
async def generate_notes(req: LectureRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable missing on Vercel.")

    raw_text = req.direct_text.strip()

    # Agar direct text nahi diya, toh YouTube URL se extract karo
    if not raw_text:
        if not req.url.strip():
            raise HTTPException(status_code=400, detail="Please provide either a YouTube URL or paste lecture text.")
        
        video_id = extract_video_id(req.url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

        try:
            raw_text = get_transcript_via_web(video_id)
        except Exception:
            raw_text = ""

    # Agar scrape nahi ho paya, clear instruction return karo fallback ke liye
    if not raw_text or len(raw_text) < 30:
        raise HTTPException(
            status_code=422,
            detail="YouTube IP-Restriction: Could not auto-fetch subtitles for this video from cloud server. Please switch to 'Paste Transcript' tab and paste the text directly!"
        )

    # Gemini 1.5 Flash processing
    prompt = f"""You are an elite academic educator and note generator.
Analyze this lecture text:
\"\"\"{raw_text[:12000]}\"\"\"

Generate:
1. A structured, clear summary with bulleted Key Takeaways and Core Concepts.
2. Exactly 5 high-yield revision Flashcards.

Return ONLY a strict valid JSON object:
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
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type": "application/json"})
        resp = model.generate_content(prompt)
        return json.loads(resp.text)
    except Exception as e:
        # Fallback without JSON enforcement
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            match = re.search(r'\{.*\}', resp.text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"summary": resp.text, "flashcards": []}
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e2)}")
