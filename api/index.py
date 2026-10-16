import os
import re
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
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
        r'(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})',
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
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable missing on Vercel.")

    video_id = extract_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL. Please provide a standard YouTube video link.")

    transcript_text = ""
    try:
        # Transcript list fetch karo (manually created + auto-generated dono ke liye)
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Priority order: English -> Hindi -> Koi bhi available generated transcript
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB', 'hi'])
        except Exception:
            # Agar primary languages na milein toh pehli available transcript utha lo
            transcript = next(iter(transcript_list))

        fetched_data = transcript.fetch()
        transcript_text = " ".join([t['text'] for t in fetched_data])

    except (TranscriptsDisabled, NoTranscriptFound):
        raise HTTPException(
            status_code=400, 
            detail="This video does not have closed captions (CC/Subtitles) enabled by the creator. Please try a lecture video that has subtitles/CC turned on."
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Unable to fetch transcript: {str(e)}. Try testing with a video that has English CC enabled."
        )

    if not transcript_text.strip():
        raise HTTPException(status_code=400, detail="Extracted transcript is empty.")

    # Token limits ke hisaab se slice karo
    transcript_text = transcript_text[:12000]

    prompt = f"""You are an expert educator. Analyze this YouTube lecture transcript:
{transcript_text}

Provide:
1. A concise, structured summary highlighting key concepts.
2. 5 revision flashcards with questions and direct answers.

Return ONLY strict valid JSON matching this schema:
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
            "gemini-1.5-flash", 
            generation_config={"response_mime_type": "application/json"}
        )
        resp = model.generate_content(prompt)
        data = json.loads(resp.text)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")
