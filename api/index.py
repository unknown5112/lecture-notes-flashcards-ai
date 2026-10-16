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

class LectureRequest(BaseModel):
    direct_text: str

@app.post("/api/generate")
async def generate_notes(req: LectureRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable missing on Vercel.")

    raw_text = req.direct_text.strip()
    if not raw_text or len(raw_text) < 20:
        raise HTTPException(status_code=400, detail="Transcript text is missing or too short.")

    prompt = f"""You are an elite academic educator and note generator.
Analyze this lecture text:
\"\"\"{raw_text[:12000]}\"\"\"

Generate:
1. A structured, clear summary with bulleted Key Takeaways and Core Concepts.
2. Exactly 5 high-yield revision Flashcards.

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
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type": "application/json"})
        resp = model.generate_content(prompt)
        return json.loads(resp.text)
    except Exception as e:
        # Fallback parsing
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            match = re.search(r'\{.*\}', resp.text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"summary": resp.text, "flashcards": []}
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e2)}")
