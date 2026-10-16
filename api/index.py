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

class LectureRequest(BaseModel):
    direct_text: str

@app.post("/api/generate")
async def generate_notes(req: LectureRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing in Vercel settings.")

    text = req.direct_text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Text too short. Please provide at least a couple of sentences.")

    prompt = f"""You are an elite academic educator. Analyze this study material:
\"\"\"{text[:10000]}\"\"\"

Generate:
1. A structured summary with clear Key Takeaways.
2. Exactly 5 active-recall Flashcards (question and short answer).

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

    # Direct targeted models without calling slow list_models()
    candidate_models = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]
    last_err = None

    for m_name in candidate_models:
        try:
            model = genai.GenerativeModel(
                model_name=m_name,
                generation_config={"response_mime_type": "application/json"}
            )
            resp = model.generate_content(prompt)
            raw = resp.text.strip()

            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            return json.loads(raw)
        except Exception as err:
            last_err = err
            continue

    raise HTTPException(status_code=500, detail=f"AI generation failed: {str(last_err)}")
