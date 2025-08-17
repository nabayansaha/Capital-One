import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage
from agents.states import Session
from krishimitra import KrishiMitra_pipeline
import re
import json
# Import ASR pipeline
from asr.asr import (
    transcribe_audio_with_detection,
    translate_chunked_nllb_indic2en,
    translate_chunked_nllb,
    text_to_speech_elevenlabs,
)

# Create FastAPI app
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # adjust to frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
graph = KrishiMitra_pipeline()
sessions: Dict[str, Session] = {}  # store sessions by user_id


# ===============================
# Models for Text Chat
# ===============================
class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    chat_history: List[Dict[str, Any]]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # get or create session
    if req.user_id not in sessions:
        session = Session()
        session.pdf_path = "Dataset/KrishiMitra.docx"
        sessions[req.user_id] = session
    else:
        session = sessions[req.user_id]

    # append user message
    session.messages.append(HumanMessage(content=req.message))

    # invoke pipeline
    result = graph.invoke(session)

    ai_response = result.get("response", "No response")
    
    chat_history = [
        {"type": "human", "content": m.content} if m.type == "human"
        else {"type": "ai", "content": m.content}
        for m in session.messages
    ]
    match = re.search(r"chat_history=\[.*?content='(.*?)'\)", ai_response, re.DOTALL)

    if match:
        ai = match.group(1).replace("\\n", "\n")
        print("AI Response:\n", ai_response)
    else:
        print("No AI response found")

    try:
        data = json.loads(ai)  # if it's valid JSON
        if isinstance(data, dict):
            # format dict into readable string
            formatted = []
            for k, v in data.items():
                if isinstance(v, list):
                    v_str = ", ".join(str(x) for x in v)
                else:
                    v_str = str(v)
                formatted.append(f"{k.replace('_',' ').title()}: {v_str}")
            ai = "\n".join(formatted)
    except Exception:
        # if not JSON, leave as is
        pass

    print("AI Response:\n", ai)

    return ChatResponse(response=ai.strip(), chat_history=chat_history)


# ===============================
# Models for Audio Chat
# ===============================
class AudioChatResponse(BaseModel):
    original_text: str
    detected_language: str
    english_text: str
    chatbot_response_en: str
    chatbot_response_local: str
    audio_file: Optional[str]  # path to TTS audio file


@app.post("/chat_audio", response_model=AudioChatResponse)
async def chat_audio(user_id: str = Form(...), file: UploadFile = File(...)):
    """
    Full pipeline:
    Speech -> Text (lang detect) -> English -> RAG chatbot -> English response
    -> Local language -> Speech (TTS)
    """
    # Save uploaded audio temporarily
    audio_path = f"temp_{user_id}_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # Step 1: Transcribe with language detection
    asr_result = transcribe_audio_with_detection(audio_path)
    original_text = asr_result.get("text", "")
    lang_code = asr_result.get("language_code", "unknown")

    # Step 2: Translate Indic -> English
    english_text = (
        translate_chunked_nllb_indic2en(original_text, lang_code)
        if lang_code != "en"
        else original_text
    )

    # Step 3: Pass into chatbot
    if user_id not in sessions:
        session = Session()
        session.pdf_path = "Dataset/KrishiMitra.docx"
        sessions[user_id] = session
    else:
        session = sessions[user_id]

    session.messages.append(HumanMessage(content=english_text))
    result = graph.invoke(session)
    chatbot_response_en = result.get("response", "No response")

    # Step 4: Translate chatbot response back to local language
    chatbot_response_local = (
        translate_chunked_nllb(chatbot_response_en, lang_code)
        if lang_code != "en"
        else chatbot_response_en
    )

    # Step 5: Convert local response into speech
    audio_file = text_to_speech_elevenlabs(
        chatbot_response_local, filename=f"response_{user_id}.mp3"
    )

    # Cleanup uploaded file
    os.remove(audio_path)

    return AudioChatResponse(
        original_text=original_text,
        detected_language=lang_code,
        english_text=english_text,
        chatbot_response_en=chatbot_response_en,
        chatbot_response_local=chatbot_response_local,
        audio_file=audio_file,
    )
