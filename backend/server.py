import os
import re
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage
from agents.states import Session
from krishimitra import KrishiMitra_pipeline
from asr.asr import audio_to_english_transcript, english_to_original_language
from utils.vision import ask_vlm  # VLM function
from utils.chat import invoke_llm_langchain  # <-- refine query before graph

# ===============================
# Initialize FastAPI & Middleware
# ===============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Initialize Graph & Session Store
# ===============================
graph = KrishiMitra_pipeline()
sessions: Dict[str, Session] = {}

# ===============================
# Response Models
# ===============================
class ChatResponse(BaseModel):
    response: str
    chat_history: Optional[List[Dict[str, Any]]] = None

# ===============================
# /chat_dynamic Endpoint
# ===============================
@app.post("/chat_dynamic", response_model=ChatResponse)
async def chat_dynamic(
    user_id: str = Form(...),
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    # ----------------------
    # Get or create session
    # ----------------------
    if user_id not in sessions:
        session = Session()
        session.pdf_path = "Dataset/KrishiMitra.docx"
        sessions[user_id] = session
    else:
        session = sessions[user_id]

    original_language = "en"  # default

    # ----------------------
    # Image input -> VLM
    # ----------------------
    if file and file.content_type.startswith("image"):
        img_path = f"temp_{user_id}_{file.filename}"
        with open(img_path, "wb") as f:
            f.write(await file.read())

        fixed_prompt = "Tell me the name of the crop in the image and ,Is there any anomalies in the shown crop? format your answer as crop: <crop_name>, anomalies: <(which is the disease name)>."
        try:
            ai_response = ask_vlm(img_path, fixed_prompt)
        except Exception as e:
            ai_response = f"Error calling VLM: {e}"

        os.remove(img_path)
        return ChatResponse(response=ai_response.strip(), chat_history=[])

    # ----------------------
    # Audio input -> ASR -> English -> Graph -> Original Language
    # ----------------------
    if file and file.content_type.startswith("audio"):
        audio_path = f"temp_{user_id}_{file.filename}"
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        asr_result = audio_to_english_transcript(audio_path)
        os.remove(audio_path)

        message = asr_result["english_text"]
        detected_lang = asr_result["detected_lang"]
        if detected_lang and detected_lang != "en":
            original_language = detected_lang

    # ----------------------
    # Text input -> Refine Query -> Graph
    # ----------------------
    # ----------------------
    # Text input -> Refine Query -> Graph
    # ----------------------
    if message:
        try:
            refine_prompt = HumanMessage(
                content= message
            )

            refined_messages, in_tok, out_tok = invoke_llm_langchain([refine_prompt])

            # Get the refined query (last AI message content)
            refined_message = refined_messages[-1].content
            refined_message = message

        except Exception as e:
            refined_message = message 

        # Save refined query to session
        session.messages.append(HumanMessage(content=refined_message))


    # ----------------------
    # Run KrishiMitra Graph
    # ----------------------
    result = graph.invoke(session)
    ai_response = result.get("response", "No response")

    match = re.search(r"chat_history=\[.*?content='(.*?)'\)", ai_response, re.DOTALL)
    if match:
        ai = match.group(1).replace("\\n", "\n")
    else:
        ai = ai_response

    # Convert back to original language if needed
    if original_language != "en":
        try:
            ai = english_to_original_language(ai, original_language)
        except Exception as e:
            ai += f"\n\n(Note: Failed to translate back to {original_language}: {e})"

    # Try parsing JSON-like responses
    try:
        data = json.loads(ai)
        if isinstance(data, dict):
            formatted = []
            for k, v in data.items():
                v_str = ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
                formatted.append(f"{k.replace('_',' ').title()}: {v_str}")
            ai = "\n".join(formatted)
    except Exception:
        pass

    # Build chat history
    chat_history = [
        {"type": "human", "content": m.content} if m.type == "human"
        else {"type": "ai", "content": m.content}
        for m in session.messages
    ]

    return ChatResponse(response=ai.strip(), chat_history=chat_history)
