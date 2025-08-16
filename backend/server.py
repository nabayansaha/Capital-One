from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage
from agents.states import Session
from krishimitra import KrishiMitra_pipeline

# Create FastAPI app
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
   allow_origins=["http://localhost:5173"],  # Change to ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
graph = KrishiMitra_pipeline()
sessions: Dict[str, Session] = {}  # store sessions by user_id


# Request/Response Models
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
    chat_history = [{"type": "human", "content": m.content} if m.type == "human" 
                    else {"type": "ai", "content": m.content} 
                    for m in session.messages]

    return ChatResponse(response=ai_response, chat_history=chat_history)
