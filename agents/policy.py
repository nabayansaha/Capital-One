import logging
from datetime import datetime
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from agents.states import Session
from agents.schemas import TokenTracker, QAPair, Messages
from utils.chat import invoke_llm_langchain
from rag.rag import RAG
import yaml

logger = logging.getLogger(__name__)

# Load prompts (optional key -> safe fallback)
with open("utils/prompts.yaml", "r") as f:
    _prompts = yaml.safe_load(f)

DEFAULT_POLICY_SYSTEM = (
    "You are an assistant for Indian agricultural policy interpretation. "
    "Use ONLY the provided RAG evidence (policy PDFs, government circulars, official guidelines). "
    "Do not invent data. Explain clearly and practically for farmers/extension officers. "
    "Cite sections/clauses from the RAG evidence where relevant."
)
POLICY_SYS = (_prompts.get("Policy_prompts", {}) or {}).get(
    "System_message", DEFAULT_POLICY_SYSTEM
)

def get_policy_data(state: Session, topic: Optional[str] = None) -> Session:
    """
    Policy Agent: RAG-only (no web search).
    Flow:
      - Read last user query (and optional topic)
      - Retrieve with RAG
      - First LLM pass (draft)
      - Refine using ONLY RAG evidence + draft
      - Update token tracker, chat history, qa_pairs, and session messages
    """
    logger.info(f"[PolicyAgent] Start | session_id={state.id}")

    # 1) Build query
    user_query = state.messages[-1].content if state.messages else "Policy query"
    query = f"{user_query}"
    if topic:
        query = f"{user_query} | topic: {topic}"
    logger.info(f"[PolicyAgent] Query: {query}")

    # 2) RAG retrieval (RAG-only)
    try:
        rag = RAG(state.pdf_path)         # If your RAG takes just path
        index = rag.create_db()
        retriever = rag.create_retriever(index)
    except TypeError:
        # Fallback in case constructor differs
        rag = RAG(pdf_path=getattr(state, "pdf_path", None))
        retriever = rag.create_retriever()
    rag_response = rag.rag_query(query, retriever)
    logger.info("[PolicyAgent] RAG response ready")

    # Log RAG evidence to chat history
    state.chat_history.append(
        Messages(type="rag", time=datetime.now(), content=str(rag_response))
    )

    # 3) First LLM pass (draft) — instructed to rely only on RAG evidence
    draft_messages = [
        SystemMessage(content=POLICY_SYS),
        HumanMessage(
            content=(
                f"User query: {user_query}\n"
                f"Topic (optional): {topic or 'N/A'}\n\n"
                "Using ONLY the following RAG evidence, draft a clear answer "
                "with section/paragraph references:\n"
                f"{rag_response}"
            )
        ),
    ]
    draft_llm, in1, out1 = invoke_llm_langchain(draft_messages)
    draft_content = draft_llm[-1].content
    logger.info("[PolicyAgent] Draft created")

    state.chat_history.append(
        Messages(type="ai_draft", time=datetime.now(), content=draft_content)
    )

    # 4) Refine strictly with RAG evidence + draft (no external data)
    refine_messages = [
        SystemMessage(content=POLICY_SYS),
        HumanMessage(
            content=(
                "Refine the answer. Keep it strictly grounded to the RAG evidence. "
                "Add numbered steps/clauses if helpful and mention relevant sections.\n\n"
                f"RAG evidence:\n{rag_response}\n\n"
                f"Draft:\n{draft_content}"
            )
        ),
    ]
    final_msgs, in2, out2 = invoke_llm_langchain(refine_messages)
    final_content = final_msgs[-1].content
    logger.info("[PolicyAgent] Final answer ready")

    # 5) Update tokens
    state.token_tracker.net_input_tokens = (state.token_tracker.net_input_tokens or 0) + in1 + in2
    state.token_tracker.net_output_tokens = (state.token_tracker.net_output_tokens or 0) + out1 + out2
    state.token_tracker.net_tokens = (state.token_tracker.net_input_tokens or 0) + (state.token_tracker.net_output_tokens or 0)

    # 6) Save Q&A (ensure QAPair.answer1/answer2 are Optional[str] in your schema)
    state.qa_pairs[query] = QAPair(
        query=query,
        answer1=final_content,
        answer2=draft_content,
        references=[str(rag_response)],
    )

    # 7) Append AI message + chat history
    state.messages.append(AIMessage(content=final_content))
    state.chat_history.append(
        Messages(type="ai", time=datetime.now(), content=final_content)
    )

    logger.info(f"[PolicyAgent] Done | session_id={state.id}")
    return state


# ----------------- Example run -----------------
if __name__ == "__main__":

    session = Session(
        id="policy_session_1",
        ragkey="rag_key_1",
        messages=[HumanMessage(content="Explain PM-Kisan eligibility and the verification steps.")],
        token_tracker=TokenTracker(),
        qa_pairs={},
        chat_history=[]
    )
    # Log the human message
    session.chat_history.append(
        Messages(type="human", time=datetime.now(), content=session.messages[-1].content)
    )

    # Optional: set session.pdf_path to your policy PDF(s) location as required by RAG
    # e.g., session.pdf_path = "data/policies/pm_kisan.pdf"
    # Make sure your RAG class supports this.

    session = get_policy_data(session, topic="PM-Kisan")
    print(session.chat_history[-1].content)
