import logging
import json
from datetime import datetime
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from agents.states import Session
from agents.schemas import TokenTracker, QAPair, Messages
from utils.chat import invoke_llm_langchain
from rag.rag import RAG
from utils.webs import TavilySearchTool   # <- put your TavilySearchTool in utils/tavily.py
import yaml

logging.basicConfig(
    filename="KrishiMitra.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

with open("utils/prompts.yaml", "r") as f:
    _prompts = yaml.safe_load(f)

DEFAULT_MARKET_SYSTEM = (
    "You are an Indian agricultural market analysis assistant. "
    "Prefer India-specific sources (Agmarknet, mandi data, govt. advisories). "
    "Be concise, practical, and farmer-friendly."
)
MARKET_SYS = (_prompts.get("Market_prompts", {}) or {}).get(
    "System_message", DEFAULT_MARKET_SYSTEM
)

# Tavily setup
tavily_tool = TavilySearchTool(max_results=8, search_depth="advanced")

def get_market_data(state: Session, location: Optional[str] = None) -> Session:
    """
    Market agent with RAG + TavilySearch + LLM refine.
    Updates session: messages, chat_history, qa_pairs, token_tracker.
    """
    logger.info(f"[MarketAgent] Start | session_id={state.id}")

    # 1) Pull query
    query = state.messages[-1].content if state.messages else "Market analysis"
    if location:
        query = f"{query} | location: {location}"
    logger.info(f"[MarketAgent] Query: {query}")

    # 2) Tavily search
    tavily_results = tavily_tool.invoke_tool(query)
    logger.info(f"{tavily_results} results from Tavily")
    # After calling tavily
    if isinstance(tavily_results, list):
        safe_results = [r.dict() if hasattr(r, "dict") else str(r) for r in tavily_results]
    else:
        safe_results = tavily_results.dict() if hasattr(tavily_results, "dict") else str(tavily_results)

    tavily_json = json.dumps(safe_results, indent=2)
    logger.info("[MarketAgent] Tavily results fetched")

    # log Tavily step
    state.chat_history.append(
        Messages(type="search", time=datetime.now(), content=tavily_json)
    )

    # 3) RAG retrieval
    try:
        rag = RAG(state.pdf_path)
        index = rag.create_db()
        retriever = rag.create_retriever(index)
    except TypeError:
        rag = RAG(pdf_path=getattr(state, "pdf_path", None))
        retriever = rag.create_retriever()
    rag_response = rag.rag_query(query, retriever)
    logger.info("[MarketAgent] RAG response ready")

    state.chat_history.append(
        Messages(type="rag", time=datetime.now(), content=str(rag_response))
    )

    # 4) First LLM pass
    messages = [
        SystemMessage(content=MARKET_SYS),
        HumanMessage(content=f"{query}\n\nTavily results:\n{tavily_json}"),
    ]
    llm_response, in1, out1 = invoke_llm_langchain(messages)
    llm_content = llm_response[-1].content
    logger.info("[MarketAgent] LLM draft complete")

    state.chat_history.append(
        Messages(type="ai_draft", time=datetime.now(), content=llm_content)
    )

    # 5) Refine with RAG + Tavily + LLM outputs
    refine_messages = [
        SystemMessage(content=MARKET_SYS),
        HumanMessage(
            content=(
                f"Combine the evidence and produce a clear, India-specific market analysis. most recent data given {datetime.now()} is the datetime today\n"
                f"Tavily search:\n{tavily_json}\n\n"
                # f"RAG evidence:\n{rag_response}\n\n"
                # f"LLM draft:\n{llm_content}"
            )
        ),
    ]
    final_msgs, in2, out2 = invoke_llm_langchain(refine_messages)
    final_content = final_msgs[-1].content
    logger.info("[MarketAgent] Final refined answer ready")

    # 6) Update tokens
    state.token_tracker.net_input_tokens = (state.token_tracker.net_input_tokens or 0) + in1 + in2
    state.token_tracker.net_output_tokens = (state.token_tracker.net_output_tokens or 0) + out1 + out2
    state.token_tracker.net_tokens = (
        (state.token_tracker.net_input_tokens or 0) +
        (state.token_tracker.net_output_tokens or 0)
    )

    # 7) Save Q&A
    state.qa_pairs[query] = QAPair(
        query=query,
        answer1=final_content,
        answer2=llm_content,
        references=[str(rag_response), tavily_json],
    )

    # 8) Append final AI message
    state.messages.append(AIMessage(content=final_content))
    state.chat_history.append(
        Messages(type="ai", time=datetime.now(), content=final_content)
    )

    logger.info(f"[MarketAgent] Done | session_id={state.id}")
    return state


# Example run
if __name__ == "__main__":

    session = Session(
        id="market_session_1",
        ragkey="rag_key_1",
        messages=[HumanMessage(content="Onion market trends in Maharashtra")],
        token_tracker=TokenTracker(),
        qa_pairs={},
        chat_history=[]
    )
    session.chat_history.append(
        Messages(type="human", time=datetime.now(), content=session.messages[-1].content)
    )
    session = get_market_data(session, location="Maharashtra")
    print(session.chat_history[-1].content)
