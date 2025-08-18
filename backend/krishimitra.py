import os
import logging
from typing import Dict, Any, List, Optional, Sequence, Annotated
import json
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from agents.schemas import TokenTracker, QAPair, Messages
from agents.market import get_market_data
from agents.crop import get_crop_data
from agents.weather import get_weather_data
from agents.policy import get_policy_data
from agents.states import Session
import re
from utils.chat import invoke_llm_langchain

logging.basicConfig(
    filename="KrishiMitra.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def route_query(state: Session) -> Dict[str, Any]:
    """
    Ask LLM which agent should handle the query.
    """
    if not state.messages:
        return {"next": "FallbackAgent"}  # no input → fallback

    system_prompt = """You are a router. 
Decide which agent should handle the latest user query.
Options: CropResearch, WeatherAgent, PolicyAgent.
If none clearly fits, return FallbackAgent. for things like loan rates and stuff do fallback. also note that CropResearch proviedes data about a single crop in a given format so use it only when the user asks about a single crop by name.
and WeatherAgent provides weather data for your current location so don't call it for questions like "What seed variety suits this unpredictable weather? use Fallback in that scenario".
Return ONLY the agent name (no explanation)."""

    routing_messages = [
        HumanMessage(content=system_prompt),
        HumanMessage(content=f"User query: {state.messages[-1].content}")
    ]

    try:
        msgs, _, _ = invoke_llm_langchain(routing_messages)
        nxt = msgs[-1].content.strip()
    except Exception as e:
        logger.exception("Routing LLM failed, using fallback")
        nxt = "FallbackAgent"

    if nxt not in {"CropResearch", "MarketAgent", "WeatherAgent", "PolicyAgent", "FallbackAgent"}:
        nxt = "FallbackAgent"

    logger.info(f"Routing to: {nxt}")
    return {"next": nxt}



def fallback_node(state: Session) -> Dict[str, Any]:
    """Fallback: directly call LLM if no other agent fits."""
    try:
        state.messages.append(
            SystemMessage(content="Answer in the indian context, be concise and clear. also don't state your knowledge cutoff date."))
        msgs, _, _ = invoke_llm_langchain(state.messages)
        out = msgs[-1].content
    except Exception as e:
        logger.exception("Error in fallback_node")
        out = f"FallbackAgent error: {e}"

    msgs = list(state.messages) + [AIMessage(content=out)]
    return {"response": out, "messages": msgs}


def crop_node(state: Session) -> Dict[str, Any]:
    """Run crop research agent and append an AI message."""
    try:
        # print("DEBUG: Session PDF path:", state.pdf_path)
        out = get_crop_data(state)
        # print("DEBUG: Crop output:", out)
    except Exception as e:
        logger.exception("Error in crop_node")
        out = f"CropResearch error: {e}"
        print(out)
    msgs = list(state.messages) + [AIMessage(content=str(out))]
    return {"response": str(out), "messages": msgs}

def market_node(state: Session) -> Dict[str, Any]:
    """Run market agent and append an AI message."""
    try:
        out = get_market_data(state)
    except Exception as e:
        logger.exception("Error in market_node")
        out = f"MarketAgent error: {e}"
    msgs = list(state.messages) + [AIMessage(content=str(out))]
    return {"response": str(out), "messages": msgs}

def weather_node(state: Session) -> Dict[str, Any]:
    """Run weather agent and append an AI message."""
    try:
        out = get_weather_data(state)
    except Exception as e:
        logger.exception("Error in weather_node")
        out = f"WeatherAgent error: {e}"
    msgs = list(state.messages) + [AIMessage(content=str(out))]
    return {"response": str(out), "messages": msgs}

def policy_node(state: Session) -> Dict[str, Any]:
    """Run policy agent and append an AI message."""
    try:
        out = get_policy_data(state)
    except Exception as e:
        logger.exception("Error in policy_node")
        out = f"PolicyAgent error: {e}"
    msgs = list(state.messages) + [AIMessage(content=str(out))]
    return {"response": str(out), "messages": msgs}

def KrishiMitra_pipeline():
    builder = StateGraph(Session)

    builder.add_node("route_query", route_query)
    builder.add_node("CropResearch", crop_node)
    builder.add_node("MarketAgent", market_node)
    builder.add_node("WeatherAgent", weather_node)
    builder.add_node("PolicyAgent", policy_node)
    builder.add_node("FallbackAgent", fallback_node)  # NEW

    builder.add_edge(START, "route_query")
    builder.add_conditional_edges(
        "route_query",
        lambda s: s.next,
        {
            "CropResearch": "CropResearch",
            "MarketAgent": "MarketAgent",
            "WeatherAgent": "WeatherAgent",
            "PolicyAgent": "PolicyAgent",
            "FallbackAgent": "FallbackAgent",
        },
    )

    builder.add_edge("CropResearch", END)
    builder.add_edge("MarketAgent", END)
    builder.add_edge("WeatherAgent", END)
    builder.add_edge("PolicyAgent", END)
    builder.add_edge("FallbackAgent", END)

    return builder.compile()


def run_chatbot():
    graph = KrishiMitra_pipeline()
    session = Session()
    session.pdf_path = "Dataset/KrishiMitra.docx"  # ensure PDF path is set

    print("🌱 Welcome to KrishiMitra! (type 'quit' to exit)\n")

    while True:
        user_input = input("👨‍🌾 You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            print("👋 Goodbye!")
            break

        # Append user input to session
        session.messages.append(HumanMessage(content=user_input))

        # Invoke graph
        result = graph.invoke(session)

        # Print AI response directly
        ai_response = result.get("response", "No response")
        ai_response_str = ai_response  # your string
        ai_response_json = json.loads(ai_response_str)

        # Get the last AI message
        last_ai = next((m for m in reversed(ai_response_json["chat_history"]) if m["type"] == "ai"), None)

        if last_ai:
            print(f"🤖 KM: {last_ai['content']}\n")

if __name__ == "__main__":
   run_chatbot()
