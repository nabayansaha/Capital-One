import os
import logging
from typing import Dict, Any, List, Optional, Sequence, Annotated
import json
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from agents.schemas import TokenTracker, QAPair, Messages
from agents.market import get_market_data
from agents.crop import get_crop_data
from agents.weather import get_weather_data
from agents.policy import get_policy_data
from agents.states import Session
import re

logging.basicConfig(
    filename="KrishiMitra.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def route_query(state: Session) -> Dict[str, Any]:
    """
    Decide which agent should handle the query.
    Returns a partial state update with 'next' set.
    """
    if not state.messages:
        return {"next": "PolicyAgent"}  # safe fallback

    last = state.messages[-1].content.lower()

    if any(w in last for w in ["crop", "fertilizer", "yield", "paddy", "wheat", "sowing", "harvest"]):
        nxt = "CropResearch"
    elif any(w in last for w in ["price", "market", "sell", "msp", "buy", "mandi", "rates"]):
        nxt = "MarketAgent"
    elif any(w in last for w in ["weather", "rain", "temperature", "forecast", "humidity", "wind"]):
        nxt = "WeatherAgent"
    elif any(w in last for w in ["policy", "scheme", "subsidy", "government", "pm-kisan", "insurance"]):
        nxt = "PolicyAgent"
    else:
        nxt = "PolicyAgent"

    logger.info(f"Routing to: {nxt}")
    return {"next": nxt}

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

    # Edges
    builder.add_edge(START, "route_query")
    builder.add_conditional_edges(
        "route_query",
        lambda s: s.next,  # <- reads 'next' from Session (not subscript)
        {
            "CropResearch": "CropResearch",
            "MarketAgent": "MarketAgent",
            "WeatherAgent": "WeatherAgent",
            "PolicyAgent": "PolicyAgent",
        },
    )

    builder.add_edge("CropResearch", END)
    builder.add_edge("MarketAgent", END)
    builder.add_edge("WeatherAgent", END)
    builder.add_edge("PolicyAgent", END)

    compiled_graph = builder.compile()

    # Save graph images to assets/
    try:
        output_dir = "assets"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "KM_graph.png")
        graph_image = compiled_graph.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
            wrap_label_n_words=9,
            output_file_path=None,
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color="white",
            padding=10,
        )
        with open(output_path, "wb") as f:
            f.write(graph_image)
        logger.info(f"Graph visualization saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not save graph visualization: {e}")

    return compiled_graph

# def run_chatbot():
#     graph = KrishiMitra_pipeline()
#     session = Session()   # persistent memory

#     print("🌱 Welcome to KrishiMitra! (type 'quit' to exit)\n")

#     while True:
#         user_input = input("👨‍🌾 You: ")
#         if user_input.strip().lower() in {"quit", "exit"}:
#             print("👋 Goodbye!")
#             break

#         # Append user input to session
#         session.messages.append(HumanMessage(content=user_input))

#         # Invoke graph WITHOUT overwriting session
#         result = graph.invoke(session)

#         # Extract AI response
#         str_ans = result.get("response", "No response")

#         # Optional: print only AIMessage content using regex
#         ai_contents = re.findall(r"AIMessage\(content='(.*?)', additional_kwargs=", str(str_ans), re.DOTALL)
#         for content in ai_contents:
#             print(content.encode().decode('unicode_escape'))

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
