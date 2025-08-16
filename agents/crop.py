from agents.states import Session, CropResearchSession, MarketAgent, WeatherAgent
from typing import Dict, Any, Optional, List
from pydantic import Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agents.schemas import TokenTracker, QAPair
from utils.chat import invoke_llm_langchain
from rag.rag import RAG
import logging
import warnings
from agents.schemas import Messages
from dotenv import load_dotenv
from datetime import datetime
import os
import yaml


warnings.filterwarnings("ignore")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename="KrishiMitra.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

with open("utils/prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

def get_crop_data(state: Session) ->  Session:
    """
    Get crop data for the given state.
    """
    logger.info(f"Fetching crop data for state: {state.id}")

    query = state.messages[-1].content if state.messages else "Get crop data"
    logger.info(f"Querying RAG with: {query}")

    rag = RAG(state.pdf_path)
    index = rag.create_db()
    retriever = rag.create_retriever(index)
    rag_response = rag.rag_query(query,retriever)
    logger.info(f"RAG response: {rag_response}")

    # Step 1: Get LLM response
    messages = [
        SystemMessage(content=prompts["Crop_prompts"]["System_message"]),
        HumanMessage(content=query),
    ]
    llm_response, input_tokens, output_tokens = invoke_llm_langchain(messages)
    llm_content = llm_response[-1].content
    logger.info(f"LLM response: {llm_content}")

    # Step 2: Refine with RAG + LLM response
    refine_messages = [
        SystemMessage(content=prompts["Crop_prompts"]["System_message"]),
        HumanMessage(
            content=f"Write a proper response using RAG response: {rag_response} "
                    f"and LLM response: {llm_content}"
        )
    ]
    final_response, input_tokens2, output_tokens2 = invoke_llm_langchain(refine_messages)
    final_content = final_response[-1].content
    logger.info(f"Final response: {final_content}")

    # Update session state
    state.token_tracker.net_input_tokens = (state.token_tracker.net_input_tokens or 0) + input_tokens + input_tokens2
    state.token_tracker.net_output_tokens = (state.token_tracker.net_output_tokens or 0) + output_tokens + output_tokens2
    state.token_tracker.net_tokens = (
        (state.token_tracker.net_input_tokens or 0) +
        (state.token_tracker.net_output_tokens or 0)
    )

    state.qa_pairs[query] = QAPair(
        query=query,
        answer1=final_content,  # You may want to redefine QAPair to store string instead of bool
        answer2=None,
        references=[str(rag_response)]
    )

    # Instead of overwriting with a single message
    state.chat_history.append(
        Messages(
            type="ai",
            time=datetime.now(),
            content=final_content
        )
    )

    state.response = final_content
    logger.info(f"Updated session state: {state}")
    return state

if __name__ == "__main__":
    # Example usage
    session = Session(
    id="session_1",
    ragkey="rag_key_1",
    messages=[HumanMessage(content="What are the best crops for the summer season?")],
    token_tracker=TokenTracker(),
    qa_pairs={},
    chat_history=[]
    )

    # Add the first human message
    session.chat_history.append(
        Messages(
            type="human",
            time=datetime.now(),
            content=session.messages[-1].content
        )
    )

    session = get_crop_data(session)
    print(session.chat_history)