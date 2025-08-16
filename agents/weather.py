import logging
from datetime import datetime
from typing import Optional, Dict
import requests
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents.states import Session
from agents.schemas import TokenTracker, QAPair, Messages
from utils.location import get_user_location

logger = logging.getLogger(__name__)


def fetch_weather(lat: float, lon: float) -> Dict:
    """
    Call Open-Meteo API for current weather.
    Docs: https://open-meteo.com/
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def get_weather_data(state: Session, topic: Optional[str] = None) -> Session:
    """
    Weather Agent:
    - Gets device/IP location
    - Queries Open-Meteo
    - Synthesizes weather report
    - Updates session (qa_pairs, chat_history, messages, token tracking)
    """
    logger.info(f"[WeatherAgent] Start | session_id={state.id}")

    user_query = state.messages[-1].content if state.messages else "Weather report"
    logger.info(f"[WeatherAgent] User query: {user_query}")

    # 1) Location
    location_info = get_user_location()
    if "error" in location_info:
        final_content = f"❌ Could not determine your location: {location_info['error']}"
        state.messages.append(AIMessage(content=final_content))
        state.chat_history.append(
            Messages(type="ai", time=datetime.now(), content=final_content)
        )
        return state

    lat, lon = location_info["latitude"], location_info["longitude"]

    # 2) Fetch weather
    try:
        weather_json = fetch_weather(lat, lon)
        current = weather_json.get("current_weather", {})
        temperature = current.get("temperature")
        windspeed = current.get("windspeed")
        weather_time = current.get("time")

        place_desc = location_info.get("city") or location_info.get("region") or "your area"
        final_content = (
            f"🌤 Weather report for {place_desc}:\n"
            f"- Temperature: {temperature}°C\n"
            f"- Wind Speed: {windspeed} km/h\n"
            f"- Report Time (UTC): {weather_time}\n"
        )

    except Exception as e:
        final_content = f"❌ Failed to fetch weather data: {str(e)}"

    # 3) Update session
    # Save to QAPair (no LLM here, so answer1 = final, answer2 = None)
    state.qa_pairs[user_query] = QAPair(
        query=user_query,
        answer1=final_content,
        answer2=None,
        references=[str(location_info)],
    )

    # Append AI message
    state.messages.append(AIMessage(content=final_content))
    state.chat_history.append(
        Messages(type="ai", time=datetime.now(), content=final_content)
    )

    logger.info(f"[WeatherAgent] Done | session_id={state.id}")
    return state


# ----------------- Example run -----------------
if __name__ == "__main__":
    session = Session(
        id="weather_session_1",
        ragkey="weather_key_1",
        messages=[HumanMessage(content="What's the weather like right now?")],
        token_tracker=TokenTracker(),
        qa_pairs={},
        chat_history=[]
    )
    # Log human message
    session.chat_history.append(
        Messages(type="human", time=datetime.now(), content=session.messages[-1].content)
    )

    session = get_weather_data(session)
    print(session.chat_history[-1].content)
