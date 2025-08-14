# weather_agent_gemini.py
import os
import re
import json
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.tools import TavilySearchResults


@dataclass
class WeatherData:
    location: str
    temperature: str
    humidity: str
    rainfall: str
    wind_speed: str
    pressure: str
    forecast_days: int
    timestamp: datetime


load_dotenv()


class GeminiWeatherAgent:
    def __init__(self, gemini_api_key: str):
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")

        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.search_tool = TavilySearchResults()

        self.system_context = """
You are an agricultural weather advisor for Indian farmers.
You must always respond in valid JSON format only — no extra text.

Your JSON must have the following fields:
{
    "location": "<string>",
    "date": "<YYYY-MM-DD>",
    "weather": {
        "temperature_c": <float>,
        "humidity_percent": <int>,
        "rainfall_mm": <float>,
        "wind_speed_kmph": <float>,
        "pressure_hpa": <int>
    },
    "summary": "<string>",
    "agriculture": {
        "implications": "<string>",
        "recommendations": ["<string>", "..."],
        "risk_level": "<Low|Medium|High>",
        "timing": ["<string>", "..."]
    }
}
# Additional optional objects: forecast, crop_impact, irrigation, alerts
"""

    def _extract_json(self, text: str) -> str:
        """Extract JSON from the model output safely."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output")
        return match.group(0)

    def _gemini_generate(self, prompt: str) -> Dict[str, Any]:
        response = self.model.generate_content(prompt)
        raw_text = response.text.strip()

        try:
            # First try direct parsing
            return json.loads(raw_text)
        except json.JSONDecodeError:
            try:
                # Try extracting JSON part
                json_str = self._extract_json(raw_text)
                return json.loads(json_str)
            except Exception as e:
                raise ValueError(f"Model did not return valid JSON: {e}\nRaw output:\n{raw_text}")

    def get_current_weather(self, location: str) -> Dict:
        search_query = f"current weather {location} India temperature humidity rainfall wind pressure"
        search_results = self.search_tool.run(search_query)[:2000]

        prompt = f"""{self.system_context}
Using the following live search data for {location}, produce a JSON weather report for farmers:
{search_results}
"""
        return self._gemini_generate(prompt)

    def get_weather_forecast(self, location: str, days: int) -> Dict:
        search_query = f"weather forecast {location} India next {days} days temperature rain humidity"
        search_results = self.search_tool.run(search_query)[:2000]

        prompt = f"""{self.system_context}
Using this {days}-day forecast for {location}, produce a JSON weather forecast with recommendations:
{search_results}
"""
        return self._gemini_generate(prompt)

    def analyze_crop_weather_impact(self, crop: str, location: str, concern: str) -> Dict:
        weather_results = self.search_tool.run(f"current weather {location} India {concern}")[:2000]
        crop_results = self.search_tool.run(f"{crop} cultivation weather requirements {concern} India {location}")[:2000]

        prompt = f"""{self.system_context}
Analyze the current weather impact on {crop} in {location}, focusing on {concern}:
Weather Data: {weather_results}
Crop Data: {crop_results}
"""
        return self._gemini_generate(prompt)

    def calculate_irrigation_needs(self, crop: str, location: str) -> Dict:
        search_results = self.search_tool.run(f"{crop} irrigation requirements {location} current weather soil moisture")[:2000]

        prompt = f"""{self.system_context}
Calculate irrigation needs for {crop} in {location}:
Data: {search_results}
"""
        return self._gemini_generate(prompt)

    def get_agricultural_alerts(self, location: str) -> Dict:
        search_results = self.search_tool.run(f"weather alert warning {location} India agriculture farming")[:2000]

        prompt = f"""{self.system_context}
Create JSON agricultural alerts for {location}:
Data: {search_results}
"""
        return self._gemini_generate(prompt)

    def process_weather_query(self, query: str, user_context: Dict = None) -> Dict:
        if user_context:
            if 'location' in user_context:
                query += f" in {user_context['location']}"
            if 'crop_type' in user_context:
                query += f" for {user_context['crop_type']} cultivation"

        intent_prompt = f"""
Classify this agricultural query:
Query: "{query}"
Extract in format: Intent|Location|Crop|Days|Concern
"""
        intent_response = self.model.generate_content(intent_prompt).text.strip()
        parts = (intent_response + "|||||").split("|")[:5]
        intent, location, crop, days, concern = [p.strip() or None for p in parts]

        if intent == "forecast":
            days = int(days) if days and days.isdigit() else 7
            location = location or "India"
            response = self.get_weather_forecast(location, days)

        elif intent == "crop_impact":
            response = self.analyze_crop_weather_impact(crop or "crop", location or "India", concern or "weather")

        elif intent == "irrigation":
            response = self.calculate_irrigation_needs(crop or "crop", location or "India")

        elif intent == "alerts":
            response = self.get_agricultural_alerts(location or "India")

        else:
            response = self.get_current_weather(location or "India")

        return {
            "success": True,
            "response": response,
            "agent_type": "weather",
            "timestamp": datetime.now().isoformat(),
            "intent": intent or "unknown"
        }


class GeminiWeatherService:
    def __init__(self, gemini_api_key: str):
        self.weather_agent = GeminiWeatherAgent(gemini_api_key)

    async def process_query(self, query: str, user_context: Dict = None) -> Dict:
        import asyncio
        return await asyncio.to_thread(self.weather_agent.process_weather_query, query, user_context)


# Testing
if __name__ == "__main__":
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    weather_service = GeminiWeatherService(gemini_api_key)

    test_queries = [
        "What's the weather in Punjab for wheat farming?",
        "Should I irrigate my rice fields in West Bengal this week?",
        "7-day forecast for cotton farming in Gujarat",
        "Weather alerts for tomato cultivation in Maharashtra"
    ]

    print("🌾 Gemini Weather Agent Testing")
    print("=" * 50)

    for query in test_queries:
        try:
            result = weather_service.weather_agent.process_weather_query(query)
            print(f"\n🔍 Query: {query}")
            print(f"🤖 Intent: {result.get('intent', 'N/A')}")
            print(f"💬 Response:\n{json.dumps(result['response'], indent=2)}")
        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")
        print("-" * 80)
