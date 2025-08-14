# crop_research_agent_gemini.py
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
class CropData:
    crop_name: str
    scientific_name: str
    crop_type: str
    optimal_climate: str
    soil_requirements: str
    sowing_season: str
    harvesting_time: str
    average_yield: str
    common_diseases: list
    pest_management: list
    water_requirements: str
    fertilizer_recommendations: list
    market_price_range: str
    cultivation_tips: list
    location_specific_notes: str
    timestamp: datetime


load_dotenv()


class GeminiCropResearchAgent:
    def __init__(self, gemini_api_key: str):
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")

        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.search_tool = TavilySearchResults()

        self.system_context = """
You are an agricultural crop research assistant for Indian farmers.
You must always respond in valid JSON format only — no extra text.

Your JSON must strictly follow this schema:
{
    "crop_name": "<string>",
    "scientific_name": "<string>",
    "crop_type": "<string>",  // e.g., cereal, pulse, oilseed, vegetable, fruit
    "optimal_climate": "<string>",
    "soil_requirements": "<string>",
    "sowing_season": "<string>",
    "harvesting_time": "<string>",
    "average_yield": "<string>",
    "common_diseases": ["<string>", "..."],
    "pest_management": ["<string>", "..."],
    "water_requirements": "<string>",
    "fertilizer_recommendations": ["<string>", "..."],
    "market_price_range": "<string>",
    "cultivation_tips": ["<string>", "..."],
    "location_specific_notes": "<string>"
}
Make sure:
- Units are included where applicable (e.g., kg/ha, mm, °C)
- If data is unavailable, use "Not available"
- Do not add any commentary outside the JSON
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
            return json.loads(raw_text)
        except json.JSONDecodeError:
            try:
                json_str = self._extract_json(raw_text)
                return json.loads(json_str)
            except Exception as e:
                raise ValueError(f"Model did not return valid JSON: {e}\nRaw output:\n{raw_text}")

    def get_crop_details(self, crop: str, location: Optional[str] = None) -> Dict:
        search_query = f"{crop} cultivation details soil climate season yield pests diseases India {location or ''}"
        search_results = self.search_tool.run(search_query)[:2000]

        prompt = f"""{self.system_context}
Using the following research data, provide detailed crop information for {crop} in {location or 'India'}:
{search_results}
"""
        return self._gemini_generate(prompt)

    def compare_crops(self, crop1: str, crop2: str, location: Optional[str] = None) -> Dict:
        search_results1 = self.search_tool.run(f"{crop1} cultivation details India {location or ''}")[:1000]
        search_results2 = self.search_tool.run(f"{crop2} cultivation details India {location or ''}")[:1000]

        prompt = f"""{self.system_context}
Compare the following two crops for farmers in {location or 'India'}:
CROP 1: {crop1} Data: {search_results1}
CROP 2: {crop2} Data: {search_results2}

Return a JSON object with "crop1" and "crop2" keys following the schema above.
"""
        return self._gemini_generate(prompt)

    def process_crop_query(self, query: str) -> Dict:
        intent_prompt = f"""
Classify this agricultural query:
Query: "{query}"
Extract in format: Intent|Crop1|Crop2|Location
Possible intents: single_crop, compare_crops
"""
        intent_response = self.model.generate_content(intent_prompt).text.strip()
        parts = (intent_response + "||||").split("|")[:4]
        intent, crop1, crop2, location = [p.strip() or None for p in parts]

        if intent == "compare_crops":
            response = self.compare_crops(crop1 or "crop", crop2 or "crop", location)
        else:
            response = self.get_crop_details(crop1 or "crop", location)

        return {
            "success": True,
            "response": response,
            "agent_type": "crop_research",
            "timestamp": datetime.now().isoformat(),
            "intent": intent or "single_crop"
        }


class GeminiCropResearchService:
    def __init__(self, gemini_api_key: str):
        self.crop_agent = GeminiCropResearchAgent(gemini_api_key)

    async def process_query(self, query: str) -> Dict:
        import asyncio
        return await asyncio.to_thread(self.crop_agent.process_crop_query, query)


# Testing
if __name__ == "__main__":
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    crop_service = GeminiCropResearchService(gemini_api_key)

    test_queries = [
        "Give me complete details of wheat cultivation in Punjab",
        "Compare rice and maize for cultivation in Bihar",
        "Tell me about tomato farming in Maharashtra"
    ]

    print("🌱 Gemini Crop Research Agent Testing")
    print("=" * 50)

    for query in test_queries:
        try:
            result = crop_service.crop_agent.process_crop_query(query)
            print(f"\n🔍 Query: {query}")
            print(f"🤖 Intent: {result.get('intent', 'N/A')}")
            print(f"💬 Response:\n{json.dumps(result['response'], indent=2)}")
        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")
        print("-" * 80)
