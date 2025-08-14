# market_analysis_agent_gemini.py

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
class MarketData:
    crop_name: str
    location: str
    current_price: str
    price_unit: str
    price_trend: str
    demand_trend: str
    supply_trend: str
    recent_news: list
    government_policies: list
    risk_factors: list
    opportunities: list
    recommendations: list
    forecast_price_range: str

load_dotenv()

class GeminiMarketAnalysisAgent:
    def __init__(self, gemini_api_key: str):
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        # Setup search tool per LangChain docs
        self.search_tool = TavilySearchResults(
            max_results=10,
            include_answer=True,
            include_raw_content=True
        )  # :contentReference[oaicite:0]{index=0}

        self.system_context = """
You are an agricultural market analysis assistant for Indian farmers.
You must ONLY use India-specific data (e.g. mandi price, Agmarknet, India agriculture).
Discard data from other countries unless explicitly asked.
Always output valid JSON ONLY, respecting this schema:

{
  "crop_name": "<string>",
  "location": "<string>",
  "current_price": "<string>",
  "price_unit": "<string>",
  "price_trend": "<string>",
  "demand_trend": "<string>",
  "supply_trend": "<string>",
  "recent_news": ["<string>", "..."],
  "government_policies": ["<string>", "..."],
  "risk_factors": ["<string>", "..."],
  "opportunities": ["<string>", "..."],
  "recommendations": ["<string>", "..."],
  "forecast_price_range": "<string>"
}

Use 'Not available' if suitable data isn’t found.
"""

    def _extract_json(self, text: str) -> str:
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
            json_str = self._extract_json(raw_text)
            return json.loads(json_str)

    def get_market_analysis(self, crop: str, location: Optional[str] = None) -> Dict:
        loc = location or "India"
        query = (
            f"{crop} mandi price trend {loc} agricultural news Agmarknet India"
        )
        search_results = self.search_tool.run({"query": query})  # :contentReference[oaicite:1]{index=1}
        prompt = f"""{self.system_context}
Analyze market conditions for {crop} in {loc} using ONLY India-specific info:
{search_results}
"""
        return self._gemini_generate(prompt)

    def compare_market(self, crop1: str, crop2: str, location: Optional[str] = None) -> Dict:
        loc = location or "India"
        res1 = self.search_tool.run({"query": f"{crop1} mandi price trend {loc} Agmarknet India"})
        res2 = self.search_tool.run({"query": f"{crop2} mandi price trend {loc} Agmarknet India"})
        prompt = f"""{self.system_context}
Compare market conditions for {crop1} and {crop2} in {loc}. Use only India-specific data:
CROP1 DATA: {res1}
CROP2 DATA: {res2}
"""
        return self._gemini_generate(prompt)

    def process_market_query(self, query: str) -> Dict:
        intent_prompt = f"""
Classify this market query:
Query: "{query}"
Extract: Intent|Crop1|Crop2|Location
Possible intents: single_market, compare_market
"""
        resp = self.model.generate_content(intent_prompt).text.strip()
        intent, c1, c2, loc = (resp + "||||").split("|")[:4]
        intent, crop1, crop2, location = [x.strip() or None for x in (intent, c1, c2, loc)]
        if intent == "compare_market":
            analysis = self.compare_market(crop1 or "crop", crop2 or "crop", location)
        else:
            analysis = self.get_market_analysis(crop1 or "crop", location)
        return {
            "success": True,
            "response": analysis,
            "agent_type": "market_analysis",
            "timestamp": datetime.now().isoformat(),
            "intent": intent or "single_market"
        }

class GeminiMarketAnalysisService:
    def __init__(self, gemini_api_key: str):
        self.agent = GeminiMarketAnalysisAgent(gemini_api_key)

    async def process_query(self, query: str) -> Dict:
        import asyncio
        return await asyncio.to_thread(self.agent.process_market_query, query)

# Testing
if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    svc = GeminiMarketAnalysisService(api_key)
    for q in [
        "Onion market trends in Maharashtra",
        "Compare wheat and rice markets in Punjab"
    ]:
        result = svc.agent.process_market_query(q)
        print(json.dumps(result, indent=2))
