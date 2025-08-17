import os
import base64
import json
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MOONDREAM_API_KEY")
QUERY_URL = "https://api.moondream.ai/v1/query"

def ask_vlm(image_path: str, question: str) -> str:
    if not API_KEY:
        raise ValueError("Missing MOONDREAM_API_KEY in environment")

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    payload = {
        "image_url": data_uri,
        "question": question,
        "stream": False
    }

    headers = {
        "X-Moondream-Auth": API_KEY,
        "Content-Type": "application/json"
    }

    resp = requests.post(QUERY_URL, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json().get("answer")

if __name__ == "__main__":
    path = "/Users/naba/Desktop/freelance/data/material_issue_data/issue/4.png"
    prompt = "What objects are in this image, and what are they doing?"
    try:
        ans = ask_vlm(path, prompt)
        print("Answer:", ans)
    except Exception as e:
        print("Error:", e)
