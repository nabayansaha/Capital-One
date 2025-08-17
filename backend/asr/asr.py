import os
import time
import requests
from typing import Optional, Dict, Any
import assemblyai as aai
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from typing import Dict, Any
from typing import Optional

# =========================
# Config & Constants
# =========================
load_dotenv()

ASSEMBLY_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not ASSEMBLY_KEY:
    raise ValueError("Missing ASSEMBLYAI_API_KEY in .env")

aai.settings.api_key = ASSEMBLY_KEY
transcriber = aai.Transcriber()

FLORES_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ne": "npi_Deva",
    "ur": "urd_Arab",
}

# Hardcoded NLLB endpoint (do not read from .env)
NLLB_BASE = "https://winstxnhdw-nllb-api.hf.space"
NLLB_TRANSLATE = f"{NLLB_BASE}/api/v4/translator"


# =========================
# Step 1: Transcribe + Detect Language
# =========================
def transcribe_audio_with_detection(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe 'audio_path' and detect language via AssemblyAI.
    Returns: dict with keys: text, lang_code, transcript (raw object).
    """
    config = aai.TranscriptionConfig(language_detection=True)
    transcript = transcriber.transcribe(audio_path, config)
    if transcript.error:
        raise RuntimeError(f"Transcription error: {transcript.error}")

    text = (transcript.text or "").strip()
    lang_code = transcript.json_response.get("language_code", "unknown")
    return {"text": text, "lang_code": lang_code, "transcript": transcript}


# =========================
# Step 2: Indic -> English (NLLB)
# =========================
def translate_text_nllb_api_indic2en(indic_text: str, source_lang_code: str,
                                     retries: int = 2, timeout: int = 60) -> str:
    """
    Translate Indic -> English using HF Space NLLB API (GET).
    """
    source = FLORES_MAP.get(source_lang_code, source_lang_code)
    params = {
        "text": indic_text,
        "source": source,
        "target": FLORES_MAP["en"],  # always English target
    }
    for attempt in range(retries + 1):
        try:
            r = requests.get(NLLB_TRANSLATE, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            out = data.get("text") or data.get("result") or ""
            return (out or "").strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            print(f"[ERROR] NLLB API (Indic->EN) failed: {e}")
            return ""


def translate_chunked_nllb_indic2en(indic_text: str, source_lang_code: str,
                                    max_chars: int = 900) -> str:
    """
    Split long Indic text and translate to English in chunks.
    """
    text = (indic_text or "").strip()
    if len(text) <= max_chars:
        return translate_text_nllb_api_indic2en(text, source_lang_code)

    parts, buf, cur = [], [], 0
    for sent in text.split(". "):
        add = (sent + ". ").strip()
        if cur + len(add) > max_chars and buf:
            parts.append(" ".join(buf).strip())
            buf = [add]
            cur = len(add)
        else:
            buf.append(add)
            cur += len(add)
    if buf:
        parts.append(" ".join(buf).strip())

    out = []
    for p in parts:
        out.append(translate_text_nllb_api_indic2en(p, source_lang_code))
    return " ".join(out).strip()


# =========================
# Step 3: English -> Indic (NLLB)
# =========================
def translate_text_nllb_api(english_text: str, target_lang_code: str,
                            retries: int = 2, timeout: int = 60) -> str:
    """
    Translate English -> target_lang_code using HF Space NLLB API.
    """
    target = FLORES_MAP.get(target_lang_code, target_lang_code)
    params = {
        "text": english_text,
        "source": FLORES_MAP["en"],
        "target": target
    }
    for attempt in range(retries + 1):
        try:
            r = requests.get(NLLB_TRANSLATE, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            out = data.get("text") or data.get("result") or ""
            return (out or "").strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            print(f"[ERROR] NLLB API (EN->Indic) failed: {e}")
            return ""


def translate_chunked_nllb(english_text: str, target_lang_code: str,
                           max_chars: int = 900) -> str:
    """
    Very simple char-based chunker; split on periods if long (English -> Indic).
    """
    text = (english_text or "").strip()
    if len(text) <= max_chars:
        return translate_text_nllb_api(text, target_lang_code)

    parts, buf, cur = [], [], 0
    for sent in text.split(". "):
        add = (sent + ". ").strip()
        if cur + len(add) > max_chars and buf:
            parts.append(" ".join(buf).strip())
            buf = [add]
            cur = len(add)
        else:
            buf.append(add)
            cur += len(add)
    if buf:
        parts.append(" ".join(buf).strip())

    out = []
    for p in parts:
        out.append(translate_text_nllb_api(p, target_lang_code))
    return " ".join(out).strip()


# =========================
# Step 3: TTS (ElevenLabs streaming)
# =========================
def text_to_speech_elevenlabs(
    text: str,
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
    model_id: str = "eleven_multilingual_v2",
    filename: str = "output.mp3",
) -> Optional[str]:
    """
    Stream TTS from ElevenLabs and save to MP3. Returns local filename or None on error.
    """
    try:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ELEVENLABS_API_KEY environment variable.")

        eleven = ElevenLabs(api_key=api_key)
        audio_stream = eleven.text_to_speech.stream(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
        )
        with open(filename, "wb") as f:
            for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    f.write(chunk)
        return filename
    except Exception as e:
        print(f"[ERROR] ElevenLabs TTS failed: {e}")
        return None

def audio_to_english_transcript(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe an audio file and return English transcript.
    
    Returns dict:
        {
            "original_text": str,
            "detected_lang": str,
            "english_text": str
        }
    """
    # Step 1: Transcribe + detect language
    result = transcribe_audio_with_detection(audio_path)
    original_text = result["text"]
    lang_code = result["lang_code"]

    # Step 2: Translate if necessary
    if not original_text:
        english_text = ""
    elif lang_code in ["en", "unknown", None, ""]:
        # Either English or undetected: assume English
        english_text = original_text
    else:
        # Translate from Indic (or any detected language) -> English
        english_text = translate_chunked_nllb_indic2en(original_text, lang_code)

    return {
        "original_text": original_text,
        "detected_lang": lang_code,
        "english_text": english_text
    }

def english_to_original_language(english_text: str, target_lang_code: Optional[str]) -> str:
    """
    Translate English text back to the original language.
    
    If target_lang_code is None, unknown, or 'en', returns the text as-is.
    """
    if not english_text:
        return ""
    
    if not target_lang_code or target_lang_code in ["en", "unknown"]:
        return english_text
    
    translated_text = translate_chunked_nllb(english_text, target_lang_code)
    return translated_text

def text_to_english(text: str, lang_code: str) -> Dict[str, str]:
    """
    Convert text from a different language to English.
    
    Args:
        text (str): The input text in its original language.
        lang_code (str): Language code of the input text (e.g., 'hi', 'bn', 'ta', 'en').

    Returns:
        dict: {
            "original_text": str,
            "detected_lang": str,
            "english_text": str
        }
    """
    if not text:
        return {"original_text": "", "detected_lang": lang_code or "unknown", "english_text": ""}

    if lang_code in ["en", "unknown", None, ""]:
        english_text = text
    else:
        english_text = translate_chunked_nllb_indic2en(text, lang_code)

    return {
        "original_text": text,
        "detected_lang": lang_code or "en",
        "english_text": english_text
    }


if __name__ == "__main__":
    # Example usage
    audio_path = "/Users/naba/Desktop/capital-one/backend/temp_user1_voice.wav"
    result = audio_to_english_transcript(audio_path)
    print("Original Text:", result["original_text"])
    print("Detected Language:", result["detected_lang"])
    print("English Text:", result["english_text"])

    # Translate back to original language if needed
    translated_text = english_to_original_language(result["english_text"], result["detected_lang"])
    print("Translated Back to Original Language:", translated_text)