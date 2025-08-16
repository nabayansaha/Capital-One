from pydantic import BaseModel, Field
from typing import List, Any, Dict, Sequence, Optional
from datetime import datetime

class TokenTracker(BaseModel):
    net_input_tokens: int = Field(None, description="Number of input tokens")
    net_output_tokens: int = Field(None, description="Number of output tokens")
    net_tokens: int = Field(None, description="Number of tokens")

class QAPair(BaseModel):
    query: str
    answer1: Optional[str] = Field(None, description="Primary answer string")
    answer2: Optional[str] = Field(None, description="Secondary/comparison answer string")
    references: List[str] = Field(default_factory=list, description="List of references")


class Messages(BaseModel):
    time: datetime = Field(None, description="Timestamp of the message")
    type: str = Field(None, description="Type of the message, e.g., 'human', 'ai'")
    content: str = Field(None, description="Content of the message")
   