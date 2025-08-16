from pydantic import BaseModel, Field
from typing import List, Any, Dict, Sequence
from datetime import datetime

class TokenTracker(BaseModel):
    net_input_tokens: int = Field(None, description="Number of input tokens")
    net_output_tokens: int = Field(None, description="Number of output tokens")
    net_tokens: int = Field(None, description="Number of tokens")

class QAPair(BaseModel):
    query: str = Field(None, description="Query string")
    answer1: bool = Field(None, description="Answer string")
    answer2: bool = Field(None, description="Answer string") #will be used for the comparison
    references: List[str] = Field(
        None, description="List of references related to the query and answer"
    )

class Messages(BaseModel):
    type: str = Field(None, description="Type of the message, e.g., 'human', 'ai'")
    time: datetime = Field(None, description="Timestamp of the message")
    content: str = Field(None, description="Content of the message")
   