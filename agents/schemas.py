from pydantic import BaseModel, Field
from typing import List, Any, Dict, Sequence


class TokenTracker(BaseModel):
    net_input_tokens: int = Field(None, description="Number of input tokens")
    net_output_tokens: int = Field(None, description="Number of output tokens")
    net_tokens: int = Field(None, description="Number of tokens")

