from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import Dict, Any, List
from agents.schemas import TokenTracker, QAPair, Messages


# ---------------- Session Schema (yours + routing/response) ----------------
class Session(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[str] = Field(
        default=None, description="Unique identifier for the session"
    )
    pdf_path: Optional[str] = Field(
        default=r"Dataset/KrishiMitra.docx",
        description="Path to the document associated with the session",
    )
    ragkey: Optional[str] = Field(
        default=None, description="Key for RAG session"
    )
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list, description="List of messages in the session"
    )
    token_tracker: TokenTracker = Field(
        default_factory=TokenTracker, description="Token tracker for the session"
    )
    qa_pairs: Dict[str, QAPair] = Field(
        default_factory=dict, description="Dictionary of QA pairs in the session"
    )
    chat_history: List[Messages] = Field(
        default_factory=list, description="Chat history for the session"
    )

    # 👇 Added for routing + storing the final text answer
    next: Optional[str] = Field(
        default=None, description="Next agent to route the query"
    )
    response: Optional[str] = Field(
        default=None, description="Final response text from the selected agent"
    )


class CropResearchSession(Session):
    """
    Session for crop research, extending the base session with specific fields.
    """
    crop_name: Optional[str] = Field(
        default=None, description="Name of the crop being researched"
    )
    research_topic: Optional[str] = Field(
        default=None, description="Specific topic of research within the crop domain"
    )
    additional_info: Dict[str, Any] = Field(
        default_factory=dict, description="Additional information related to the session"
    )

class MarketAgent(Session):
    """
    Session for market-related queries, extending the base session with specific fields.
    """
    market_name: Optional[str] = Field(
        default=None, description="Name of the market being queried"
    )
    query_type: Optional[str] = Field(
        default=None, description="Type of query related to the market"
    )
    market_data: Dict[str, Any] = Field(
        default_factory=dict, description="Market data relevant to the session"
    )
class WeatherAgent(Session):
    """
    Session for weather-related queries, extending the base session with specific fields.
    """
    location: Optional[str] = Field(
        default=None, description="Location for which weather data is being queried"
    )
    date_range: Optional[str] = Field(
        default=None, description="Date range for the weather query"
    )
    weather_data: Dict[str, Any] = Field(
        default_factory=dict, description="Weather data relevant to the session"
    )

class PolicyAgent(Session):
    """
    Session for policy-related queries, extending the base session with specific fields.
    """
    policy_area: Optional[str] = Field(
        default=None, description="Area of policy being queried"
    )
    policy_details: Dict[str, Any] = Field(
        default_factory=dict, description="Details of the policy relevant to the session"
    )
    related_documents: List[str] = Field(
        default_factory=list, description="List of documents related to the policy query"
    )