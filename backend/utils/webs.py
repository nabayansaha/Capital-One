from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

class TavilySearchTool:
    def __init__(
        self,
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_images=True,
    ):
        self.tool = TavilySearchResults(
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )

    def invoke_tool(
        self, query, tool_id="1", tool_name="tavily", tool_type="tool_call"
    ):
        model_generated_tool_call = {
            "args": {"query": query},
            "id": tool_id,
            "name": tool_name,
            "type": tool_type,
        }
        return self.tool.invoke(model_generated_tool_call)
