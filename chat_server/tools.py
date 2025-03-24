import uuid
import os
from typing import Annotated, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore
from chat_server import Configuration
from dotenv import load_dotenv

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

async def upsert_memory(
        content:str, 
        context:str, 
        *, 
        memoty_id:Optional[uuid.UUID]=None,
        config: Annotated[RunnableConfig, InjectedToolArg] ,
        store: BaseStore
        ):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """

    memory_id = memory_id or str(uuid.uuid4())
    user_id = Configuration.from_runnable_config(config).user_id
    namespace = ("memories", user_id)
    await store.aput(
        namespace,
        key = memory_id,
        value = {
            "content": content,
            "context": context,
        }
    )
    return f"Stored memory_id {memory_id}"




search_tool = TavilySearchResults(max_results=3, include_images=False)
