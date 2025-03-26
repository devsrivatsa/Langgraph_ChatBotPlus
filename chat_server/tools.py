import uuid
import os
from typing import Annotated, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from chat_server.configuration import Configuration
from chat_server.prompts import STORE_MEMORY_INSTRUCTIONS
from chat_server.state import Memory
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
web_search_tool = TavilySearchResults(max_results=3, include_images=False)



async def upsert_memory(
        content:str, 
        context:str, 
        *, 
        memory_id:Optional[uuid.UUID]=None,
        config: Annotated[RunnableConfig, InjectedToolArg] ,
        store: Annotated[BaseStore, InjectedToolArg]
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
        value = Memory(content, context, memory_id)
    )
    return f"Stored memory_id {memory_id}"





# Factory functions for memory tools
def get_manage_memory_tool(namespace:tuple):
    """Returns a configured memory management tool.
    
    This tool allows the agent to create, update, and delete memories.
    The tool will be used with the same content/context structure as the
    existing upsert_memory function.
    """
    # Default namespace pattern that will be filled at runtime
    return create_manage_memory_tool(
        namespace=namespace,
        instructions=STORE_MEMORY_INSTRUCTIONS,
        schema=Memory
    )

def get_search_memory_tool(namespace:tuple):
    """Returns a configured memory search tool.
    
    This tool allows the agent to search for relevant memories.
    """
    # Default namespace pattern that will be filled at runtime
    return create_search_memory_tool(
        namespace=namespace,
    )