from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph
from langgraph.utils.config import get_store

from langmem import create_manage_memory_tool

from chat_server import configuration
from chat_server.state import State
from chat_server.tools import search_web, update_memory

from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

llm = ChatOpenAI(
    model="o1-mini",
    api_key=OPENAI_API_KEY,
)

short_term_memory = MemorySaver()
store = InMemoryStore(
    index = {
        "index": {
            "embed": embeddings,
            "dims": 1536,
        }
    }
)

async def call_model(state: State, config:RunnableConfig, *, store:BaseStore) -> dict:
    configurable = configuration.Configuration.from_runnable_config(config)
    
    #retrieve the recent memories for context
    memories = await store.asearch(
        ("memories", configurable.user_id),
        query = str([m.context for m in state.messages[-3:]]),
        limit=10,
    )

    #format memories to include it in the prompt
    formatted_memories = "\n".join([f"{m.key}: {m.value} (similarity: {m.score})" for m in memories])
    if formatted_memories:
        formatted_memories = f"""<memories>\n{formatted_memories}\n</memories>"""
    
    system_prompt = configurable.system_prompt.format(
        user_info=formatted_memories,
        time=datetime.now().isoformat(),
    )

    msg = await llm.bind_tools([search_web, update_memory]).ainvoke(
        [{"role": "system", "content":system_prompt}, *state.messages],
        {"configurable": {"model": configurable.model.split(":")[1], "provider": configurable.model.split(":")[0]}},
    )

def route_message(state: State):
    """Determine the next step based on the presence of tool calls"""
    msg = state.messages[-1]
    if msg.tool_calls:
        



graph_builder = StateGraph(state_schema=State, config_schema=configuration.Configuration)
graph_builder.add_node(call_model)
graph_builder.add_node(store_memory)
graph_builder.add_edge("__start__", "call_model")
graph_builder.add_condition_edges("call_model", route_message, [])