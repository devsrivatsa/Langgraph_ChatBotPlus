from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langmem import create_manage_memory_tool

from chat_server.configuration import Configuration
from chat_server.state import State
from chat_server.tools import web_search_tool, upsert_memory, get_manage_memory_tool, get_search_memory_tool
from chat_server.prompts import SYSTEM_PROMPT

from dotenv import load_dotenv
from datetime import datetime
import asyncio
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)
print("hello world",embeddings)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
)

short_term_memory = InMemorySaver()
store = InMemoryStore(
    index={
        "embed": embeddings, 
        "dims": 1536,
    }
)

async def call_model(state: State, config:RunnableConfig, *, store:BaseStore) -> dict:
    configurable = Configuration.from_runnable_config(config)

    #create memory tools with this user's namespace
    manage_memory_tool = get_manage_memory_tool(namespace=("memories", configurable.user_id))
    search_memory_tool = get_search_memory_tool(namespace=("memories", configurable.user_id))
    
    #retrieve the recent memories for context
    memories = await store.asearch(
        ("memories", configurable.user_id),
        query = str([m.content for m in state.messages[-3:]]),
        limit=10,
    )
    #TODO: the memory structure has been changed to a pydantic BaseModel. it is no longer a k:v pair. Hence have to change this line for the call model to work
    #format memories to include it in the prompt
    formatted_memories = "\n".join([f"{m.key}: {m.value} (similarity: {m.score})" for m in memories])
    if formatted_memories:
        formatted_memories = f"""<memories>\n{formatted_memories}\n</memories>"""
    
    print("maaaaooooo:",state.messages[0])

    msg = await llm.bind_tools([
        # web_search_tool, 
        upsert_memory,
        manage_memory_tool,
        search_memory_tool
    ]).ainvoke(
        [{"role": "system", "content":SYSTEM_PROMPT}, *state.messages],
        {"configurable": {
            "model": configurable.model.split(":")[1], 
            "provider": configurable.model.split(":")[0],
            "user_id": configurable.user_id
        }},
    )

    return {"messages": [msg]}


async def store_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    #get the tool calls from the last message
    tool_calls = state.messages[-1].tool_calls
    formatted_result = []
    
    for tc in tool_calls:
        #1. Manual search for initial context - to provide immediate context in the system prompt
        if tc["name"] == "upsert_memory":
            mem = await upsert_memory(**tc["args"], config=config, store=store)
            formatted_result.append({"role":"tool", "content":mem, "tool_call_id": tc["id"]})
        
        #2. use manage_memory_tool for dynamic retrieval: Keep the search_memory_tool available so the LLM can look up additional memories when needed
            # For LangMem's manage_memory tool, the tool itself handles storage
            # We just need to format the response
            # The args will contain action and memory data according to the schema
        elif tc["name"] == "manage_memory":
            # 1. Extract action type: create, delete, update are the only valid actions for this tool. We want the default to be create
            action = tc["args"].get("action", "create") 
            
            # 2. Create a manage_memory_tool instance for this specific user
            manage_memory_tool = get_manage_memory_tool(
                namespace=(
                    "memories", 
                    Configuration.from_runnable_config(config).user_id
                )
            )

            # 3. Execute the tool with the provided arguments
            result = await manage_memory_tool.ainvoke(tc["args"])

            #4. Format the response

            formatted_result.append({
                "role":"tool", 
                "content": f"Memory {action}d: {result}", 
                "tool_call_id": tc["id"]
            })
    
    return {"messages": formatted_result}



async def retrieve_memory(state:State, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    tool_call = state.messages[-1].tool_calls[-1]
    tool_args = tool_call["args"]
    search_memory_tool = get_search_memory_tool(namespace=("memories", configurable.user_id))
    memories, _ = await search_memory_tool.ainvoke(tool_args)
    print("mymemories", memories)
    tool_msg = ToolMessage(
        content = str(memories["value"]),
        tool_call_id = tool_call["id"],
        name = tool_call["name"]
    )
    
    return {"messages": [tool_msg]}
    

web_search = ToolNode(
    tools=[web_search_tool],
    name="web_search_tool",
)


def route_message(state: State):
    """Determine the next step based on the presence of tool calls."""
    tool_calls = state.messages[-1].tool_calls
    if tool_calls:
        print("maao_tool_called", tool_calls)
        tool_called = tool_calls[-1]["name"]
        if tool_called in ["manage_memory", "upsert_memory"]:
            return "store_memory"
        if tool_called == "search_memory":
            return "retrieve_memory"    
    # Otherwise, finish; user can send the next message
    return END

graph_builder = StateGraph(state_schema=State, config_schema=Configuration)
graph_builder.add_node(call_model)
graph_builder.add_node(store_memory)
graph_builder.add_node(retrieve_memory)
# graph_builder.add_node(web_search)
graph_builder.add_edge("__start__", "call_model")
graph_builder.add_conditional_edges("call_model", route_message, ["store_memory", "retrieve_memory", "__end__"])
graph_builder.add_edge("store_memory", "call_model")
graph_builder.add_edge("retrieve_memory", "call_model")

graph = graph_builder.compile(store=store, checkpointer=short_term_memory)
graph.name = "memory_chatbot_plus"

__all__ = ["graph"]

# if __name__ == "__main__":
#     for t in [        upsert_memory,
#         manage_memory_tool,
#         search_memory_tool]