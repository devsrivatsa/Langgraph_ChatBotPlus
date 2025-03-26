from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid

from chat_server.graph import graph
from chat_server.configuration import Configuration
from chat_server.state import State

app = FastAPI(title="Memory Chatbot API")

class Message(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    messages: List[Message]
    user_id: Optional[str] = None
    model: str = "o1-mini:openai"  # default model
    system_prompt: Optional[str] = None

class ChatResponse(BaseModel):
    messages: List[Message]
    thread_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Generate thread_id if this is a new conversation
        thread_id = str(uuid.uuid4())
        
        # Convert incoming messages to the format expected by the graph
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        # Prepare the initial state and config
        initial_state = State(messages=formatted_messages)
        
        config = {
            "configurable": {
                "thread_id": thread_id,  # For persistence across calls
                "user_id": request.user_id or str(uuid.uuid4()),
                "model": request.model,
                "system_prompt": request.system_prompt or "You are a helpful AI assistant."
            }
        }

        # Run the graph
        result = await graph.ainvoke(initial_state, config)
        
        # Format the response
        response_messages = [
            Message(
                role=msg["role"],
                content=msg["content"]
            )
            for msg in result["messages"]
        ]
        
        return ChatResponse(
            messages=response_messages,
            thread_id=thread_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
