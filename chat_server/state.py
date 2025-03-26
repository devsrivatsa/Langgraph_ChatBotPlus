from __future__ import annotations
from dataclasses import dataclass
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing import Annotated, Optional
from pydantic import BaseModel
import uuid

@dataclass(kw_only=True)
class State:
    """Main graph state"""
    messages: Annotated[list[AnyMessage], add_messages]

class Memory(BaseModel):
    content: str
    context: str
    memory_id: Optional[uuid.UUID] = None


__all__ = ["State", "Memory"]