from __future__ import annotations
from dataclasses import dataclass
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing import Annotated

@dataclass(kw_only=True)
class State:
    """Main graph state"""
    messages: Annotated[list[AnyMessage], add_messages]


__all__ = ["State"]