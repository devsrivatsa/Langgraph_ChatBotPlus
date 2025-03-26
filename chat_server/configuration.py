import os
from dataclasses import dataclass, field, fields
from typing import List, Dict, Any, Optional
from typing_extensions import Annotated
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

load_dotenv()

@dataclass(kw_only=True)
class Configuration:
    """Main configuration class for the memory graph"""
    user_id: str = "default_user"
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai:o1-mini",
        metadata={
            "api_key": os.getenv("OPENAI_API_KEY"),
            "description": "The name of the language model to use. Should be in the format: provider:model_name",
        },
    )
    embedding_model: Annotated[str, {"__template_metadata__": {"kind": "embeddings"}}] = field(
        default="openai:text-embedding-3-small",
        metadata={
            "api_key": os.getenv("OPENAI_API_KEY"),
            "description": "The name of the embedding model to use. Should be in the format: provider:model_name",
        },
    )

    @classmethod
    def from_runnable_config(cls, config:Optional[RunnableConfig]=None) -> "Configuration":
        """Create a configuration from a runnable config"""
        configurable = config["configurable"] if config and config["configurable"] else {}
        
        values: dict[str, Any] = {
            field.name: os.environ.get(field.name.upper(), configurable.get(field.name))
            for field in fields(cls)
            if field.init
        }

        return cls(**{k:v for k,v in values.items() if v is not None})
