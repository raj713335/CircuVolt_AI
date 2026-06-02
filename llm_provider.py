

"""
LLM Provider for CircuVolt AI
Supports one mode:
  2. OpenAI-compatible API (direct endpoint for open use)

Set (default) LLM_PROVIDER=openai in .env to switch.
"""
import os
import copy
import base64
import json
from typing import Any, Dict, List, Optional, Sequence, Union
from datetime import datetime, timedelta

import requests
from pydantic import ConfigDict, PrivateAttr
from dotenv import load_dotenv

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult

load_dotenv()


# ──────────────────────────────────────────────────────────────────
# Factory: get the right LLM based on LLM_PROVIDER env var
# ──────────────────────────────────────────────────────────────────

def get_llm() -> BaseChatModel:
    """
    Return the configured LLM instance.


    For OpenAI mode you can optionally set:
      OPENAI_API_KEY      – your API key
      OPENAI_API_BASE     – custom base URL (e.g. Azure, local vLLM)
      OPENAI_MODEL        – model name (default gpt-4o-mini)
    """
    provider = os.getenv("LLM_PROVIDER", "target").lower().strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: Dict[str, Any] = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0,
            "streaming": True,
        }
        # Allow custom base URL for Azure / vLLM / Ollama
        api_base = os.getenv("OPENAI_API_BASE")
        if api_base:
            kwargs["base_url"] = api_base

        return ChatOpenAI(**kwargs)

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Set LLM_PROVIDER=openai in .env.")
