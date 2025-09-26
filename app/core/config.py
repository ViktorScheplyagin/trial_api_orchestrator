"""Application configuration loading utilities."""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR.parent / "config" / "providers.yaml"


class ProviderModel(BaseModel):
    id: str
    name: str
    priority: int = Field(default=100)
    base_url: str
    chat_completions_path: str
    availability: Dict[str, Any] = Field(default_factory=dict)
    credentials: Dict[str, Any] = Field(default_factory=dict)
    models: Dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    providers: List[ProviderModel]


@lru_cache(maxsize=1)
def load_config(path: pathlib.Path | None = None) -> AppConfig:
    """Load provider configuration from YAML."""
    config_path = path or DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text())
    return AppConfig(**raw)
