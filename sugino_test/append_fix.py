
# -------------------------------------------------------------------------
# HOTFIX: Appended register_llm_config for AutoGen 0.4 compatibility
# -------------------------------------------------------------------------
from typing import Type

_llm_config_classes: list[Type[LLMConfigEntry]] = []

def register_llm_config(cls: Type[LLMConfigEntry]) -> Type[LLMConfigEntry]:
    if issubclass(cls, LLMConfigEntry):
        _llm_config_classes.append(cls)
        return cls
    else:
        raise TypeError(f"Expected a subclass of LLMConfigEntry, got {cls}")
