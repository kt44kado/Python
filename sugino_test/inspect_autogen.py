import sys
import os

print("Sys path:", sys.path)

try:
    import autogen
    print("Successfully imported autogen")
    print("autogen location:", autogen.__file__ if hasattr(autogen, "__file__") else "namespace")
    print("autogen attributes:", dir(autogen))
except ImportError as e:
    print("Failed to import autogen:", e)

try:
    from autogen import BaseAgent
    print("Found BaseAgent in autogen")
except ImportError:
    print("BaseAgent not in autogen")

try:
    from autogen import AssistantAgent
    print("Found AssistantAgent in autogen")
except ImportError:
    print("AssistantAgent not in autogen")

try:
    import autogen_agentchat
    print("Successfully imported autogen_agentchat")
    print("autogen_agentchat attributes:", dir(autogen_agentchat))
except ImportError as e:
    print("Failed to import autogen_agentchat:", e)
