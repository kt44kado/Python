import os
from autogen_agentchat.agents import UserProxyAgent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Simple test to check if the libraries are working
def main():
    print("Testing autogen-agentchat and python-dotenv...")
    agent = UserProxyAgent(name="TestUser")
    print("AgentChat instance created successfully.")

if __name__ == "__main__":
    main()
