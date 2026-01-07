import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())

from agents import create_groupchat, DR_SUM_AGENTS_DEF

def verify_mcp_tools():
    print("Verifying MCP tool registration...")
    
    # Create a group chat with a subset of agents
    selected_agents = ["VWSDT0100", "Use"] # typo in User? No, get_agent_names filters out User. 
    # Let's use a known agent ID from definitions
    agent_id = DR_SUM_AGENTS_DEF[0]["id"] # e.g. VWSDT0100
    
    # create_groupchat expects list of agent NAMES (or IDs acting as names)
    # agents.py: get_dr_sum_agent_full_name use Name(ID) format usually
    # But let's check how create_groupchat works. It takes selected_agents list.
    # It creates keys in agent_map. 
    # DR_SUM_AGENT_MAP keys are the full names.
    
    from agents import DR_SUM_AGENT_MAP
    agent_name = list(DR_SUM_AGENT_MAP.keys())[0]
    print(f"Testing with agent: {agent_name}")
    
    group_chat = create_groupchat([agent_name, "User"])
    
    # Find the agent
    agent = next((a for a in group_chat.agents if a.name == agent_name), None)
    if not agent:
        print(f"Error: Agent {agent_name} not found in group chat.")
        return

    # Check registered tools
    # AutoGen agents store tools in client.tools or similar, but register_for_llm registers it for the LLM config.
    # We can check llm_config["tools"]
    
    if hasattr(agent, "llm_config") and agent.llm_config and "tools" in agent.llm_config:
        tools = agent.llm_config["tools"]
        print(f"Found {len(tools)} tools registered for {agent_name}:")
        for tool in tools:
            print(f" - {tool.get('function', {}).get('name')}")
            
        # Check if expected MCP tools are present
        expected_tools = ["get_database_list", "get_table_list", "execute_select"]
        missing = [t for t in expected_tools if not any(t == tool.get('function', {}).get('name') for tool in tools)]
        
        if missing:
            print(f"WARNING: Some expected tool definitions are missing: {missing}")
        else:
            print("SUCCESS: All expected MCP tools appear to be registered in llm_config.")
    else:
        print("Error: No tools found in agent's llm_config.")

if __name__ == "__main__":
    verify_mcp_tools()
