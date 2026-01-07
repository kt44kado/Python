import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    import agents
    print("Successfully imported agents.")
    print(f"Loaded {len(agents.DR_SUM_AGENTS_DEF)} agent definitions.")
    print(f"First agent: {agents.DR_SUM_AGENTS_DEF[0]}")
    
    # Check if description is used in system message
    msg = agents.generate_agent_system_message("受注情報", ["User", "受注情報"])
    if "【データ概要】受注情報" in msg:
        print("System message correctly includes description.")
    else:
        print("WARNING: System message missing description.")
        print(msg)

except Exception as e:
    print(f"Error: {e}")
