import glob
import os

files = glob.glob(r"venv\Lib\site-packages\autogen\oai\*.py")
# Possible import patterns
target1 = "from autogen import LLMConfig"
target2 = "from .. import LLMConfig"
target3 = "from ..llm_config import LLMConfig"

replacement = "from ..llm_config.config import LLMConfig"

for file_path in files:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        patched = False
        if target1 in content:
            content = content.replace(target1, replacement)
            patched = True
        if target2 in content:
            content = content.replace(target2, replacement)
            patched = True
        if target3 in content:
            content = content.replace(target3, replacement)
            patched = True
            
        if patched:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Patched {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
