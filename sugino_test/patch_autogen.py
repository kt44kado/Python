import glob
import os

files = glob.glob(r"venv\Lib\site-packages\autogen\oai\*.py")
target = "from ..llm_config import LLMConfigEntry"
replacement = "from ..llm_config.entry import LLMConfigEntry"

for file_path in files:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if target in content:
            new_content = content.replace(target, replacement)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Patched {file_path}")
        else:
            print(f"Skipped {file_path} (target not found)")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
