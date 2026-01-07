import glob
import os

files = glob.glob(r"venv\Lib\site-packages\autogen\oai\*.py")
target = "from ..llm_config import register_llm_config"
# Since we exported it in __init__.py, the original import is technically VALID now!
# BUT to avoid circular import (if __init__ still triggers it via other imports), it's safer to import from .entry directly.
# However, if __init__.py imports from .entry, and .entry does NOT import anything else, we are safe.
# The issue before was types.py importing anthropic.py importing llm_config importing types.py.
# If we change anthropic.py to import from ..llm_config.entry, we bypass llm_config package init?
# Yes.
replacement = "from ..llm_config.entry import register_llm_config"

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
            # Maybe it wasn't on one line? Check logic.
            pass
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
