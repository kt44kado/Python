import glob
import os

files = glob.glob(r"venv\Lib\site-packages\autogen\oai\*.py")
target = "import autogen"

for file_path in files:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        new_lines = []
        patched = False
        for line in lines:
            if line.strip() == target:
                # Comment it out or skip
                # commenting out is safer for debug
                new_lines.append(f"# {line}")
                patched = True
            else:
                new_lines.append(line)
        
        if patched:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            print(f"Patched {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
