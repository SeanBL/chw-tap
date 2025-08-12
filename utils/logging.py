import os
from datetime import datetime

def log_to_file(log_path, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)  # Optional: also print to console
    dirpath = os.path.dirname(log_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")


