from pathlib import Path
import yaml

def load_configurations(path):
    with open(path,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)