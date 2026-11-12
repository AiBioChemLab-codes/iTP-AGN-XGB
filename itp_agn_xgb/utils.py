"""Shared constants and file utilities."""
from pathlib import Path
import hashlib
import json


TASKS = ['AAP', 'ABP', 'ACP', 'AIP', 'AVP', 'CPP', 'PBP', 'QSP']
STAGES = ['audit', 'extract', 'split', 'select', 'evaluate', 'compare', 'report']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
