import os
import yaml
from types import SimpleNamespace
from typing import Any, Dict

def _dict_to_ns(d: Dict[str, Any]) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_ns(v))
        else:
            setattr(ns, k, v)
    return ns

def load_config(path: str) -> SimpleNamespace:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return _dict_to_ns(data)
