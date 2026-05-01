import inspect
import types
from typing import Any, Dict, Set
import numpy as np

def to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj,np.ndarray) or isinstance(obj,np.ndarray):
            return obj.tolist()
    elif hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            result[key] = to_dict(value)  # 递归处理
        return result
    elif isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    else:
        return obj
