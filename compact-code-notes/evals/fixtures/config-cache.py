# 独立的练习源码，不属于用户的训练框架。
import json
from pathlib import Path


class ConfigStore:
    def __init__(self):
        self._cache = {}

    def load(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1].copy()
        values = json.loads(path.read_text(encoding="utf-8"))
        self._cache[path] = (signature, values)
        return values.copy()
