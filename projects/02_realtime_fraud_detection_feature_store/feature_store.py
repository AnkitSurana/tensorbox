"""
In-Memory Low-Latency Feature Store.
"""

from typing import Dict, Any, Optional

class FeatureStore:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def put(self, entity_id: str, features: Dict[str, Any]):
        self._store[entity_id] = features

    def get(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(entity_id, None)

    def count(self) -> int:
        return len(self._store)
