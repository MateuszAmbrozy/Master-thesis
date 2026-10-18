"""Compatibility interface retaining the historical classic settings."""
from ..environment import MAPFEnv as BaseMAPFEnv, Map, PROFILES


class MAPFEnv(BaseMAPFEnv):
    def __init__(self, *args, settings=None, **kwargs):
        super().__init__(*args, settings=settings or PROFILES["classic"], **kwargs)
