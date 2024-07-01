from types import SimpleNamespace
from typing import Any, Dict

import yaml

__all__ = [
    "yaml_to_dotdict",
]


class DotDict(SimpleNamespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = DotDict(**value)
            setattr(self, key, value)


def yaml_to_dotdict(yaml_file: str) -> DotDict:
    with open(yaml_file, "r") as op:
        data = yaml.safe_load(op)
    return DotDict(**data)
