from types import SimpleNamespace
import yaml

__all__ = [
    "yaml_to_dotdict",
]


class DotDict(SimpleNamespace):
    """
    A dictionary subclass that supports dot notation access to its keys.

    Args:
        **kwargs: Key-value pairs to be converted into dot notation accessible attributes.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = DotDict(**value)
            setattr(self, key, value)


def yaml_to_dotdict(yaml_file: str) -> DotDict:
    """
    Convert a YAML file to a DotDict.

    Args:
        yaml_file (str): The path to the YAML file.

    Returns:
        DotDict: A DotDict object with attributes corresponding to the YAML keys.
    """
    with open(yaml_file, "r") as op:
        data = yaml.safe_load(op)
    return DotDict(**data)
