from ml_collections import config_dict
from ruamel.yaml import YAML
from typing import Any
import os
import pathlib

def load_config_from_yaml(pathFilename: str | os.PathLike[Any]) -> config_dict.ConfigDict:
    rYaml = YAML(typ='safe')
    dataYaml = rYaml.load(pathlib.Path(pathFilename).read_text())
    return config_dict.ConfigDict(dataYaml)
