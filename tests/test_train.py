from hhSCNet import load_config_from_yaml
from ml_collections import config_dict
from pathlib import Path
from ruamel.yaml import YAML
from types import SimpleNamespace
import pathlib
import pytest

def test_load_config_from_yaml():
    """Tests loading a config from a YAML file using configSample.yaml."""
    config_file = pathlib.Path(__file__).parent / "dataSamples/configSample.yaml"
    config = load_config_from_yaml(config_file)
    assert isinstance(config, config_dict.ConfigDict)
    assert config.data.wav == "/root/autodl-tmp/musdbhq" # type: ignore
    assert config.data.samplerate == 44100 # type: ignore
    assert config.model.sources == ['drums', 'bass', 'other', 'vocals'] # type: ignore
    # ...additional assertions as needed...

def test_load_config_from_yaml_invalid_path():
    """Tests handling of an invalid file path."""
    with pytest.raises(FileNotFoundError):
        load_config_from_yaml("nonexistent_file.yaml")

def test_load_config_from_yaml_malformed_yaml():
    """Tests handling of a malformed YAML file."""
    malformed_yaml_file = Path("./temp_malformed.yaml")
    malformed_yaml_file.write_text("invalid yaml")
    with pytest.raises(Exception):
        load_config_from_yaml(malformed_yaml_file)
    malformed_yaml_file.unlink()

def test_yaml_loading(tmp_path):
    # Create a path to the sample config
    sample_config_path = pathlib.Path(__file__).parent / "dataSamples/configSample.yaml"
    
    # Create args namespace with config path
    args = SimpleNamespace(config_path=str(sample_config_path))
    
    rYaml = YAML(typ='safe')    
    dataYaml = rYaml.load(pathlib.Path(args.config_path).read_text())
    
    # Basic assertions to verify the YAML was loaded correctly
    assert isinstance(dataYaml, dict)
    assert 'data' in dataYaml
    assert 'model' in dataYaml
    assert dataYaml['data']['samplerate'] == 44100
    assert dataYaml['model']['sources'] == ['drums', 'bass', 'other', 'vocals']
