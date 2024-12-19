import pathlib
import pytest
from ml_collections import config_dict
from typing import List
from ruamel.yaml import YAML
from types import SimpleNamespace

from hhSCNet import loadModelConfigurationYaml
from hhSCNet.commandLine import createParserWithCommands, processCommandLine, registryCommands, extractFunctionMetadata

# Constants for test data paths
pathDirectoryTests = pathlib.Path(__file__).parent
pathDirectorySamples = pathDirectoryTests / "dataSamples"
pathConfigSample = pathDirectorySamples / "configSample.yaml"

def build_command_args(command: str, **kwargs: str) -> List[str]:
    """Build command line arguments list from command and parameters."""
    result = [command]
    for key, value in kwargs.items():
        result.extend([f'--{key}', value])
    return result

@pytest.fixture
def mock_command_modules(monkeypatch):
    """Mock all registered command modules."""
    for registryCommand in registryCommands:
        def mock_function(*args, **kwargs): pass
        module = type('MockModule', (), {registryCommand.function: mock_function})
        monkeypatch.setattr(f'hhSCNet{registryCommand.module}', module)

@pytest.fixture
def parserCommand():
    """Create the command parser."""
    return createParserWithCommands()

@pytest.fixture
def configurationYaml():
    """Load the sample YAML configuration for testing."""
    return loadModelConfigurationYaml(pathConfigSample)

def test_processCommandLine_basic_execution(mock_command_modules):
    """Test basic command execution for each registered command."""
    for registryCommand in registryCommands:
        # Extract required parameters from function signature
        # no, DRY. commandLine does this
        pass

def test_load_config_yaml(configurationYaml):
    """Test configuration loading from YAML."""
    assert isinstance(configurationYaml, config_dict.ConfigDict)
    assert configurationYaml.data.wav == "/root/autodl-tmp/musdbhq" # type: ignore
    assert configurationYaml.data.samplerate == 44100 # type: ignore
    assert configurationYaml.model.sources == ['drums', 'bass', 'other', 'vocals'] # type: ignore

@pytest.mark.parametrize("invalidConfig", [
    "nonexistent_file.yaml",
    pathDirectorySamples / "tmp/malformed.yaml"
])
def test_load_config_yaml_errors(invalidConfig):
    """Test error handling for invalid configurations."""
    with pytest.raises((FileNotFoundError, Exception)):
        loadModelConfigurationYaml(invalidConfig)

def test_yaml_loading():
    """Test direct YAML loading functionality."""
    args = SimpleNamespace(config_path=str(pathConfigSample))
    rYaml = YAML(typ='safe')    
    dataYaml = rYaml.load(pathlib.Path(args.config_path).read_text())
    
    assert isinstance(dataYaml, dict)
    assert all(key in dataYaml for key in ['data', 'model'])
    assert dataYaml['data']['samplerate'] == 44100
    assert dataYaml['model']['sources'] == ['drums', 'bass', 'other', 'vocals']
