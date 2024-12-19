from typing import Optional, List, Dict, Any, Callable, NoReturn
import argparse
import importlib
import inspect
import sys
from dataclasses import dataclass
from typing import get_type_hints

@dataclass
class registrantCommand:
    """Registry entry for a command."""
    name: str
    module: str
    function: str
    docstring: str = ""  # SSOT: pulled from function at runtime

registryCommands = [
    registrantCommand(
        name='train',
        module='.train',
        function='trainModel'
    ),
    registrantCommand(
        name='inference',
        module='.inference',
        function='runInference'
    )
]

def extractFunctionMetadata(functionTarget: Callable) -> Dict[str, Any]:
    """Extract parameter information from function signature and docstring."""
    signatureFunction = inspect.signature(functionTarget)
    docstringFunction = inspect.getdoc(functionTarget) or ""
    
    dictionaryDescriptionsParameters = {}
    if docstringFunction:
        listLines = docstringFunction.split('\n')
        isInParameters = False
        for line in listLines:
            lineContent = line.strip()
            if lineContent.lower().startswith('parameters'):
                isInParameters = True
                continue
            if isInParameters and lineContent:
                if not lineContent.startswith(' '):
                    isInParameters = False
                    continue
                if ':' in lineContent:
                    parameterName, descriptionParameter = lineContent.split(':', 1)
                    dictionaryDescriptionsParameters[parameterName.strip()] = descriptionParameter.strip()
    
    dictionaryParameters = {}
    for nameParameter, parameterInfo in signatureFunction.parameters.items():
        dictionaryParameters[nameParameter] = {
            'help': dictionaryDescriptionsParameters.get(nameParameter, ''),
            'type': get_type_hints(functionTarget).get(nameParameter, str),
            'default': None if parameterInfo.default is inspect.Parameter.empty else parameterInfo.default,
            'required': parameterInfo.default is inspect.Parameter.empty
        }
    
    return dictionaryParameters

def addParameterToParser(parserCommand: argparse.ArgumentParser, nameParameter: str, dictionaryParameterInfo: Dict[str, Any]) -> None:
    """Add a parameter to the parser."""
    dictionaryKwargs = {
        'help': dictionaryParameterInfo['help'],
        'type': dictionaryParameterInfo['type'],
        'required': dictionaryParameterInfo['required'],
        'dest': nameParameter
    }
    
    if not dictionaryParameterInfo['required']:
        dictionaryKwargs['default'] = dictionaryParameterInfo['default']
    
    parserCommand.add_argument(f'--{nameParameter}', **dictionaryKwargs)

def createParserWithCommands() -> argparse.ArgumentParser:
    """Create the command line parser with all registered commands."""
    parserMain = argparse.ArgumentParser(
        prog='hhSCNet',
        description="hhSCNet - Music Source Separation"
    )
    subparsers = parserMain.add_subparsers(dest='command', required=True)
    
    for registryCommand in registryCommands:
        moduleCommand = importlib.import_module(registryCommand.module, package=__package__)
        functionCommand = getattr(moduleCommand, registryCommand.function)
        registryCommand.docstring = inspect.getdoc(functionCommand) or ""
        
        # Create parser for this command
        parserCommand = subparsers.add_parser(
            registryCommand.name, 
            help=registryCommand.docstring.split('\n')[0]
        )
        
        # Add parameters based on function signature and docstring
        dictionaryParameters = extractFunctionMetadata(functionCommand)
        for nameParameter, dictionaryParameterInfo in dictionaryParameters.items():
            addParameterToParser(parserCommand, nameParameter, dictionaryParameterInfo)
    
    return parserMain

def processCommandLine(listArguments: Optional[List[str]] = None) -> int:
    """Process command line arguments and execute the corresponding function."""
    if listArguments is None:
        listArguments = sys.argv[1:]

    parserMain = createParserWithCommands()
    arguments = parserMain.parse_args(listArguments)

    # Get command info
    registryCommand = next((c for c in registryCommands if c.name == arguments.command), None)
    if not registryCommand:
        parserMain.print_help()
        return 1

    # Import function and execute
    moduleCommand = importlib.import_module(registryCommand.module, package=__package__)
    functionCommand = getattr(moduleCommand, registryCommand.function)

    # Convert arguments to keyword arguments
    keywordArguments = vars(arguments)
    keywordArguments.pop('command')

    functionCommand(**keywordArguments)
    return 0

if __name__ == "__main__":
    import sys
    from hhSCNet import main
    sys.exit(main())