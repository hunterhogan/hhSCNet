import warnings
warnings.filterwarnings('always', module='hhSCNet.*')

from hhSCNet.purgatory import loadModelConfigurationYaml
from hhSCNet.inference import runInference
from hhSCNet.train import trainModel
from hhSCNet.commandLine import processCommandLine

def main() -> int:
	"""Entry point for the command line interface."""
	import sys
	return processCommandLine()

if __name__ == "__main__":
	import sys
	sys.exit(main())