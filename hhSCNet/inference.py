import os
import time

import numpy as np
import soundfile as sf
import torch

from hhSCNet import loadModelConfigurationYaml

from .apply import apply_model
from .SCNet import SCNet
from .utils import convert_audio, load_model


class Separator:
    def __init__(self, model, checkpoint_path):
        self.separator = load_model(model, checkpoint_path)

        if torch.cuda.device_count():
            self.device = torch.device('cuda')
        else:
            print("WARNING, using CPU")
            self.device = torch.device('cpu')
        self.separator.to(self.device)

    @property
    def instruments(self):
        return ['bass', 'drums', 'other', 'vocals']

    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def separate_music_file(self, mixed_sound_array, sample_rate):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate
        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """
        mix = torch.from_numpy(np.asarray(mixed_sound_array.T, np.float32))

        # convert audio to GPU
        mix = mix.to(self.device)
        mix_channels = mix.shape[0]
        mix = convert_audio(mix, sample_rate, 44100, self.separator.audio_channels)

        b = time.time()
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std
        # Separate
        with torch.no_grad():
            estimates = apply_model(self.separator, mix[None], overlap=0.5, progress=False)[0]

        # Printing some sanity checks.
        print(time.time() - b, mono.shape[-1] / sample_rate, mix.std(), estimates.std())

        estimates = estimates * std + mean

        estimates = convert_audio(estimates, 44100, sample_rate, mix_channels)

        separated_music_arrays = {}
        output_sample_rates = {}
        for instrument in self.instruments:
            idx = self.separator.sources.index(instrument)
            separated_music_arrays[instrument] = torch.squeeze(estimates[idx]).detach().cpu().numpy().T
            output_sample_rates[instrument] = sample_rate

        return separated_music_arrays, output_sample_rates


    def load_audio(self, file_path):
        try:
            data, sample_rate = sf.read(file_path, dtype='float32')
            return data, sample_rate
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            raise

    def save_sources(self, sources, output_sample_rates, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for name, src in sources.items():
            save_path = os.path.join(save_dir, f'{name}.wav')
            sf.write(save_path, src, output_sample_rates[name])
            print(f"Saved {name} to {save_path}")

    def process_directory(self, input_dir, output_dir):
        for entry in os.listdir(input_dir):
            entry_path = os.path.join(input_dir, entry)
            if os.path.isdir(entry_path):
                mixture_path = os.path.join(entry_path, 'mixture.wav')
                if os.path.isfile(mixture_path):
                    print(f"Processing {mixture_path}")
                    entry_name = os.path.basename(entry)
                else:
                    continue
            elif os.path.isfile(entry_path) and entry_path.lower().endswith('.wav'):
                print(f"Processing {entry_path}")
                mixture_path = entry_path
                entry_name = os.path.splitext(os.path.basename(entry))[0]
            else:
                continue

            mixed_sound_array, sample_rate = self.load_audio(mixture_path)
            separated_music_arrays, output_sample_rates = self.separate_music_file(mixed_sound_array, sample_rate)
            save_dir = os.path.join(output_dir, entry_name)
            self.save_sources(separated_music_arrays, output_sample_rates, save_dir)

def runInference(pathInput: str, pathOutput: str, 
                modelConfiguration: str = "./conf/config.yaml",
                checkpoint: str = "./result/checkpoint.th") -> None:
    """Run inference on audio files.
    
    Parameters
        pathInput: Input directory containing audio files to separate
        pathOutput: Output directory to save separated sources
        modelConfiguration: Path to model configuration YAML file
        checkpoint: Path to model checkpoint file
    """
    config = loadModelConfigurationYaml(modelConfiguration)

    if not os.path.exists(pathOutput):
        os.mkdir(pathOutput)

    model = SCNet(**config.model)
    model.eval()
    separator = Separator(model, checkpoint)
    separator.process_directory(pathInput, pathOutput)

# Some or all of the work in this file may be restricted by the following copyright.
"""
MIT License

Copyright (c) 2024 starrytong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
