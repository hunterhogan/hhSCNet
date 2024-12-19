import os
import tempfile
import typing as tp
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import julius
import torch


# Audio
def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    if wav.ndim == 1:
        src_channels = 1
    else:
        src_channels = wav.shape[-2]

    if src_channels == channels:
        pass
    elif channels == 1:
        if src_channels > 1:
            wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        wav = wav.expand(-1, channels, -1)
    elif src_channels >= channels:
        wav = wav[..., :channels, :]
    else:
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav

def convert_audio(wav, from_samplerate, to_samplerate, channels):
    """Convert audio from a given samplerate to a target one and target number of channels."""
    # Convert channels first
    wav = convert_audio_channels(wav, channels)
    
    # Resample audio if necessary
    if from_samplerate != to_samplerate:
        wav = julius.resample_frac(wav, from_samplerate, to_samplerate)
    return wav


# model
def load_model(model, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No model checkpoint file found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        if 'best_state' not in checkpoint:
            raise KeyError(f"Checkpoint does not contain the state")
            
        state_dict = checkpoint['best_state']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
               new_state_dict[k[7:]] = v
            else:
               new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        return model

def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}

@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state, strict=False)
    try:
        yield
    finally:
        model.load_state_dict(old_state)

#other
@contextmanager
def temp_filenames(count: int, delete=True):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)

def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor

def EMA(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatidly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: dict, weight: float = 1) -> dict:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update

class DummyPoolExecutor:
    class DummyResult:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.func(*self.args, **self.kwargs)

    def __init__(self, workers=0):
        pass

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return

# metric
def new_sdr(references, estimates):
    """
    Compute the SDR for a song
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(torch.square(references), dim=(2, 3))
    den = torch.sum(torch.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * torch.log10(num / den)
    return scores

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
