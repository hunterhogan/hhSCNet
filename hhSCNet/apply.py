import random
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.nn import functional as F

from .utils import DummyPoolExecutor, center_trim

accelerator = Accelerator()

class TensorChunk:
	def __init__(self, tensor, offset=0, length=None):
		total_length = tensor.shape[-1]
		assert offset >= 0
		assert offset < total_length

		if length is None:
			length = total_length - offset
		else:
			length = min(total_length - offset, length)

		self.tensor = tensor
		self.offset = offset
		self.length = length
		self.device = tensor.device

	@property
	def shape(self):
		shape = list(self.tensor.shape)
		shape[-1] = self.length
		return shape

	def padded(self, target_length):
		delta = target_length - self.length
		total_length = self.tensor.shape[-1]
		assert delta >= 0

		start = self.offset - delta // 2
		end = start + target_length

		correct_start = max(0, start)
		correct_end = min(total_length, end)

		pad_left = correct_start - start
		pad_right = end - correct_end

		out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
		assert out.shape[-1] == target_length
		return out


def tensor_chunk(tensor_or_chunk):
	if isinstance(tensor_or_chunk, TensorChunk):
		return tensor_or_chunk
	else:
		assert isinstance(tensor_or_chunk, torch.Tensor)
		return TensorChunk(tensor_or_chunk)


def apply_model(model, mix, shifts=1, split=True, segment=20, samplerate=44100,
				overlap=0.25, transition_power=1., progress=False, device=None,
				num_workers=0, pool=None):
	"""
	Apply model to a given mixture.

	Args:
		shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
			and apply the opposite shift to the output. This is repeated `shifts`-times and
			all predictions are averaged. This effectively makes the model time equivariant
			and improves SDR by up to 0.2 points.
		split (bool): if True, the input will be broken down in 8 seconds extracts
			and predictions will be performed individually on each and concatenated.
			Useful for model with large memory footprint like Tasnet.
		progress (bool): if True, show a progress bar (requires split=True)
		device (torch.device, str, or None): if provided, device on which to
			execute the computation, otherwise `mix.device` is assumed.
			When `device` is different from `mix.device`, only local computations will
			be on `device`, while the entire tracks will be stored on `mix.device`.
	"""
	device = accelerator.device
	if pool is None:
		if num_workers > 0 and device.type == 'cpu':
			pool = ThreadPoolExecutor(num_workers)
		else:
			pool = DummyPoolExecutor()
	kwargs = {
		'shifts': shifts,
		'split': split,
		'overlap': overlap,
		'transition_power': transition_power,
		'progress': progress,
		'device': device,
		'pool': pool,
	}
	model = accelerator.unwrap_model(model)
	model.to(device)

	assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
	batch, channels, length = mix.shape
	if split:
		kwargs['split'] = False
		out = torch.zeros(batch, len(model.sources), channels, length, device=mix.device)
		sum_weight = torch.zeros(length, device=mix.device)
		segment = int(samplerate * segment)
		stride = int((1 - overlap) * segment)
		offsets = range(0, length, stride)
		scale = stride / samplerate
		weight = torch.cat([torch.arange(1, segment // 2 + 1, device=device),
						 torch.arange(segment - segment // 2, 0, -1, device=device)])			  
		assert len(weight) == segment
		# If the overlap < 50%, this will translate to linear transition when
		# transition_power is 1.
		weight = (weight / weight.max())**transition_power
		futures = []
		for offset in offsets:
			chunk = TensorChunk(mix, offset, segment)
			future = pool.submit(apply_model, model, chunk, **kwargs)
			futures.append((future, offset))
			offset += segment
		if progress:
			futures = tqdm(futures, unit_scale=scale, ncols=120, unit='seconds')
		for future, offset in futures:
			chunk_out = future.result()
			chunk_length = chunk_out.shape[-1]
			out[..., offset:offset + segment] += (weight[:chunk_length] * chunk_out).to(mix.device)
			sum_weight[offset:offset + segment] += weight[:chunk_length].to(mix.device)
		assert sum_weight.min() > 0
		out /= sum_weight
		return out
	elif shifts:
		kwargs['shifts'] = 0
		max_shift = int(0.5 * samplerate)
		mix = tensor_chunk(mix)
		padded_mix = mix.padded(length + 2 * max_shift)

		out = torch.tensor(0)

		for _ in range(shifts):
			offset = random.randint(0, max_shift)
			shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
			shifted_out = apply_model(model, shifted, **kwargs)
			out += shifted_out[..., max_shift - offset:]
		out /= shifts
		return out
	else:
		mix = tensor_chunk(mix)
		padded_mix = mix.padded(length).to(device)
		with torch.no_grad():
			out = model(padded_mix)
		return center_trim(out, length)

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
