import random

import torch
from torch import nn


class Shift(nn.Module):
	"""
	Randomly shift audio in time by up to `shift` samples.
	"""
	def __init__(self, shift=8192, same=False):
		super().__init__()
		self.shift = shift
		self.same = same

	def forward(self, wav):
		batch, sources, channels, time = wav.size()
		length = time - self.shift
		if self.shift > 0:
			if not self.training:
				wav = wav[..., :length]
			else:
				srcs = 1 if self.same else sources
				offsets = torch.randint(self.shift, [batch, srcs, 1, 1], device=wav.device)
				offsets = offsets.expand(-1, sources, channels, -1)
				indexes = torch.arange(length, device=wav.device)
				wav = wav.gather(3, indexes + offsets)
		return wav

class FlipChannels(nn.Module):
	"""
	Flip left-right channels.
	"""
	def forward(self, wav):
		batch, sources, channels, time = wav.size()
		if self.training and wav.size(2) == 2:
			left = torch.randint(2, (batch, sources, 1, 1), device=wav.device)
			left = left.expand(-1, -1, -1, time)
			right = 1 - left
			wav = torch.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
		return wav

class FlipSign(nn.Module):
	"""
	Random sign flip.
	"""
	def forward(self, wav):
		batch, sources, channels, time = wav.size()
		if self.training:
			signs = torch.randint(2, (batch, sources, 1, 1), device=wav.device, dtype=torch.float32)
			wav = wav * (2 * signs - 1)
		return wav

class Remix(nn.Module):
	"""
	Shuffle sources to make new mixes.
	"""
	def __init__(self, proba=1, group_size=4):
		"""
		Shuffle sources within one batch.
		Each batch is divided into groups of size `group_size` and shuffling is done within
		each group separately. This allow to keep the same probability distribution no matter
		the number of GPUs. Without this grouping, using more GPUs would lead to a higher
		probability of keeping two sources from the same track together which can impact
		performance.
		"""
		super().__init__()
		self.proba = proba
		self.group_size = group_size

	def forward(self, wav):
		batch, streams, channels, time = wav.size()
		device = wav.device

		if self.training and random.random() < self.proba:
			group_size = self.group_size or batch
			if batch % group_size != 0:
				raise ValueError(f"Batch size {batch} must be divisible by group size {group_size}")
			groups = batch // group_size
			wav = wav.view(groups, group_size, streams, channels, time)
			permutations = torch.argsort(torch.rand(groups, group_size, streams, 1, 1, device=device),
									  dim=1)
			wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
			wav = wav.view(batch, streams, channels, time)
		return wav

class Scale(nn.Module):
	def __init__(self, proba=1., min=0.25, max=1.25):
		super().__init__()
		self.proba = proba
		self.min = min
		self.max = max

	def forward(self, wav):
		batch, streams, channels, time = wav.size()
		device = wav.device
		if self.training and random.random() < self.proba:
			scales = torch.empty(batch, streams, 1, 1, device=device).uniform_(self.min, self.max)
			wav *= scales
		return wav

# Some or all of the work in this file may be restricted by the following copyright.
"""
MIT License

Copyright (c) Meta, Inc. and its affiliates.

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
