import torch
import torch.nn.functional as F

def spec_rmse_loss(estimate, sources, stft_config):

	_, _, _, lenc = estimate.shape
	spec_estimate = estimate.view(-1, lenc)
	spec_sources = sources.view(-1, lenc)

	spec_estimate = torch.stft(spec_estimate, **stft_config, return_complex=True)
	spec_sources = torch.stft(spec_sources, **stft_config, return_complex=True)

	spec_estimate = torch.view_as_real(spec_estimate)
	spec_sources = torch.view_as_real(spec_sources)

	new_shape = estimate.shape[:-1] + spec_estimate.shape[-3:]
	spec_estimate = spec_estimate.view(*new_shape)
	spec_sources = spec_sources.view(*new_shape)


	loss = F.mse_loss(spec_estimate, spec_sources, reduction='none')


	dims = tuple(range(2, loss.dim()))
	loss = loss.mean(dims).sqrt().mean(dim=(0, 1))  

	return loss

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
