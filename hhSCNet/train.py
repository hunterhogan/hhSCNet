import os
import torch
from torch.utils.data import DataLoader
from .wav import get_wav_datasets
from .SCNet import SCNet
from .solver import Solver
import argparse
import yaml
import sys
from ml_collections import ConfigDict
from accelerate import Accelerator
from .log import logger

accelerator = Accelerator()

def get_model(config):

    model = SCNet(**config.model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters: {total_params}")
    return model


def get_solver(args):
    with open(args.config_path, 'r') as file:
          config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))

    torch.manual_seed(config.seed)
    model = get_model(config)
  
    # torch also initialize cuda seed if available
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    if config.optim.optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.optim.lr,
            betas=(config.optim.momentum, config.optim.beta2),
            weight_decay=config.optim.weight_decay)
    elif config.optim.optim == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.optim.lr,
            betas=(config.optim.optim.momentum, config.optim.beta2),
            weight_decay=config.optim.weight_decay)

    train_set, valid_set = get_wav_datasets(config.data)

    logger.info("train/valid set size: %d %d", len(train_set), len(valid_set))
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.misc.num_workers, drop_last=True)
    train_loader = accelerator.prepare_data_loader(train_loader)

    valid_loader = DataLoader(
        valid_set, batch_size=1, shuffle=False,
        num_workers=config.misc.num_workers)
    valid_loader = accelerator.prepare_data_loader(valid_loader)

    loaders = {"train": train_loader, "valid": valid_loader}

    model, optimizer = accelerator.prepare(model, optimizer)
    
    return Solver(loaders, model, optimizer, config, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='./result/', help="path to config file")
    parser.add_argument("--config_path", type=str, default='./conf/config.yaml', help="path to save checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if not os.path.isfile(args.config_path):
        print(f"Error: config file {args.config_path} does not exist.")
        sys.exit(1)
        
    solver = get_solver(args)
    accelerator.wait_for_everyone()
    solver.train()



if __name__ == "__main__":
    main()

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
SOFTWARE.
"""
