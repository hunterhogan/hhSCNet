# hhSCNet

See [SCNet: Sparse Compression Network for Music Source Separation](https://arxiv.org/abs/2401.13276)

## Usage

### Command Line Interface

#### Separate audio

```sh
hhSCNet inference --pathInput "./input/" --pathOutput "./output/" --modelConfiguration "./conf/config.yaml" --checkpoint "./result/checkpoint.th"
```

#### Train a model

```sh
hhSCNet train --modelConfiguration "./conf/config.yaml" --pathSave "./result/"
```

### Python API

#### Inference

```python
from hhSCNet import runInference

runInference(
    pathInput="./input/",
    pathOutput="./output/",
    modelConfiguration="./conf/config.yaml",
    checkpoint="./result/checkpoint.th"
)
```

#### Training

```python
from hhSCNet import trainModel

trainModel(
    modelConfiguration="./conf/config.yaml",
    pathSave="./result/"
)
```

## Install this package

### From Github

```sh
pip install hhSCNet@git+https://github.com/hunterhogan/hhSCNet.git
```

### From a local directory

#### Windows

```cmd
git clone https://github.com/hunterhogan/hhSCNet.git \path\to\hhSCNet
pip install hhSCNet@file:\path\to\hhSCNet
```

#### POSIX

```bash
git clone https://github.com/hunterhogan/hhSCNet.git /path/to/hhSCNet
pip install hhSCNet@file:/path/to/hhSCNet
```

## Install updates

```sh
pip install --upgrade hhSCNet@git+https://github.com/hunterhogan/hhSCNet.git
```

## Cite the original paper

```bibtex
@misc{tong2024scnet,
      title={SCNet: Sparse Compression Network for Music Source Separation}, 
      author={Weinan Tong and Jiaxu Zhu and Jun Chen and Shiyin Kang and Tao Jiang and Yang Li and Zhiyong Wu and Helen Meng},
      year={2024},
      eprint={2401.13276},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
