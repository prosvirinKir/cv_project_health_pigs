# agro-hack
The solution from neuropeppa team for pigs tracking and computing theirs activity

<div align="center">

[![license](https://shields.io/badge/license-MIT-green)](https://github.com/Ilyabasharov/agro-hack/blob/main/LICENSE)

[📘Introduction](https://github.com/Ilyabasharov/agro-hack/blob/main/README.md#introduction) |
[🛠️Installation](https://github.com/Ilyabasharov/agro-hack/blob/main/README.md#installation) |
[👀Project Structure](https://github.com/Ilyabasharov/agro-hack/blob/main/README.md#project-structure)

</div>

## Introduction

This is the part of smart pigs farm. It can efficiently track pigs and compute their activity to prevent diseases in future.

The master branch works with **PyTorch 1.10** and **CUDA 10.2**.

<details open>
<summary>Major features</summary>

- **Modular Design**

  We decompose the search framework into two components: detection and tracking modules. 

- **High efficiency**

  It supports GPU support for faster inference.

- **State of the art**

  We use [HTC](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc) for robust detection and [PointTrack](https://github.com/detectRecog/PointTrack) for precise instance comparation between each other and pigs identification.
  
</details>

## License

This project is released under the [MIT license](LICENSE).

## Project Structure

```
├── configs           <- Main configs for loading models, databases etc.
├── src               <- Scripts for evaluation/training.
├── notebooks         <- Module operation demonstrations.
├── resources         <- Logos, images etc.
├── scripts           <- Scripts for configuration, downloading weights.
│    │
│    └── download_weights.sh     <- Downloading all used NN weights.
├── LICENSE           <- License.
├── README.md         <- You are here.
├── main.py           <- Entrypoint file
└── requirements.txt  <- Required libraries. Should be installed before running the project.
```

## Installation

```bash
# clone the repo
git clone https://github.com/Ilyabasharov/agro-hack.git
cd agro-hack

# install the requirements
pip install -r requirements.txt

# download weights
cd scripts && sh download_weights.sh
```

## Running the project

You can evaluate at video. For this you should define input_path and output_path in [main.py](https://github.com/Ilyabasharov/agro-hack/blob/main/src/main.py) file.
Read this file before you start working!

```bash
# return to root
cd src
python3 main.py --input_folder=<path_to_input_folder> --output_folder=<path_to_output_folder>
```
