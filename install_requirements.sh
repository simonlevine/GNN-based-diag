#!/bin/bash

# CUDA=cu110

pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-sparse==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-cluster==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-spline-conv==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-geometric

# gsutil cp -r gs://new-bucket-fall2020/data ./