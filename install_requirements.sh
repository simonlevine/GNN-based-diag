#!/bin/bash

# CUDA=cu110
pip install torch torchvision
pip install torch-scatter==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-sparse==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-cluster==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-spline-conv==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-geometric