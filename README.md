# GNN-based-diag
A repository for the 02-740 Bioimage Informatics course project on GNN-based cancer detection and classification.

- WIP: https://drive.google.com/file/d/18EfaqRfm8PwYN7Vg5b_r0sqqOF8FZt5o/view?usp=sharing

- Pipeline:
    - a) Images --> Node/Edge list --> Graphical Embedding (Pytorch Geometric/Node2Vec) -> GNN/Sk-learn Classifiers
    - b) Images --> Classical Embedding --> Sk-learn classifiers

## Initial Results:
- We use RGB values on truncated portions of this dataset (https://www.kaggle.com/andrewmvd/malignant-lymphoma-classification) as node labels.
- 120 epochs with a baseline GCN:
![Run 1][run1.png]

- 120 epochs with a resnet-style skip connection GCN:
![Run 2][run2.png]
