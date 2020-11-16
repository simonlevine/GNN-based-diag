import networkx as nx
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt

def visualize_graph(instance):
    graph = to_networkx(instance)
    plt.figure(1,figsize=(14,12)) 
    nx.draw(graph,node_size=75,linewidths=6,node_color='red',edge_color='#00b4d9')
    plt.savefig('./first_graph.png')
