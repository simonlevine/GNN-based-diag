import torch
import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data import Dataset
from loguru import logger
from tqdm.notebook import tqdm

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader

from modules import GNN,GCN,CellGraphDataset
# from visualizations import visualize_graph

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8

def main():

    dataset = CellGraphDataset(root='./data', name = 'DS_sample',use_node_attr=False,use_edge_attr=False)
    # visualize_graph(dataset[0])
    print_graph_stats(dataset)

    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    n_trn = (int(round(.67*len(dataset))))

    train_dataset = dataset[:n_trn]
    test_dataset = dataset[n_trn:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()

    model = GNN(hidden_channels=64,dataset=dataset).to(DEVICE)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 201):
        train_results = train(model,train_loader,optimizer,criterion)
        test_results = test(model,test_loader)

        train_acc = train_results['accuracy']
        test_acc = test_results['accuracy']
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


def train(model, train_loader, optimizer, criterion):
    model.train()
    results={'loss':[],'accuracy':[]}
    correct = 0

    for i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.

        out = model(data['x'].to(DEVICE), data['edge_index'].to(DEVICE),data['batch'].to(DEVICE))  # Perform a single forward pass.
        loss = criterion(out, data['y'].to(DEVICE))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data['y'].to(DEVICE)).sum())  # Check against ground-truth labels.

        results['loss'].append(loss)
        results['accuracy'].append(correct / BATCH_SIZE)
        # logger.warning(f'Batch{i}: {results}')

    return results

def test(model,test_loader):
    model.eval()
    correct = 0
    results={'accuracy':[]}

    for data in test_loader:  # Iterate in batches over the training/test dataset.
    # data.to(DEVICE)
        out = model(data['x'].to(DEVICE), data['edge_index'].to(DEVICE),data['batch'].to(DEVICE))  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data['y'].to(DEVICE)).sum())  # Check against ground-truth labels.
        accuracy = correct / len(test_loader.dataset)  # Derive ratio of correct predictions.

        results['accuracy'].append(accuracy)
    # logger.warning(f'Test Results : {results}')
    return results  # Derive ratio of correct predictions.



def print_graph_stats(dataset):

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')
    # Gather some statistics about the first graph.
    print('FOR THE FIRST GRAPH:')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

if __name__=='__main__':
    main()