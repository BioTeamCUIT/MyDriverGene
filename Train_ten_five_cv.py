import os
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp


import torch
import torch.nn.functional as F

from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold

from Utils.HGDataset import HGDataset

from Models.HAN import HAN
from Models.HGT import HGT

from torch_geometric.utils import from_scipy_sparse_matrix


def train(mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    pred = np.round(
        torch.sigmoid(out[data['gene'].label_index[mask]]).cpu().detach().numpy())
    # Attention: Use the mask to get the label index, then use the label index to get the genes used to train
    # loss = F.cross_entropy(out[data['gene'].label_index[mask]], data['gene'].y[mask])
    loss = F.binary_cross_entropy_with_logits(input=out[data['gene'].label_index[mask]].squeeze(),
                                              target=data['gene'].y[mask])
    acc = metrics.accuracy_score(y_true=data['gene'].y[mask].cpu(), y_pred=pred)
    loss.backward()
    optimizer.step()
    return acc, loss.item()


@torch.no_grad()
def func_test(mask):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    pred = np.round(
        torch.sigmoid(out[data['gene'].label_index[mask]]).cpu().detach().numpy())
    loss = F.binary_cross_entropy_with_logits(input=out[data['gene'].label_index[mask]].squeeze(),
                                              target=data['gene'].y[mask])
    acc = metrics.accuracy_score(y_true=data['gene'].y[mask].cpu(), y_pred=pred)
    auroc = metrics.roc_auc_score(data['gene'].y[mask].cpu(), pred)
    pr, rec, _ = metrics.precision_recall_curve(data['gene'].y[mask].cpu(), pred)
    aupr = metrics.auc(rec, pr)

    return pred, acc, auroc, aupr, loss.item()


def drop_nodes(data, aug_ratio):
    gene_node_num, _ = data['gene'].x.size()
    _, gene_edge_num = data['gene', 'to', 'gene'].edge_index.size()
    drop_num = int(gene_node_num * aug_ratio)

    # Get the nodes to drop and keep
    idx_perm = np.random.permutation(gene_node_num)
    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()

    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}
    gene_edge_index = data['gene', 'to', 'gene'].edge_index.numpy()

    # Get the new adj
    gene_adj = torch.zeros((gene_node_num, gene_node_num))
    gene_adj[gene_edge_index[0], gene_edge_index[1]] = 1
    gene_adj = gene_adj[idx_nondrop, :][:, idx_nondrop]
    # edge_index = from_scipy_sparse_matrix(sp.coo_matrix(gene_adj))[0]
    edge_index = gene_adj.nonzero().t()

    # # Remove the labels if the nodes were dropped
    # label_index = data['gene'].label_index[~np.isin(data['gene'].label_index, torch.tensor(idx_drop))]
    # # Re-index the label index
    # label_index_np = label_index.numpy()
    # for index, item in enumerate(label_index_np):
    #     if item in idx_dict:
    #         label_index_np[index] = idx_dict[item]
    # label_index = torch.tensor(label_index_np)
    # data['gene'].label_index = label_index
    labels = data['gene'].y.numpy()
    label_index_np = data['gene'].label_index.numpy()
    new_labels = []
    new_label_index = []
    for index, item in enumerate(label_index_np):
        if item in idx_dict:
            new_labels.append(labels[index])
            new_label_index.append(idx_dict[item])
    try:
        data['gene'].x = data['gene'].x[idx_nondrop]
        data['gene'].y = torch.tensor(new_labels)
        data['gene'].label_index = torch.tensor(new_label_index)
        data['gene', 'to', 'gene'].edge_index = edge_index
    except:
        data = data
    return data


if __name__ == '__main__':
    dataset = HGDataset(root="E:\\xiong\\Bioinfomatics\\DriverGenes\\WorkingSpace\\MyDriveGenes")
    data = dataset[0]
    data1 = drop_nodes(data, 0.2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folds = 2
    repeats = 2

    # AUC = np.zeros(shape=(10, 5))
    # AUPR = np.zeros(shape=(10, 5))
    # Train_ACC = np.zeros(shape=(repeats, folds))
    # Test_ACC = np.zeros(shape=(repeats, folds))

    kf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
    splits = kf.split(data['gene'].label_index, data['gene'].y)
    pbar = tqdm(enumerate(splits))
    for i, (train_mask, val_mask) in pbar:
        model = HAN(metadata=data.metadata(), in_channels=-1, out_channels=1)
        # model = HGT(hidden_channels=64, out_channels=1, num_heads=2, num_layers=1, data=data)
        data, model = data.to(device), model.to(device)

        with torch.no_grad():  # Initialize lazy modules.
            out = model(data.x_dict, data.edge_index_dict)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=5e-4)

        for epoch in range(2000):
            # pbar.set_description("Round:{}, Fold:{}, Epoch:{}".format((i // 5) + 1, (i % 5) + 1, epoch))
            # pbar.set_description("Round:{}, Fold:{}, Epoch:{}, Train_ACC:{}, Train_Loss:{}, Test_ACC:{}, Test_Loss:{}".format(
            #     (i // 5) + 1, (i % 5) + 1, epoch, train_acc, train_loss, test_acc, test_loss))
            train_acc, train_loss = train(mask=train_mask)
            _, test_acc, test_auroc, test_aupr, test_loss = func_test(mask=val_mask)
            # pbar.set_postfix_str("Train_ACC:{}, Train_Loss:{}, Test_ACC:{}, Test_Loss:{}".format(train_acc, train_loss, test_acc, test_loss))
            print("Round:{}, Fold:{}, Epoch:{}, Train_ACC:{}, Train_Loss:{}, Test_ACC:{}, Test_Loss:{}".format(
                (i // 5) + 1, (i % 5) + 1, epoch, train_acc, train_loss, test_acc, test_loss))
        # Test
        # _, test_acc, test_auroc, test_aupr, test_loss = func_test(mask=val_mask)
        # Train_ACC[i, (i % 5) + 1] = train_acc

        # break
