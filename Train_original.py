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
from Models.MyModel import MyModel
from torch_geometric.utils import from_scipy_sparse_matrix


def train(mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    pred = np.round(
        torch.sigmoid(out[data['gene'].label_index[mask]]).cpu().detach().numpy())
    # Attention: Use the mask to get the label index, then use the label index to get the genes used to train
    # loss = F.cross_entropy(out[data['gene'].label_index[mask]], data['gene'].y[mask])
    # loss = F.binary_cross_entropy_with_logits(input=out[data['gene'].label_index[mask]].squeeze(),
    #                                           target=data['gene'].y[mask])
    loss = criterion(out[data['gene'].label_index[mask]].squeeze(), data['gene'].y[mask])
    ACC = metrics.accuracy_score(y_true=data['gene'].y[mask].cpu(), y_pred=pred)
    F1 = metrics.f1_score(y_true=data['gene'].y[mask].cpu(), y_pred=pred)
    AUROC = metrics.roc_auc_score(y_true=data['gene'].y[mask].cpu(), y_score=pred)
    precision, recall, _ = metrics.precision_recall_curve(data['gene'].y[mask].cpu(), pred)
    AUPR = metrics.auc(recall, precision)
    loss.backward()
    optimizer.step()
    return ACC, F1, AUROC, AUPR, loss.item()


@torch.no_grad()
def func_test(mask):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    pred = np.round(
        torch.sigmoid(out[data['gene'].label_index[mask]]).cpu().detach().numpy())
    # loss = F.binary_cross_entropy_with_logits(input=out[data['gene'].label_index[mask]].squeeze(),
    #                                           target=data['gene'].y[mask])
    loss = criterion(out[data['gene'].label_index[mask]].squeeze(), data['gene'].y[mask])
    ACC = metrics.accuracy_score(y_true=data['gene'].y[mask].cpu(), y_pred=pred)
    F1 = metrics.f1_score(y_true=data['gene'].y[mask].cpu(), y_pred=pred)
    AUROC = metrics.roc_auc_score(y_true=data['gene'].y[mask].cpu(), y_score=pred)
    precision, recall, _ = metrics.precision_recall_curve(data['gene'].y[mask].cpu(), pred)
    AUPR = metrics.auc(recall, precision)

    return pred, ACC, F1, AUROC, AUPR, loss.item()


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
    protein_to_gene = data["protein", "to", "gene"].edge_index
    new_p_to_g = torch.empty(size=(2, 0), dtype=torch.int64)
    for i in range(protein_to_gene.shape[1]):
        if protein_to_gene[1, i].item() in idx_dict:
            temp = torch.tensor([[protein_to_gene[0, i]], [idx_dict[protein_to_gene[1, i].item()]]], dtype=torch.int64)
            new_p_to_g = torch.concat([new_p_to_g, temp], dim=1)

    try:
        data['gene'].x = data['gene'].x[idx_nondrop]
        data['gene'].y = torch.tensor(new_labels)
        data['gene'].label_index = torch.tensor(new_label_index)
        data['gene', 'to', 'gene'].edge_index = edge_index
        data["protein", "to", "gene"].edge_index = new_p_to_g
        data["gene", "rev_to", "protein"].edge_index = torch.concat(
            [data["protein", "to", "gene"].edge_index[1, :].reshape(1, -1),
             data["protein", "to", "gene"].edge_index[0, :].reshape(1, -1)], dim=0)

    except:
        data = data

    protein_node_num, _ = data['protein'].x.size()
    _, protein_edge_num = data['protein', 'to', 'protein'].edge_index.size()
    drop_num = int(protein_node_num * aug_ratio)

    # Get the nodes to drop and keep
    idx_perm = np.random.permutation(protein_node_num)
    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()

    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}
    protein_edge_index = data['protein', 'to', 'protein'].edge_index.numpy()

    # Get the new adj
    protein_adj = torch.zeros((protein_node_num, protein_node_num))
    protein_adj[protein_edge_index[0], protein_edge_index[1]] = 1
    gene_adj = protein_adj[idx_nondrop, :][:, idx_nondrop]
    # edge_index = from_scipy_sparse_matrix(sp.coo_matrix(gene_adj))[0]
    edge_index = gene_adj.nonzero().t()

    gene_to_protein = data["gene", "rev_to", "protein"].edge_index
    new_g_to_p = torch.empty(size=(2, 0), dtype=torch.int64)
    for i in range(gene_to_protein.shape[1]):
        if gene_to_protein[1, i].item() in idx_dict:
            temp = torch.tensor([[gene_to_protein[0, i]], [idx_dict[gene_to_protein[1, i].item()]]], dtype=torch.int64)
            new_g_to_p = torch.concat([new_g_to_p, temp], dim=1)
    try:
        data['protein'].x = data['protein'].x[idx_nondrop]

        data['protein', 'to', 'protein'].edge_index = edge_index
        data["gene", "rev_to", "protein"].edge_index = new_g_to_p
        data["protein", "to", "gene"].edge_index = torch.concat(
            [data["gene", "rev_to", "protein"].edge_index[1, :].reshape(1, -1),
             data["gene", "rev_to", "protein"].edge_index[0, :].reshape(1, -1)], dim=0)

    except:
        data = data
    return data


def mask_nodes(data, aug_ratio):
    node_num, feat_dim = data["gene"].x.size()
    mask_num = int(node_num * aug_ratio)

    token = data["gene"].x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    # Use the mean value to replace the true feature
    # What about using 0 ???
    data["gene"].x[idx_mask] = token

    node_num, feat_dim = data["protein"].x.size()
    mask_num = int(node_num * aug_ratio)

    token = data["protein"].x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data["protein"].x[idx_mask] = token
    return data


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HGDataset(root="/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")
    data = dataset[0]

    folds = 5
    repeats = 1

    ACC = np.zeros(shape=(repeats, folds))
    F1 = np.zeros(shape=(repeats, folds))
    ROC = np.zeros(shape=(repeats, folds))
    PR = np.zeros(shape=(repeats, folds))

    kf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
    splits = kf.split(data['gene'].label_index, data['gene'].y)
    pbar = tqdm(enumerate(splits))
    for i, (train_mask, val_mask) in enumerate(splits):
        # model = HAN(metadata=data.metadata(), in_channels=-1, out_channels=1)
        # model = MyModel(metadata=data.metadata(), in_channels=-1, out_channels=1)
        model = HGT(hidden_channels=64, out_channels=1, num_heads=4, num_layers=2, data=data)
        data, model = data.to(device), model.to(device)

        max_pr = 0
        max_roc = 0

        with torch.no_grad():  # Initialize lazy modules.
            out = model(data.x_dict, data.edge_index_dict)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2.7]).to(device))
        # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2.7]))

        for epoch in range(2000):
            # pbar.set_description("Round:{}, Fold:{}, Epoch:{}".format((i // 5) + 1, (i % 5) + 1, epoch))
            # pbar.set_description("Round:{}, Fold:{}, Epoch:{}, Train_ACC:{}, Train_Loss:{}, Test_ACC:{}, Test_Loss:{}".format(
            #     (i // 5) + 1, (i % 5) + 1, epoch, train_acc, train_loss, test_acc, test_loss))
            train_ACC, train_F1, train_AUROC, train_AUPR, train_loss = train(mask=train_mask)
            test_pred, test_ACC, test_F1, test_AUROC, test_AUPR, test_loss = func_test(mask=val_mask)
            # Save the best model
            if test_AUPR > max_pr:
                max_pr = test_AUPR
                # torch.save(obj=model, f=save)
            # pbar.set_postfix_str("Train_ACC:{}, Train_Loss:{}, Test_ACC:{}, Test_Loss:{}".format(train_acc, train_loss, test_acc, test_loss))
            print("Round:{}, Fold:{}, Epoch:{}, Test_ACC:{}, Test_F1:{}, Test_AUPR:{}, Test_AUROC:{}, Test_Loss:{}".format(
                (i // 5) + 1, (i % 5) + 1, epoch, np.round(test_ACC, 4), np.round(test_F1, 4), np.round(test_AUPR, 4), np.round(test_AUROC, 4), np.round(test_loss, 4)))



        # Test
        # _, test_acc, test_auroc, test_aupr, test_loss = func_test(mask=val_mask)
        # Train_ACC[i, (i % 5) + 1] = train_acc

        # break
