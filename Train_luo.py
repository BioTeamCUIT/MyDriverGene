import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.metrics import f1_score as F1
# from Utils.HGDataset import HGDataset

from Models.HAN import HAN
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号无法正常显示的问题

def train_step(mask,model):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    pred = np.round(
        torch.sigmoid(out[data['gene'].label_index[mask]]).cpu().detach().numpy())
    # Attention: Use the mask to get the label index, then use the label index to get the genes used to train
    # loss = F.cross_entropy(out[data['gene'].label_index[mask]], data['gene'].y[mask])
    loss = F.binary_cross_entropy_with_logits(input=out[data['gene'].label_index[mask]].squeeze(), target=data['gene'].y[mask])
    acc = metrics.accuracy_score(y_true=data['gene'].y[mask].cpu(), y_pred=pred)
    loss.backward()
    optimizer.step()
    return acc, loss.item()


@torch.no_grad()
def func_test(mask,model):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    pred = np.round(
        torch.sigmoid(out[data['gene'].label_index[mask]]).cpu().detach().numpy())
    loss = F.binary_cross_entropy_with_logits(input=out[data['gene'].label_index[mask]].squeeze(), target=data['gene'].y[mask])
    acc = metrics.accuracy_score(y_true=data['gene'].y[mask].cpu(), y_pred=pred)
    auroc = metrics.roc_auc_score(data['gene'].y[mask].cpu(), pred)
    pr, rec, _ = metrics.precision_recall_curve(data['gene'].y[mask].cpu(), pred)
    aupr = metrics.auc(rec, pr)
    f1 = F1(y_true=data['gene'].y[mask].cpu(), y_pred=pred)

    return pred,f1, acc, auroc, aupr, loss.item()

def normalize(x):
    assert len(x.shape)==2
    for i in range(x.shape[1]):
        min_value = min(x[:,i])
        max_value = max(x[:,i])
        gap =max_value -min_value
        x[:,i]=(x[:,i]-min_value)/gap
    return x

def drop_gene_nodes(data, aug_ratio):
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
    edge_index = gene_adj.nonzero().t()


    labels = data['gene'].y.numpy()
    label_index_np = data['gene'].label_index.numpy()
    new_labels = []
    new_label_index = []
    for index, item in enumerate(label_index_np):
        if item in idx_dict:
            new_labels.append(labels[index])
            new_label_index.append(idx_dict[item])
    protein_to_gene =data["protein","to","gene"].edge_index
    new_p_to_g = torch.empty(size=(2,0),dtype=torch.int64)
    for i in range(protein_to_gene.shape[1]):
        if protein_to_gene[1, i].item() in idx_dict:
            temp = torch.tensor([[protein_to_gene[0, i]], [idx_dict[protein_to_gene[1, i].item()]]], dtype=torch.int64)
            new_p_to_g = torch.concat([new_p_to_g, temp], dim=1)

    try:
        data['gene'].x = data['gene'].x[idx_nondrop]
        data['gene'].y = torch.tensor(new_labels)
        data['gene'].label_index = torch.tensor(new_label_index)
        data['gene', 'to', 'gene'].edge_index = edge_index
        data["protein","to","gene"].edge_index = new_p_to_g
        data["gene", "rev_to", "protein"].edge_index = torch.concat(
            [data["protein","to","gene"].edge_index[1,:].reshape(1,-1),
              data["protein","to","gene"].edge_index[0,:].reshape(1,-1)],dim = 0)

    except:
        data = data
    return data
def drop_protein_nodes(data, aug_ratio):
    protein_node_num, _ = data['protein'].x.size()
    _, gene_edge_num = data['protein', 'to', 'protein'].edge_index.size()
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

    gene_to_protein =data["gene","rev_to","protein"].edge_index
    new_g_to_p = torch.empty(size=(2,0),dtype=torch.int64)
    for i in range(gene_to_protein.shape[1]):
        if gene_to_protein[1, i].item() in idx_dict:
            temp = torch.tensor([[gene_to_protein[0, i]], [idx_dict[gene_to_protein[1, i].item()]]], dtype=torch.int64)
            new_g_to_p = torch.concat([new_g_to_p, temp], dim=1)

    try:
        data['protein'].x = data['protein'].x[idx_nondrop]

        data['protein', 'to', 'protein'].edge_index = edge_index
        data["gene", "rev_to", "protein"].edge_index = new_g_to_p
        data["protein","to","gene"].edge_index = torch.concat(
            [data["gene", "rev_to", "protein"].edge_index[1,:].reshape(1,-1),
              data["gene", "rev_to", "protein"].edge_index[0,:].reshape(1,-1)],dim = 0)

    except:
        data = data
    return data
def mask_nodes(data, P_aug_ratio,G_aug_ratio):

    node_num, feat_dim = data["gene"].x.size()
    mask_num = int(node_num * G_aug_ratio)

    token = data["gene"].x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data["gene"].x[idx_mask] = token

    node_num, feat_dim = data["protein"].x.size()
    mask_num = int(node_num * P_aug_ratio)

    token = data["protein"].x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data["protein"].x[idx_mask] = token
    return data

def train_and_test(data,model,max_epoch,dict,early_stopping = 10,print_interval=20,model_name = "HAN",normal = False):
    min_loss =100

    val_loss_list = []
    for epoch in range(max_epoch):
        train_acc,train_loss=train_step(data["gene"].train_mask,model)
        _,val_f1,val_acc, val_auroc, val_aupr, val_loss = func_test(data["gene"].val_mask,model)
        dict["train_loss"].append(train_loss)
        dict["train_acc"].append(train_acc)
        dict["val_loss"].append(val_loss)
        dict["val_acc"].append(val_acc)
        dict["val_auroc"].append(val_auroc)
        dict["val_aupr"].append(val_aupr)
        if val_loss <min_loss:
            min_loss = val_loss
            if normal:
                torch.save(model,f"result/model/{model_name}_normal_{lr}best.pt")
            else:
                torch.save(model, f"result/model/{model_name}_{lr}best.pt")
        val_loss_list.append(val_loss)
        # if epoch > early_stopping and val_loss > np.mean(val_loss_list[-(early_stopping + 1): -1]):
        #     print("\nEarly stopping...")
        #     break
        if  epoch % print_interval == 0:
            print(f"\nEpoch: {epoch}\n----------")
            print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
            print(f"  Val loss: {val_loss:.4f} |   Val acc: {val_acc:.4f}")
            print(f"  Val auroc: {val_auroc:.4f} |   Val aupr: {val_aupr:.4f}")
    if normal:
        model = torch.load(f"result/model/{model_name}_normal_{lr}best.pt")
    else:
        model = torch.load(f"result/model/{model_name}_{lr}best.pt")
    train_acc, train_loss = train_step(data["gene"].train_mask, model)
    _, val_f1, val_acc, val_auroc, val_aupr, val_loss = func_test(data["gene"].val_mask, model)
    _,test_f1,test_acc,test_auroc,test_aupr,test_loss = func_test(data["gene"].test_mask,model)
    print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
    print(f"  Val loss: {val_loss:.4f} |   Val acc: {val_acc:.4f}")
    print(f"  Val auroc: {val_auroc:.4f} |   Val aupr: {val_aupr:.4f}")
    print(f"  test loss: {test_loss:.4f} |   Val acc: {test_acc:.4f}")
    print(f"  test auroc: {test_auroc:.4f} |   test aupr: {test_aupr:.4f}")
    result = {"train_loss": [train_loss],
            "train_acc": [train_acc],
            "val_loss": [val_loss],
            "val_f1":[val_f1],
            "val_acc": [val_acc],
            "val_auroc": [val_auroc],
            "val_aupr": [val_aupr],
            "test_loss": [test_loss],
            "test_f1": [test_f1],
            "test_acc": [test_acc],
            "test_auroc":[ test_auroc],
            "test_aupr": [test_aupr],
            }
    df =pd.DataFrame.from_dict(result)
    if normal:
        df.to_excel(f"result/{model_name}_normal_{lr}.xlsx")
    else:
        df.to_excel(f"result/{model_name}_{lr}.xlsx")
    return dict

if __name__ == '__main__':

    dict = {"train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auroc": [],
            "val_aupr": [],
            "test_loss": [],
            "test_acc": [],
            "test_auroc": [],
            "test_aupr": [],
            }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda' )


    for i in[0.001,0.0001,0.0005]:
        for j in[True,False]:

            data = torch.load("cut_data.pt")
            model = HAN(metadata=data.metadata(), in_channels=-1, out_channels=1)
            model.to(device)
            data.to(device)
            lr =i
            normal = j
            if normal:
                data["gene"].x = normalize(data["gene"].x)
                data["protein"].x = normalize(data["protein"].x)
            with torch.no_grad():  # Initialize lazy modules.
              out = model(data.x_dict, data.edge_index_dict)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=5e-4)
            train_and_test(data, model=model, max_epoch=2000, dict=dict,normal=normal,model_name="HAN")
