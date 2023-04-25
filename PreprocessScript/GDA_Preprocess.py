import gc
import os
import requests
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero, HANConv

from sklearn.model_selection import train_test_split

# os.chdir("E:\\xiong\\Bioinfomatics\\DriverGenes\\WorkingSpace\\MyDriveGenes")
os.chdir("/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")



class HAN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=128, heads=8):
        super(HAN, self).__init__()
        self.han_conv = HANConv(in_channels=in_channels, out_channels=hidden_size, heads=heads, dropout=0.6,
                                metadata=data.metadata())
        self.lin = nn.Linear(in_features=hidden_size, out_features=out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['gene'])
        return out


def get_GDAs(disease_id, source='CURATED', disease_type='disease', disease_class='C04'):
    """
    Use the DisGeNET REST API with the new authentication system to download gene-disease associations(GDAs).
    :param disease_id:
    :param source:
    :param disease_type:
    :param disease_class:
    :return:
    """
    # Build a dict with the following format, change the value of the two keys your DisGeNET account credentials, if you don't have an account you can create one here https://www.disgenet.org/signup/
    auth_params = {"email": "mrjohncuit@gmail.com", "password": "xxx123456789"}

    api_host = "https://www.disgenet.org/api"

    api_key = None
    s = requests.Session()

    gda_df = None

    try:
        r = s.post(api_host + '/auth/', data=auth_params, verify=False)
        if (r.status_code == 200):
            # Lets store the api key in a new variable and use it again in new requests
            json_response = r.json()
            api_key = json_response.get("token")
        else:
            print(r.status_code)
            print(r.text)
    except requests.exceptions.RequestException as req_ex:
        print(req_ex)
        print("Something went wrong with the request.")

    if api_key:
        # Add the api key to the requests headers of the requests Session object in order to use the restricted endpoints.
        s.headers.update({"Authorization": "Bearer %s" % api_key})
        # Lets get all the diseases associated to a gene eg. APP (EntrezID 351) and restricted by a source.
        gda_response = s.get(api_host + '/gda/disease/' + disease_id,
                             params={'source': source, 'type': disease_type, 'disease_class': disease_class})
        gda_df = pd.json_normalize(gda_response.json())
        gda_df = gda_df[columns_keep]
        gda_df.to_csv('')
        print("Get {} associations".format(gda_df.shape[0]))

    if s:
        s.close()

    return gda_df


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['gene'].train_mask
    loss = F.cross_entropy(out[mask], data['gene'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


def count_gene_in_positive_sample():
    """
    计算获取的到基因有多少个
    其中有多少基因被标记为正样本
    存储被标记的和没有被标记的
    :return:
    """
    pancan_genelist_for_train = pd.read_csv('Data\\pancan_genelist_for_train.tsv', sep='\t')
    positive_samples = pancan_genelist_for_train[pancan_genelist_for_train['Label'] == 1]
    positive_samples = positive_samples['Hugosymbol'].values
    hit = []
    miss = []




@torch.no_grad()
def run_test():
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    # for split in ['train_mask', 'val_mask', 'test_mask']:
    for split in ['train_mask', 'test_mask']:
        mask = data['gene'][split]
        acc = (pred[mask] == data['gene'].y[mask]).sum() / mask.sum()
        accs.append(acc)

    return accs


if __name__ == '__main__':
    # Get the gene-cancer associations for given cancers.
    # 'CESC': 'C0279672' -- 这个暂时没找到
    CANCER_DICT = {'BRCA': 'C0678222', 'LUAD': 'C0152013', 'HNSC': 'C3887461', 'THCA': 'C0549473', 'LUSC': 'C0149782', 'PRAD': 'C0007112', 'COAD': 'C0338106', 'KIRC': 'C0334488',
                   'STAD': 'C0278701', 'BLCA': 'C0279680', 'LIHC': 'C2239176', 'KIRP': 'C1306837', 'ESCA': 'C0152018', 'UCEC': 'C0206687', 'READ': 'C0149978'}
    # columns_keep = ['geneid', 'gene_symbol', 'uniprotid', 'gene_dsi', 'gene_dpi', 'gene_pli', 'diseaseid', 'score']
    columns_keep = ['geneid', 'diseaseid', 'score']
    all_dgas_df = pd.DataFrame(data=None, columns=columns_keep)
    for cancer in CANCER_DICT.keys():
        # if cancer != 'CESC':
        #     continue
        print("Downloading GDAs for {}".format(cancer))
        gda_df = get_GDAs(CANCER_DICT[cancer])
        all_dgas_df = pd.concat([all_dgas_df, gda_df])
    all_dgas_df = all_dgas_df.sort_values(by='geneid', ignore_index=True)


    # Create a mapping from unique gene indices to range [0, num_gene_nodes]
    unique_gene_id = all_dgas_df['geneid'].unique()
    unique_gene_id = pd.DataFrame(data={
        'geneid': unique_gene_id,
        'mappedID': pd.RangeIndex(len(unique_gene_id)),
    })

    # Create a mapping from unique cancer indices to range [0, num_cancer_nodes]
    unique_cancer_id = all_dgas_df['diseaseid'].unique()
    unique_cancer_id = pd.DataFrame(data={
        'diseaseid': unique_cancer_id,
        'mappedID': pd.RangeIndex(len(unique_cancer_id)),
    })

    # Perform merge to obtain the edges from genes and cancers:
    score_gene_id = pd.merge(left=all_dgas_df['geneid'], right=unique_gene_id, left_on='geneid', right_on='geneid',
                             how='left', )
    score_gene_id = torch.from_numpy(score_gene_id['mappedID'].values)
    score_cancer_id = pd.merge(left=all_dgas_df['diseaseid'], right=unique_cancer_id, left_on='diseaseid',
                               right_on='diseaseid', how='left')
    score_cancer_id = torch.from_numpy(score_cancer_id['mappedID'].values)

    # With this, we are ready to construct our `edge_index` in COO format
    # following PyG semantics:
    edge_index_gene_to_cancer = torch.stack([score_gene_id, score_cancer_id])

    # Construct HeteroData
    data = HeteroData()
    # Save node indices:
    data["gene"].node_id = torch.arange(len(unique_gene_id))
    data["cancer"].node_id = torch.arange(len(unique_cancer_id))
    data["gene"].x = torch.randn(size=(len(unique_gene_id), 48))
    data["gene"].y = torch.randint(low=0, high=2, size=(len(unique_gene_id),))
    # 划分训练集和测试集的要重写
    train_index, test_index = train_test_split(np.arange(len(unique_gene_id)), test_size=0.2, random_state=42)
    train_mask = torch.zeros(len(unique_gene_id))
    train_mask[train_index] = 1
    train_mask = train_mask >= 1
    test_mask = torch.zeros(len(unique_gene_id))
    test_mask[test_index] = 1
    test_mask = test_mask >= 1
    data['gene'].train_mask = train_mask
    data['gene'].test_mask = test_mask
    data['cancer'].x = torch.randn(size=(len(unique_cancer_id), 48))
    data["gene", "score", "cancer"].edge_index = edge_index_gene_to_cancer
    data = T.ToUndirected()(data)

    best_val_acc = 0
    start_patience = patience = 100
    model = HAN(in_channels=-1, out_channels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, model = data.to(device), model.to(device)

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=5e-4)

    for epoch in range(1, 2000):
        loss = train()
        # train_acc, val_acc, test_acc = run_test()
        train_acc, test_acc = run_test()
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
        #       f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Test: {test_acc:.4f}')

        # if best_val_acc <= test_acc:
        #     patience = start_patience
        #     best_val_acc = test_acc
        # else:
        #     patience -= 1
        #
        # if patience <= 0:
        #     print('Stopping training as validation accuracy did not improve '
        #           f'for {start_patience} epochs')
        #     break

