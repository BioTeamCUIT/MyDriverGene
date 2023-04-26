import os
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold

from Utils.HGDataset import HGDataset
from Utils.utils import *

from Models.HAN import HAN
from Models.HGT import HGT
from Models.MyModel import MyModel
from torch_geometric.utils import from_scipy_sparse_matrix

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.chdir("/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")


class Trainer:
    def __init__(self, model_name, epochs):
        super(Trainer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.hidden_channels = 64
        self.out_channels = 1
        self.num_heads = 4
        self.num_layers = 3
        self.lr = 0.001
        self.weight_decay = 5e-4
        # self.model = model.to(device=self.device)
        self.model = None
        self.model_name = model_name
        # self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001, weight_decay=5e-4)
        self.optimizer = None
        # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2.7]).to(self.device))
        self.criterion = None
        self.epochs = epochs
        self.repeats = 1
        self.folds = 5


    def inference(self, data, mask, fold):
        path = os.path.join("./SavedModels", self.model_name, "fold_{}.pth".format(fold))
        self.model = torch.load(f=path, map_location='cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        out = self.model(data.x_dict, data.edge_index_dict)
        pred = np.round(
            torch.sigmoid(out[data['gene'].label_index[mask]]).cpu().detach().numpy())
        labels = data['gene'].y[mask].cpu()
        ACC, F1, AUROC, AUPR = self.measure(pred, labels)
        return pred, ACC, F1, AUROC, AUPR

    def measure(self, pred, labels):
        ACC = metrics.accuracy_score(y_true=labels, y_pred=pred)
        F1 = metrics.f1_score(y_true=labels, y_pred=pred)
        AUROC = metrics.roc_auc_score(y_true=labels, y_score=pred)
        precision, recall, _ = metrics.precision_recall_curve(labels, pred)
        AUPR = metrics.auc(recall, precision)
        return ACC, F1, AUROC, AUPR

    def save_evaluation_indicators(self, indicators):
        path = os.path.join("SavedIndicators")

        if not os.path.exists(path):
            os.makedirs(path)
        #     写入评价指标
        file_name = os.path.join(path, "{}_KEGG_PPI_Meth.xlsx".format(self.model_name))
        file = open(file_name, "a")

        file.write(str(np.round(indicators[0], 4)) + " " + str(np.round(indicators[1], 4)) + " " +
                   str(np.round(indicators[2], 4)) + " " + str(np.round(indicators[3], 4)) + "\n")

        file.close()

    def run(self, data):

        data = data.to(self.device)
        # Set other features to 0, except for Expr
        index = [i*3 for i in np.arange(16)]
        index = index + [i*3+2 for i in np.arange(16)]
        index.sort()
        data['gene'].x[:, index] = 0

        for fold in range(self.folds):
            test_pred, test_ACC, test_F1, test_AUROC, test_AUPR = self.inference(data=data,
                                                                                 mask=np.arange(data['gene'].y.shape[0]),
                                                                                 fold=fold)
            # Save the indicators
            indicators = [test_ACC, test_F1, test_AUROC, test_AUPR]
            self.save_evaluation_indicators(indicators)


if __name__ == '__main__':
    dataset = HGDataset(root="/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")
    data = dataset[0]

    model_name = "BEST"
    # model_name = 'HAN'
    trainer = Trainer(model_name=model_name, epochs=1000)
    trainer.run(data=data)
